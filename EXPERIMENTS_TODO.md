# Remaining real experiments — what to run, and where

This is the honest punch list of experiments the reviewers asked for that
are not yet actually run. Nothing here is fabricated or simulated — every
script issues real API/GPU calls. Split into what needs a rented GPU
(RunPod) vs. what only needs a laptop + the hosted NIM API key you already
have (`VLLM_API_KEY`).

## 1. MacBook-only (no GPU needed — uses the hosted NIM API)

These two only call `https://integrate.api.nvidia.com/v1/chat/completions`
for generation; the "pruning"/"compression" step runs locally on CPU
(cross-encoder and LLMLingua-2 are both small enough to run on a MacBook).

### 1a. Token-budget-matched naive baseline — `scripts/run_arcd_token_budget.py`
Fixes the fairness gap the reviewers flagged: the original naive baseline
was matched to LSPM by *sentence-count ratio*, not by *token count*, so the
two methods could end up at different KV-cache costs while claiming to be
"at the same ratio." This reruns naive truncation matched to LSPM's actual
token count instead.

```
cd vllm-arabic-rag
pip install transformers
export VLLM_API_KEY=...            # your existing NIM key
export TOKENIZER_NAME=meta-llama/Llama-3.1-8B-Instruct   # or NousResearch mirror if gated repo is a hassle
python scripts/run_arcd_token_budget.py
```
Needs `scripts/run_arcd_pilot.py` to have been run first (reuses its output).
Output: `data/arcd_token_budget_results.jsonl`.

### 1b. LLMLingua-2 baseline — `scripts/run_arcd_llmlingua2_baseline.py`
Adds a real modern compression-method comparison (Pan et al., ACL 2024
Findings), which both reviewers explicitly asked for.

```
pip install llmlingua
export VLLM_API_KEY=...
python scripts/run_arcd_llmlingua2_baseline.py
```
Output: `data/arcd_llmlingua2_results.jsonl`.

**Honest caveat to keep in the paper:** LLMLingua-2's public checkpoints
were trained on English meeting-summarization data, not Arabic RAG QA —
report this as a limitation, don't hide it. Its token-level compression can
also produce disfluent Arabic; that's expected behavior, not a bug.

After both finish, extend `scripts/analyze_arcd_results.py` to also load
`method == "naive_tokbudget"` and `method == "llmlingua2"` alongside the
existing `raw`/`lspm`/`naive` conditions and report EM/F1 for all of them.

## 2. RunPod (GPU required)

### 2a. Corrected GPU benchmark sweep — already fully built, just needs to run
This is **not new code** — `benchmark/run_full_sweep.py` (with
`precompute_contexts.py` and `analyze_sweep_results.py --repeats N`) already
implements the randomized/repeated-cell redesign that answers the
reviewers' concerns about the original benchmark's confounds (client-side
scoring cost, fixed sequential block ordering). Full commands are already
in `benchmark/BENCHMARK_INSTRUCTIONS.md`:

```
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --max-model-len 8192
pip install -r benchmark/requirements-benchmark.txt
python benchmark/precompute_contexts.py
python benchmark/run_full_sweep.py --host http://localhost:8000 --run-time 90 --repeats 3 --out-dir results/sweep_v2
python benchmark/analyze_sweep_results.py --sweep-dir results/sweep_v2 --repeats 3
```
Estimated cost: ~3 hours of GPU rental at the defaults.

### 2b. Dynamic-controller ablation (NEW) — `benchmark/run_controller_ablation.py`
Answers the reviewers' complaint that `DynamicRatioController` is described
but never actually evaluated against a fixed-ratio baseline. Drives a
ramping concurrency profile (low → high → low) against your self-hosted
vLLM server twice — once with the ratio held fixed at 0.5, once with the
controller live-polling `/metrics` and adjusting the ratio — and logs
KV-cache occupancy + chosen ratio + TTFT/latency per request.

```
# with the same vllm serve command as 2a already running:
python benchmark/run_controller_ablation.py --host http://localhost:8000 --out-dir results/controller_ablation
```
Output: `results/controller_ablation/{fixed,dynamic}_ratio_log.jsonl`. Plot
`kv_cache_occupancy_at_request` vs. `ratio` over time for the dynamic run
(should visibly track cache pressure) against the flat fixed-ratio line,
and compare TTFT/latency during the high-concurrency step between the two
conditions — that comparison is the actual ablation result.

Takes ~6 minutes per condition (`LOAD_PROFILE` in the script; extend the
step durations for a longer, more publication-grade run if GPU budget
allows).

## 3. Two SOTA baselines the reviewers named that do NOT have ready-made
   integration code — honest status, not fabricated

- **DAC** (Zhao et al.) — has a real public GitHub repo:
  `https://github.com/QQQ-yi/DAC`. Its exact inference API wasn't
  verified from search results, so no drop-in script is provided here —
  clone the repo and check its own README/examples before wiring it into
  `scripts/run_arcd_*` the same way the LLMLingua-2 script does.
- **ACC-RAG** (Guo, Zhang, Ren; ACL Findings EMNLP 2025, arXiv 2507.22931)
  — searched specifically for a code release; none was found (no GitHub
  link in the paper's search results, ACL Anthology page, or arXiv
  listing as of this writing). If you can't locate one either, the honest
  move in the paper is to cite it as a described-but-not-independently-
  reproduced comparison point (which the Discussion/Limitations section
  already does), not to claim a rerun that didn't happen.

## Summary table

| Experiment | Machine | New code? | Script |
|---|---|---|---|
| Token-budget-matched naive | MacBook | reused pattern, new script | `scripts/run_arcd_token_budget.py` |
| LLMLingua-2 baseline | MacBook | reused pattern, new script | `scripts/run_arcd_llmlingua2_baseline.py` |
| Corrected GPU benchmark sweep | RunPod | already existed | `benchmark/run_full_sweep.py` |
| Dynamic-controller ablation | RunPod | new script | `benchmark/run_controller_ablation.py` |
| DAC baseline | — | not provided (verify their API first) | n/a |
| ACC-RAG baseline | — | not provided (no code release found) | n/a |
