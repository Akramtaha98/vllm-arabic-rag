<div align="center">

# LSPM — Semantic-Driven Context Pruning for Arabic RAG

**Quality-preserving context compression for memory-constrained vLLM deployments**

Arabic's morphological richness means tokenizers routinely produce **1.5–2×
more tokens** than English for the same meaning. In a RAG pipeline that
inflates KV-cache pressure on every request. This repo is the reference
implementation, benchmark suite, and evaluation data behind the paper
*"Semantic-Driven Context Pruning for Arabic RAG: Quality-Preserving
Compression Under Token Inflation"* (Tarish & Zghair), submitted to
**MDPI *Information***.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![vLLM](https://img.shields.io/badge/inference-vLLM-6C4CF1)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

</div>

---

## What this is

**LSPM (Lightweight Semantic Pruning Middleware)** sits between retrieval and
generation in an Arabic RAG pipeline. It splits every retrieved passage into
sentences, scores each sentence against the query with a cross-encoder, keeps
only the top-scoring sentences up to a target compression ratio *r*, and
reassembles the survivors in their **original document order** before they
ever reach the generator's KV-cache. A companion **dynamic ratio controller**
can optionally read vLLM's live `/metrics` KV-cache occupancy and tighten or
relax *r* automatically as load changes.

This is not a new pruning algorithm — the split-score-reassemble mechanism is
the same one used by prior work such as DSLR. What this project adds and
validates is (1) a vLLM-serving-aware dynamic controller built on top of that
mechanism, and (2) an empirical evaluation of both, specifically on Arabic
RAG, where token inflation makes the problem more acute than in the
mostly-English settings such prior work targets.

## What's been measured (not just claimed)

Every number below comes from a script in this repo, run against real data —
see [Reproducing the results](#reproducing-the-results) for exact commands.

- **Accuracy under compression** — LSPM vs. naive length-matched truncation
  vs. LLMLingua-2, evaluated on 140 real questions from the Arabic Reading
  Comprehension Dataset (ARCD) against a noisy 6-document retrieval pool
  (1 gold passage + 5 distractors). LSPM's F1 advantage over naive truncation
  is statistically significant after Holm correction; a TOST equivalence test
  checks pruned-vs-unpruned answer fidelity. (`data/arcd_results.jsonl`,
  `results/arcd_stats_*`)
- **Generalization** — the same evaluation repeated on a second generator
  (Qwen2.5-7B-Instruct) and on a fresh, disjoint 140-question sample, to check
  the result isn't an artifact of one model or one sample.
  (`data/arcd_qwen25_results.jsonl`, `data/arcd_replication_results.jsonl`)
- **Serving cost** — a 105-run, 3-repeat GPU sweep across
  method × ratio × concurrency, measuring throughput, TTFT, tokens/sec, and
  KV-cache occupancy against a real vLLM server. In this benchmark LSPM stays
  within ~3% of raw (unpruned) throughput. This sweep reads every context
  from a precomputed lookup table to isolate vLLM-side serving cost from
  cross-encoder scoring cost (measured separately at a mean of ~506 ms/call);
  it is **not** an end-to-end online-serving number on its own — see
  [Known gaps](#known-gaps-being-actively-closed) below.
  (`results/sweep_summary_v2.csv`, `benchmark/run_full_sweep.py`)
- **Dynamic controller under memory pressure** — a redesigned ablation using
  real, long ARCD contexts against a deliberately constrained
  `--gpu-memory-utilization` setting drives KV-cache occupancy to 99.2% and
  confirms the controller's adaptive path engages correctly.
  (`results/controller_ablation_v2/`)

## Known gaps (being actively closed)

In the interest of not overclaiming: two follow-up experiments are designed,
committed, and pre-registered in this repo, but **not yet run** (they need
live GPU time):

- `benchmark/run_online_e2e_benchmark.py` — a true end-to-end serving
  benchmark that performs sentence splitting, cross-encoder scoring, and
  context reconstruction *inside* the timed request path, instead of reading
  from a precomputed lookup. This will report LSPM's real online-serving cost
  including the cross-encoder, not just vLLM-side serving cost.
- `scripts/run_controller_vs_matched_fixed.py` — compares the dynamic
  controller against a **fixed ratio matched on the controller's own realized
  average retained context**, plus an answer-quality replay pass. The
  existing controller ablation shows the controller *adapts*; this is the
  experiment needed to show whether adapting *helps* relative to an
  equally-sized static policy.

Both scripts' docstrings state the exact methodology and what they will and
won't report, so results can't be reshaped after the fact.

## How LSPM works

```
 User Query
     │
     ▼
 Vector DB (Chroma)  ───────────►  Top-K raw Arabic passages
     │
     ▼
 LSPM Middleware
   • split into sentences
   • cross-encoder relevance scoring (query × sentence)
   • keep top-k = max(1, round(r × N)) by score
   • reassemble survivors in ORIGINAL document order (not score order)
   • ratio r: fixed, or dynamic (reads vLLM's live GPU load)
     │
     ▼
 vLLM Server  ─────────────────►  OpenAI-compatible /v1/chat/completions
     │                            PagedAttention KV-cache, now under less pressure
     ▼
 Streamlit UI
```

## Repository layout

```
vllm-arabic-rag/
├── app.py                          # Streamlit demo UI
├── middleware/
│   ├── pruning.py                   # SemanticPruner + DynamicRatioController (LSPM core)
│   ├── vllm_client.py                # OpenAI-compatible HTTP client (streaming + TTFT tracking)
│   └── retriever.py                  # Chroma wrapper + in-memory mock corpus
├── eval/
│   ├── tokenization_disparity.py     # AR vs EN token-count disparity analysis
│   ├── semantic_fidelity.py          # ROUGE-L / BLEU / BERTScore: pruned vs raw
│   └── analyze_expanded.py
├── benchmark/
│   ├── locustfile.py                 # Throughput/TTFT load test (raw vs lspm vs naive)
│   ├── run_full_sweep.py             # Full ratio x concurrency GPU sweep driver
│   ├── run_controller_ablation_v2.py # Dynamic controller ablation (99.2% KV occupancy)
│   ├── run_online_e2e_benchmark.py   # NEW — true end-to-end online serving benchmark
│   ├── precompute_contexts.py
│   └── analyze_sweep_results.py
├── scripts/
│   ├── run_arcd_pilot.py             # Main ARCD accuracy evaluation (LSPM/naive/LLMLingua-2)
│   ├── run_arcd_qwen25_baseline.py   # Second-generator generalization check
│   ├── run_arcd_replication.py       # Fresh-seed disjoint replication
│   ├── run_controller_vs_matched_fixed.py  # NEW — controller vs. matched fixed ratio + quality
│   └── score_arcd.py                 # Validated Arabic-normalized EM/F1 scorer
├── data/                              # ARCD eval sets, parallel pairs, human-eval data
├── results/                            # All benchmark/eval outputs (CSV/JSONL/figures)
├── paper/                              # Manuscript sources, submission package, response letters
├── tests/
│   └── test_pruning.py
├── response_letter.sty                # LaTeX package for the MDPI reviewer response letters
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Quickstart (demo)

```bash
git clone https://github.com/Akramtaha98/vllm-arabic-rag.git
cd vllm-arabic-rag

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with a backend — pick **one**:

```bash
# Option A: self-hosted vLLM (requires a CUDA GPU)
VLLM_API_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Option B: hosted, no GPU needed — NVIDIA NIM free tier
VLLM_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
VLLM_MODEL_NAME=meta/llama-3.1-8b-instruct
VLLM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Get a free NIM key at [build.nvidia.com](https://build.nvidia.com) → pick a
> model → **Get API Key**.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Reproducing the results

<details>
<summary><b>ARCD accuracy evaluation (LSPM vs. naive vs. LLMLingua-2)</b></summary>

```bash
export VLLM_API_KEY=your_key
python scripts/run_arcd_pilot.py            # main run, resumable
python scripts/run_arcd_llmlingua2_baseline.py
python scripts/score_arcd.py                # F1/EM + significance tests
```
</details>

<details>
<summary><b>GPU serving sweep (throughput / TTFT / KV-cache)</b></summary>

```bash
vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --served-model-name meta-llama/Llama-3.1-8B-Instruct

python benchmark/precompute_contexts.py
python benchmark/run_full_sweep.py --host http://localhost:8000
python benchmark/analyze_sweep_results.py --sweep-dir results/sweep_v2 --repeats 3
```
</details>

<details>
<summary><b>Dynamic controller ablation (99.2% KV occupancy)</b></summary>

```bash
vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --served-model-name meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.3

python benchmark/run_controller_ablation_v2.py --host http://localhost:8000
```
</details>

<details>
<summary><b>NEW — end-to-end online benchmark / controller-vs-matched-fixed (not yet run)</b></summary>

```bash
python benchmark/run_online_e2e_benchmark.py --host http://localhost:8000 \
  --concurrency 1 5 10 20 --ratio 0.5 --requests-per-cell 60

python scripts/run_controller_vs_matched_fixed.py --host http://localhost:8000
```
</details>

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Deploy the demo

**Hugging Face Spaces (free, CPU):**

```bash
HF_TOKEN=hf_xxx HF_USERNAME=yourname HF_SPACE=arabic-rag-optimizer \
  bash scripts/deploy_hf.sh
```

**Docker:**

```bash
docker build -t arabic-rag-optimizer .
docker run -p 8501:8501 --env-file .env arabic-rag-optimizer
```

## Status

The manuscript is under review at MDPI *Information*. This repository is the
live companion codebase: results here reflect the current, post-revision
state of the evaluation, including the two pre-registered follow-up
experiments listed in [Known gaps](#known-gaps-being-actively-closed) above.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{tarish-zghair-lspm,
  title   = {Semantic-Driven Context Pruning for Arabic RAG:
             Quality-Preserving Compression Under Token Inflation},
  author  = {Tarish, Ali Abdul Razzaq and Zghair, Noor Abdul Khaleq},
  journal = {Information (MDPI)},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/Akramtaha98/vllm-arabic-rag}
}
```
