# Running the vLLM GPU benchmark

## Status

**v1 (single run, fixed block order) is done and already in the paper**
as Section 5.8/Appendix G. It found a real, but narrow and not free,
result: at the single c = 1 concurrency cell, LSPM's measured KV-cache
occupancy is lower than raw's at every ratio, but request throughput is
8.5-11.9% *lower* than raw's at that same cell (completion tokens/second,
a cleaner measure of model-side speed, is within 0.5%) — report this
throughput gap plainly, do not say "no throughput cost." The KV-cache
comparison does **not** support a claim at c = 10 or above either — it is
already mixed at c = 10 and reverses outright by c ≥ 25, for both LSPM and
naive truncation.
Two likely causes are identified in the paper: LSPM's cross-encoder
scoring ran inside the same Locust process that issued requests (a
possible client-side bottleneck at high simulated concurrency), and all
three method blocks ran sequentially against one continuously running
server with no restart in between, confounding elapsed run time with
method identity.

**This document now describes v2: the corrected re-run**, which fixes
both issues and adds repeated runs so the paper can report confidence
intervals instead of a single point estimate per cell. This is the top
remaining priority before the paper can claim anything beyond the single
c = 1 cell.

## What changed in the harness (v1 → v2)

- **`benchmark/precompute_contexts.py` (new).** Precomputes every
  (method, ratio, query) context once, up front, on CPU, into
  `benchmark/precomputed_contexts.json`. `run_full_sweep.py` now calls
  this automatically if the file doesn't exist yet.
- **`benchmark/locustfile.py`** now does a plain dict lookup against that
  file per request (`PRECOMPUTED_CONTEXTS_PATH` env var, set automatically
  by `run_full_sweep.py`) instead of computing LSPM's cross-encoder score
  or naive truncation live inside the Locust process. This removes the
  CPU-bound-scoring-as-bottleneck confound entirely, for all three
  methods, at every concurrency level.
- **`benchmark/run_full_sweep.py`** now pools every (method, ratio,
  concurrency) cell across all repeats into one list and **shuffles it**
  (seeded, `--seed`, default 42) before running, instead of running fixed
  method blocks in sequence. This decorrelates elapsed run time from
  method identity — no method spends its whole run "early" or "late"
  just because of which condition it is. Output now goes to
  `results/sweep_v2/rep1/`, `rep2/`, ... (one subdirectory per repeat)
  instead of directly into the sweep directory.
- **`--repeats N`** (default 3): runs the full 35-cell grid N independent
  times (shuffled together, not run as three separate back-to-back
  passes), so each cell has N independent samples.
- **`benchmark/analyze_sweep_results.py`** now aggregates across
  `repN/` subdirectories and reports **mean ± 95% CI** per cell (t-distribution
  based, exact for small n) instead of a single point estimate, and the
  comparison figure now plots shaded CI bands. It still auto-detects the
  old single-run (v1) layout if you point it at `results/sweep/` with no
  `repN` subdirectories, for backward compatibility.

## 1. Hardware you need

Same as before: a CUDA GPU with at least 16 GB VRAM (24 GB+ recommended).
An RTX 3090/4090 or a cloud instance (A10G, A100 40GB, L4, etc.) both
work — this is exactly what you already used for v1.

## 2. Install vLLM and start the server

```bash
# In a fresh venv/conda env, on the GPU machine:
pip install vllm

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --max-model-len 8192
```

Leave this running in its own terminal/session. Confirm it's up:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/metrics | grep kv_cache_usage_perc
```

## 3. Install dependencies (same as v1, no new packages)

```bash
pip install -r benchmark/requirements-benchmark.txt
```

## 4. Run the corrected sweep

```bash
python benchmark/run_full_sweep.py \
    --host http://localhost:8000 \
    --run-time 90 \
    --repeats 3 \
    --out-dir results/sweep_v2
```

The first thing this does is generate `benchmark/precomputed_contexts.json`
automatically (a few seconds, one-time cross-encoder load, CPU only — you
should see "Precomputed contexts not found... generating now"). Then it
runs 35 cells × 3 repeats = 105 total runs, shuffled, ~90s each.

**Time and cost estimate: ~175 minutes (~3 hours) at the defaults.** This
is roughly 3x the v1 runtime (v1 was one pass; this is three, for
confidence intervals) — budget GPU rental accordingly. If you want to
trade statistical confidence for time/cost:
- `--repeats 2` instead of 3 (still gives a CI, just wider) — cuts to ~2 hours.
- `--run-time 60` instead of 90 — cuts proportionally, at the cost of noisier
  per-cell percentiles.
- `--repeats 1` reproduces single-run behavior, but still gets the
  precomputed-contexts and randomized-order fixes, so even a single pass
  is meaningfully cleaner than v1 if you're tight on budget — just won't
  have CIs.

Safe to `Ctrl+C` and resume with the same command; already-completed
cells (per repeat) are skipped automatically regardless of shuffle order.

## 5. Turn the raw output into the paper's numbers

```bash
python benchmark/analyze_sweep_results.py --sweep-dir results/sweep_v2 --repeats 3
```

Produces `results/sweep_summary.csv` (now with `_mean` and `_ci95` columns
per metric, plus `n_reps`) and `results/sweep_comparison.png` (three
panels stacked vertically for readability, with shaded 95% CI bands).

I test-ran the aggregation and plotting logic against synthetic
(fabricated) 3-repeat locust/metrics CSVs before handing this off, so
that part is verified — what's untested is the real GPU run itself.

## 6. Send it back, and re-archive on Zenodo

Send me `results/sweep_summary.csv` and `results/sweep_comparison.png`
(the full `results/sweep_v2/` folder if you don't mind the size). I'll
replace Section 5.8/Appendix G's single-cell-exploratory framing with
the full, confidence-interval-backed comparison across all concurrency
levels, rebuild the LaTeX/PDF, and this closes what is currently the
paper's top open item.

**Separately, once the paper is updated with v2 results:** push the new
data/scripts/PDF to GitHub, tag a new release (e.g. `v1.1-submission`),
and create a **new Zenodo version** of the existing record (Zenodo
supports versioning under one concept DOI — don't create a brand-new
record) so the version-specific DOI cited in the paper matches what's
actually archived. I'll give exact commands once v2 results are in hand.

## Notes on what this does and doesn't prove

The locust user script sends the same small set of 5 sample Arabic
questions against a fixed mock 10-document knowledge base
(`benchmark/locustfile.py`'s `SAMPLE_QUERIES` / `middleware.retriever.MOCK_CORPUS`),
consistent with what the paper's architecture section describes — this
benchmark measures **systems performance** (throughput, latency, cache
occupancy) under load, not answer accuracy, which is what the separate
ARCD ground-truth evaluation (`scripts/run_arcd_pilot.py`,
`data/arcd_results.jsonl`) already measures. The two are deliberately
different experiments answering different questions.
