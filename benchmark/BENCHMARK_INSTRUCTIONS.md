# Running the vLLM GPU benchmark

This closes the paper's single biggest reviewer-facing gap: a real
throughput / TTFT / KV-cache-occupancy benchmark on a self-hosted,
PagedAttention-based vLLM server, comparing three conditions — **raw**
(unpruned), **LSPM** (semantic pruning), and **naive truncation**
(length-matched baseline) — at the paper's headline compression ratios
r = {0.3, 0.5, 0.7}, swept across concurrency levels {1, 10, 25, 50, 100}.

I can't run this myself (no GPU in my environment), but everything below
is ready to run as-is — I extended the harness to add the naive-truncation
condition and real TTFT measurement (via streaming) on top of what was
already built for the raw-vs-LSPM-only version. Once it finishes, send me
`results/sweep_summary.csv` and `results/sweep_comparison.png` and I'll
fold the real numbers into the paper, replacing the current analytical
KV-cache projection.

## 1. Hardware you need

A CUDA GPU with at least 16 GB VRAM (24 GB+ recommended for headroom
under concurrency). Llama-3.1-8B-Instruct in bf16 needs ~16 GB just for
weights; vLLM needs additional VRAM for the KV cache, and that additional
room is exactly what lets you see cache-occupancy differences across
ratios and concurrency levels. A single consumer card (RTX 3090/4090) or
a cloud instance (A10G, A100 40GB, L4, etc.) both work.

## 2. Install vLLM and start the server

```bash
# In a fresh venv/conda env, on the GPU machine:
pip install vllm

# Start the server with the same model used throughout the paper's pilots:
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16
```

Leave this running in its own terminal/session. Confirm it's up:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/metrics | grep gpu_cache_usage_perc
```

The second command should print a `vllm:gpu_cache_usage_perc` line. If it
doesn't, your vLLM version may have renamed the metric — check
`curl http://localhost:8000/metrics | grep cache` and tell me what you
see; the scraper's pattern is a one-line change if the metric name has
shifted.

## 3. Install the benchmark's Python dependencies

On the same machine (or one that can reach the vLLM server over the
network), from the repo root:

```bash
pip install -r benchmark/requirements-benchmark.txt
```

(This now includes `locust`, `requests`, `matplotlib`, `pandas`, and
`sentence-transformers` — the last one is needed because the LSPM
condition loads the same cross-encoder reranker used in the accuracy
experiments, via `middleware.pruning.SemanticPruner`.)

## 4. Run the full sweep

```bash
python benchmark/run_full_sweep.py \
    --host http://localhost:8000 \
    --run-time 90 \
    --out-dir results/sweep
```

This runs all 35 cells: raw (5 concurrency levels) + LSPM at r∈{0.3,0.5,0.7}
(5 concurrency levels each) + naive truncation at r∈{0.3,0.5,0.7} (5
concurrency levels each), each for 90 seconds, with a KV-cache-occupancy
scraper running in parallel. Total runtime is roughly 55-60 minutes. If
you have more time and want steadier steady-state numbers, raise
`--run-time` to 180; total time then scales to about 1.5-2 hours.

You'll see progress printed as `[i/35] method=... ratio=... concurrency=...`.
The script now **auto-skips cells that already have output**, so it's safe
to `Ctrl+C` and simply re-run the same command later to pick up where you
left off (use `--force` to re-run everything from scratch instead).

## 5. Turn the raw output into the paper's numbers

```bash
python benchmark/analyze_sweep_results.py --sweep-dir results/sweep
```

This produces:
- `results/sweep_summary.csv` — one row per (method, ratio, concurrency)
  cell: throughput (req/s), end-to-end latency percentiles (p50/p95/p99),
  **TTFT percentiles (p50/p95)**, mean completion tokens/sec, and mean/peak
  KV-cache occupancy.
- `results/sweep_comparison.png` — three panels (throughput, TTFT,
  KV-cache occupancy), each vs. concurrency, one line per (method, ratio)
  — 7 lines total: raw, LSPM×3 ratios, naive×3 ratios.

I test-ran both scripts against synthetic (fabricated) locust/metrics CSVs
before handing this off, so the parsing and plotting logic is verified —
what's untested is only the real GPU run itself, which needs your hardware.

## 6. Send it back

Send me `results/sweep_summary.csv` and `results/sweep_comparison.png`
(the full `results/sweep/` folder if you don't mind the size — the
per-cell locust CSVs let me double-check anything). I'll:
- Replace the analytical KV-cache projection with these measured numbers.
- Add a new results subsection with the throughput/TTFT/latency comparison
  across raw vs. LSPM vs. naive truncation.
- Update the abstract, title framing, and Limitations/Future Work sections
  to reflect that the systems-level claim is now measured, not projected.
- Rebuild and recompile the LaTeX/PDF submission.

## What changed from the earlier version of this harness

- Added `naive` as a third `RAG_MODE` in `benchmark/locustfile.py`, using
  the exact same length-matched truncation function as
  `scripts/run_arcd_pilot.py` (first-N-sentences by the same ratio), so
  the systems benchmark and the accuracy results share one baseline
  definition.
- Switched compression ratios from {0.2, 0.5, 0.8} to {0.3, 0.5, 0.7} —
  the paper's actual headline ratios, so the throughput numbers line up
  with the accuracy numbers already in the paper.
- Added real TTFT measurement: requests now stream (`stream=True`) and the
  locustfile records the wall-clock time to the first token delta as a
  custom locust metric (`ttft_ms[<method>_r<ratio>]`), alongside the
  existing total end-to-end latency and a tokens/sec estimate.
- `run_full_sweep.py` now auto-skips already-completed cells instead of
  always re-running the full grid, so a `Ctrl+C`'d run can be resumed with
  the exact same command.

## Notes on what this does and doesn't prove

The locust user script sends the same small set of 5 sample Arabic
questions against a fixed mock 10-document knowledge base
(`benchmark/locustfile.py`'s `SAMPLE_QUERIES` / `middleware.retriever.MOCK_CORPUS`),
consistent with what the paper's architecture section describes — this
benchmark measures **systems performance** (throughput, latency, cache
occupancy) under load, not answer accuracy, which is what the separate
ARCD ground-truth evaluation (`scripts/run_arcd_pilot.py`,
`data/arcd_results.jsonl`) already measures. The two are deliberately
different experiments answering different questions; this benchmark's
job is only to confirm the throughput/memory story is real and not just
an analytical projection.
