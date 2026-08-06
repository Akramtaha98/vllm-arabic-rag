# Running the vLLM GPU benchmark

This closes the paper's one remaining measured-systems-result gap (Section 4.5/8): a real throughput / TTFT / KV-cache-occupancy benchmark on a self-hosted, PagedAttention-based vLLM server, comparing LSPM-pruned context against the raw (unpruned) baseline across a compression-ratio x concurrency grid.

I can't run this myself (no GPU in my environment), but everything below is ready to run as-is. Once it finishes, send me the `results/sweep/` folder (or at minimum `results/sweep_summary.csv` and `results/sweep_comparison.png`) and I'll fold the real numbers into the paper, replacing the current analytical projection in Section 5.4.

## 1. Hardware you need

A CUDA GPU with at least 16 GB VRAM (24 GB+ recommended for headroom under concurrency). Llama-3.1-8B-Instruct in bf16 needs ~16 GB just for weights; vLLM needs additional VRAM for the KV cache, and that additional room is exactly what lets you see cache-occupancy differences across ratios and concurrency levels. A single consumer card (RTX 3090/4090) or a cloud instance (A10G, A100 40GB, L4, etc.) both work.

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

The second command should print a `vllm:gpu_cache_usage_perc` line. If it doesn't, your vLLM version may have renamed the metric — check `curl http://localhost:8000/metrics | grep cache` and tell me what you see; the scraper's pattern is a one-line change if the metric name has shifted.

## 3. Install the benchmark's Python dependencies

On the same machine (or one that can reach the vLLM server over the network), from the repo root:

```bash
pip install -r benchmark/requirements-benchmark.txt
pip install locust requests matplotlib pandas
```

## 4. Run the full sweep

```bash
python benchmark/run_full_sweep.py \
    --host http://localhost:8000 \
    --run-time 90 \
    --out-dir results/sweep
```

This runs all 20 cells (compression ratios {0.2, 0.5, 0.8, 1.0} — 1.0 means raw/unpruned — x concurrency levels {1, 10, 25, 50, 100}), each for 90 seconds, with a KV-cache-occupancy scraper running in parallel. Total runtime is roughly 30-35 minutes. If you have more time and want steadier steady-state numbers, raise `--run-time` to 180 (the paper's originally specified duration) — total time then scales to about an hour.

You'll see progress printed as `[i/20] ratio=... concurrency=...`. It's safe to `Ctrl+C` and resume later by editing `RATIOS`/`CONCURRENCY` in `run_full_sweep.py` to skip cells you've already completed (the script doesn't currently auto-skip finished cells — let me know if you want that added, it's a small change).

## 5. Turn the raw output into the paper's numbers

```bash
python benchmark/analyze_sweep_results.py --sweep-dir results/sweep
```

This produces:
- `results/sweep_summary.csv` — one row per (ratio, concurrency) cell: throughput (req/s), p50/p95/p99 latency, mean and peak KV-cache occupancy.
- `results/sweep_comparison.png` — two panels, throughput and KV-cache occupancy, each vs. concurrency, one line per ratio.

## 6. Send it back

Send me `results/sweep_summary.csv` and `results/sweep_comparison.png` (the full `results/sweep/` folder if you don't mind the size — the per-cell locust CSVs let me double-check anything). I'll:
- Replace Section 5.4's analytical KV-cache projection with these measured numbers.
- Add a new results subsection with the throughput/latency comparison.
- Update the abstract, title framing, and Limitations/Future Work sections to reflect that the systems-level claim is now measured, not projected.
- Rebuild and recompile the LaTeX/PDF submission.

## Notes on what this does and doesn't prove

The locust user script currently sends the same small set of 5 sample Arabic questions against a fixed mock 10-document knowledge base (see `benchmark/locustfile.py`), consistent with what the paper's architecture section describes. If you'd rather load-test against the larger ARCD-based corpus I'm building for the accuracy evaluation (see the parallel work I'm doing on ground-truth QA accuracy), that's a straightforward swap in `locustfile.py`'s `SAMPLE_QUERIES`/`MOCK_CORPUS` — say the word and I'll wire that up before you run the sweep, so both new experiments pull from the same, more realistic corpus.
