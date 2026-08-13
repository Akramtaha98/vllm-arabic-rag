"""
Polls a running vLLM server's Prometheus /metrics endpoint at a fixed
interval and records `vllm:gpu_cache_usage_perc` (KV-cache occupancy as a
fraction of capacity) over time, alongside a wall-clock timestamp. Intended
to run concurrently with a locust load-test (see run_full_sweep.py) so that
KV-cache occupancy during the run can be correlated with concurrency level.

Repair (network-aware experiment, paper/NETWORK_SPECIALIZATION_PLAN.md
Task 1 "synchronized KV-cache sampling"): each row now also records an
absolute Unix epoch timestamp (`epoch`), not just a scraper-relative
`t_seconds`. This is what lets this file's samples be joined, post-hoc, by
absolute wall-clock time against the edge gateway's per-request JSONL log
(edge_gateway.py) and the locust CSV -- t_seconds alone only supports
alignment *within* this one process's own run, which was not sufficient to
confidently correlate a specific KV-cache reading with a specific request
across separate processes. Also, optionally, samples `nvidia-smi` for GPU
utilization/memory alongside the vLLM metric, when available -- both
"server" and "GPU" measurements the paper's Task 6 asks for, from one
synchronized scrape loop.

Usage:
    python benchmark/metrics_scraper.py \
        --url http://localhost:8000/metrics \
        --interval 1.0 \
        --out results/metrics_raw_c50_r05.csv \
        --duration 180 \
        --gpu-util   # optional: also shell out to nvidia-smi each sample
"""
import argparse
import csv
import re
import shutil
import subprocess
import time

import requests

# GPU vLLM builds expose vllm:gpu_cache_usage_perc; CPU builds (and some
# newer vLLM versions) expose vllm:kv_cache_usage_perc instead. Match either
# so the same scraper works regardless of which build produced the metrics.
METRIC_PATTERN = re.compile(
    r"^vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
    re.MULTILINE,
)


def scrape_once(url: str):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
    except Exception as e:
        return None, str(e)
    matches = METRIC_PATTERN.findall(r.text)
    if not matches:
        return None, "metric not found in /metrics response (is this vLLM, and is the metric name still vllm:gpu_cache_usage_perc in your version?)"
    # If multiple engine replicas expose the metric, take the mean.
    vals = [float(v) for _, v in matches]
    return sum(vals) / len(vals), None


_HAVE_NVIDIA_SMI = shutil.which("nvidia-smi") is not None


def scrape_gpu_util():
    """Returns (gpu_util_pct, gpu_mem_used_mib, gpu_mem_total_mib, error).
    Best-effort: returns (None, None, None, reason) if nvidia-smi is
    unavailable or errors, never raises -- a missing GPU-utilization sample
    should not abort the KV-cache scrape loop."""
    if not _HAVE_NVIDIA_SMI:
        return None, None, None, "nvidia-smi not found on PATH"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None, None, None, f"nvidia-smi exit {out.returncode}: {out.stderr.strip()}"
        line = out.stdout.strip().splitlines()[0]
        util, mem_used, mem_total = [float(x.strip()) for x in line.split(",")]
        return util, mem_used, mem_total, None
    except Exception as e:
        return None, None, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Full URL to vLLM's /metrics endpoint")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between polls")
    ap.add_argument("--duration", type=float, default=180.0, help="Total seconds to scrape for")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--gpu-util", action="store_true",
                     help="Also sample nvidia-smi GPU utilization/memory each interval "
                          "(no-ops with a logged reason if nvidia-smi is unavailable)")
    args = ap.parse_args()

    if args.gpu_util and not _HAVE_NVIDIA_SMI:
        print("NOTE: --gpu-util requested but nvidia-smi is not on PATH; "
              "gpu_util_pct/gpu_mem_used_mib columns will be empty with error='nvidia-smi not found on PATH' "
              "for every row, not silently omitted.")

    fieldnames = ["epoch", "t_seconds", "gpu_cache_usage_perc", "error"]
    if args.gpu_util:
        fieldnames += ["gpu_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_error"]

    rows = []
    t_start = time.time()
    n_ok, n_err = 0, 0
    last_err = None
    while time.time() - t_start < args.duration:
        now = time.time()
        t = now - t_start
        val, err = scrape_once(args.url)
        if err:
            n_err += 1
            last_err = err
        else:
            n_ok += 1
        row = {"epoch": round(now, 3), "t_seconds": round(t, 2), "gpu_cache_usage_perc": val, "error": err}
        if args.gpu_util:
            util, mem_used, mem_total, gpu_err = scrape_gpu_util()
            row.update({"gpu_util_pct": util, "gpu_mem_used_mib": mem_used,
                        "gpu_mem_total_mib": mem_total, "gpu_error": gpu_err})
        rows.append(row)
        time.sleep(args.interval)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} samples to {args.out} ({n_ok} ok, {n_err} errors).")
    if n_err and n_ok == 0:
        print(f"WARNING: every scrape failed. Last error: {last_err}")
        print("Check that --url points at your vLLM server's /metrics endpoint and that vLLM was started with metrics enabled (this is the default).")


if __name__ == "__main__":
    main()
