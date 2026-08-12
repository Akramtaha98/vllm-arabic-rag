"""
Polls a running vLLM server's Prometheus /metrics endpoint at a fixed
interval and records `vllm:gpu_cache_usage_perc` (KV-cache occupancy as a
fraction of capacity) over time, alongside a wall-clock timestamp. Intended
to run concurrently with a locust load-test (see run_full_sweep.py) so that
KV-cache occupancy during the run can be correlated with concurrency level.

Usage:
    python benchmark/metrics_scraper.py \
        --url http://localhost:8000/metrics \
        --interval 1.0 \
        --out results/metrics_raw_c50_r05.csv \
        --duration 180
"""
import argparse
import csv
import re
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Full URL to vLLM's /metrics endpoint")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between polls")
    ap.add_argument("--duration", type=float, default=180.0, help="Total seconds to scrape for")
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()

    rows = []
    t_start = time.time()
    n_ok, n_err = 0, 0
    last_err = None
    while time.time() - t_start < args.duration:
        t = time.time() - t_start
        val, err = scrape_once(args.url)
        if err:
            n_err += 1
            last_err = err
        else:
            n_ok += 1
        rows.append({"t_seconds": round(t, 2), "gpu_cache_usage_perc": val, "error": err})
        time.sleep(args.interval)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_seconds", "gpu_cache_usage_perc", "error"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} samples to {args.out} ({n_ok} ok, {n_err} errors).")
    if n_err and n_ok == 0:
        print(f"WARNING: every scrape failed. Last error: {last_err}")
        print("Check that --url points at your vLLM server's /metrics endpoint and that vLLM was started with metrics enabled (this is the default).")


if __name__ == "__main__":
    main()
