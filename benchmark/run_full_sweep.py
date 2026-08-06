"""
Orchestrates the full benchmark protocol specified in the paper's Section
4.5/8: compression ratios {0.2, 0.5, 0.8, 1.0} (1.0 = no pruning / raw
baseline) swept against concurrency levels {1, 10, 25, 50, 100}, against a
self-hosted, PagedAttention-based vLLM server. For each of the 5x4 = 20
cells, runs locust headless for a fixed duration while concurrently
scraping vLLM's /metrics endpoint for KV-cache occupancy, and writes all
raw output to results/sweep/.

REQUIRES: a running vLLM server (see BENCHMARK_INSTRUCTIONS.md for exact
setup), reachable at --host. This script does not start vLLM itself.

Usage (from repo root, with the vLLM server already running):
    python benchmark/run_full_sweep.py \
        --host http://localhost:8000 \
        --run-time 90 \
        --out-dir results/sweep

Runtime estimate: 20 cells x (90s run + ~10s startup/teardown overhead)
= ~33 minutes total. Increase --run-time for more stable steady-state
throughput numbers if your GPU and time budget allow it (the paper used
180s per cell in its original protocol description; 90s is offered here
as a faster default you can override).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RATIOS = [0.2, 0.5, 0.8, 1.0]  # 1.0 == raw / unpruned baseline
CONCURRENCY = [1, 10, 25, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="vLLM server base URL, e.g. http://localhost:8000")
    ap.add_argument("--run-time", type=int, default=90, help="Seconds per locust cell")
    ap.add_argument("--spawn-rate", type=int, default=10, help="Locust user spawn rate")
    ap.add_argument("--out-dir", default="results/sweep")
    ap.add_argument("--metrics-path", default="/metrics", help="vLLM metrics path, appended to --host")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_url = args.host.rstrip("/") + args.metrics_path

    cells = [(r, c) for r in RATIOS for c in CONCURRENCY]
    print(f"Running {len(cells)} cells (ratios={RATIOS} x concurrency={CONCURRENCY}), "
          f"{args.run_time}s each. Estimated total time: {len(cells) * (args.run_time + 15) / 60:.1f} min.\n")

    for i, (ratio, users) in enumerate(cells, start=1):
        tag = f"raw_c{users}" if ratio == 1.0 else f"r{ratio}_c{users}"
        print(f"[{i}/{len(cells)}] ratio={ratio} concurrency={users} -> {tag}")

        env = os.environ.copy()
        env["RAG_MODE"] = "raw" if ratio == 1.0 else "pruned"
        env["COMPRESSION_RATIO"] = str(ratio)

        metrics_csv = out_dir / f"metrics_{tag}.csv"
        metrics_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "benchmark" / "metrics_scraper.py"),
             "--url", metrics_url, "--interval", "1.0",
             "--duration", str(args.run_time + 10), "--out", str(metrics_csv)],
        )

        time.sleep(1)  # let the scraper start polling before load begins

        locust_csv_prefix = str(out_dir / f"locust_{tag}")
        locust_cmd = [
            "locust", "-f", str(ROOT / "benchmark" / "locustfile.py"),
            "--host", args.host,
            "--users", str(users),
            "--spawn-rate", str(args.spawn_rate),
            "--run-time", f"{args.run_time}s",
            "--headless",
            "--csv", locust_csv_prefix,
        ]
        result = subprocess.run(locust_cmd, env=env, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  WARNING: locust exited with code {result.returncode} for cell {tag}")

        metrics_proc.wait(timeout=args.run_time + 30)
        print(f"  done: {locust_csv_prefix}_stats.csv, {metrics_csv}\n")

    print(f"\nSweep complete. All raw output in {out_dir}.")
    print("Next: python benchmark/analyze_sweep_results.py --sweep-dir", args.out_dir)


if __name__ == "__main__":
    main()
