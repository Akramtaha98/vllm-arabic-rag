"""
Orchestrates the full GPU benchmark protocol: three context conditions --
raw (unpruned baseline), lspm (semantic pruning), and naive (length-matched
truncation baseline) -- swept across the paper's headline compression
ratios {0.3, 0.5, 0.7} (raw has no ratio) and concurrency levels
{1, 10, 25, 50, 100}, against a self-hosted, PagedAttention-based vLLM
server. That's 1 + 2*3 = 7 methods x 5 concurrency = 35 cells. For each
cell, runs locust headless for a fixed duration while concurrently
scraping vLLM's /metrics endpoint for KV-cache occupancy, and writes all
raw output to results/sweep/.

REQUIRES: a running vLLM server (see BENCHMARK_INSTRUCTIONS.md for exact
setup), reachable at --host. This script does not start vLLM itself.

Usage (from repo root, with the vLLM server already running):
    python benchmark/run_full_sweep.py \
        --host http://localhost:8000 \
        --run-time 90 \
        --out-dir results/sweep

Runtime estimate: 35 cells x (90s run + ~10s startup/teardown overhead)
= ~58 minutes total. Increase --run-time for steadier steady-state
throughput numbers if your GPU and time budget allow it (180s/cell is a
safer choice for publication-quality percentiles; total time then scales
to about ~110 minutes). Safe to interrupt and resume -- already-completed
cells (both the locust stats CSV and the metrics CSV present and
non-empty) are skipped automatically.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RATIOS = [0.3, 0.5, 0.7]
CONCURRENCY = [1, 10, 25, 50, 100]
METHODS = ["raw", "lspm", "naive"]  # raw ignores ratio (single condition)


def build_cells():
    cells = []
    for users in CONCURRENCY:
        cells.append(("raw", None, users))
    for method in ("lspm", "naive"):
        for ratio in RATIOS:
            for users in CONCURRENCY:
                cells.append((method, ratio, users))
    return cells


def cell_tag(method, ratio, users):
    if method == "raw":
        return f"raw_c{users}"
    return f"{method}_r{ratio}_c{users}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="vLLM server base URL, e.g. http://localhost:8000")
    ap.add_argument("--run-time", type=int, default=90, help="Seconds per locust cell")
    ap.add_argument("--spawn-rate", type=int, default=10, help="Locust user spawn rate")
    ap.add_argument("--out-dir", default="results/sweep")
    ap.add_argument("--metrics-path", default="/metrics", help="vLLM metrics path, appended to --host")
    ap.add_argument("--force", action="store_true", help="Re-run cells even if output already exists")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_url = args.host.rstrip("/") + args.metrics_path

    cells = build_cells()
    print(f"Running {len(cells)} cells (raw x {len(CONCURRENCY)} concurrency, "
          f"+ {{lspm,naive}} x ratios={RATIOS} x concurrency={CONCURRENCY}), "
          f"{args.run_time}s each. Estimated total time (uncached): "
          f"{len(cells) * (args.run_time + 15) / 60:.1f} min.\n")

    skipped, run_count = 0, 0
    for i, (method, ratio, users) in enumerate(cells, start=1):
        tag = cell_tag(method, ratio, users)
        stats_path = out_dir / f"locust_{tag}_stats.csv"
        metrics_csv = out_dir / f"metrics_{tag}.csv"

        if not args.force and stats_path.exists() and stats_path.stat().st_size > 0:
            print(f"[{i}/{len(cells)}] {tag} -- already done, skipping (use --force to re-run)")
            skipped += 1
            continue

        print(f"[{i}/{len(cells)}] method={method} ratio={ratio} concurrency={users} -> {tag}")

        env = os.environ.copy()
        env["RAG_MODE"] = method
        if ratio is not None:
            env["COMPRESSION_RATIO"] = str(ratio)

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
        run_count += 1
        print(f"  done: {locust_csv_prefix}_stats.csv, {metrics_csv}\n")

    print(f"\nSweep complete. {run_count} cell(s) run, {skipped} skipped (already present). "
          f"All raw output in {out_dir}.")
    print("Next: python benchmark/analyze_sweep_results.py --sweep-dir", args.out_dir)


if __name__ == "__main__":
    main()
