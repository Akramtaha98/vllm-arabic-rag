"""
Orchestrates the full GPU benchmark protocol: three context conditions --
raw (unpruned baseline), lspm (semantic pruning), and naive (length-matched
truncation baseline) -- swept across the paper's headline compression
ratios {0.3, 0.5, 0.7} (raw has no ratio) and concurrency levels
{1, 10, 25, 50, 100}, against a self-hosted, PagedAttention-based vLLM
server. That's 1 + 2*3 = 7 methods x 5 concurrency = 35 cells (x --repeats,
default 3). For each cell, runs locust headless for a fixed duration while
concurrently scraping vLLM's /metrics endpoint for KV-cache occupancy, and
writes all raw output to results/sweep/repN/.

This is the CORRECTED version of the harness (paper Section 5.8/Appendix G
round-7 finding: the original single-run, fixed-block-order sweep showed a
between-method reversal at c >= 10 that could not be distinguished from two
confounds -- LSPM's cross-encoder scoring running inside this same Locust
process, and no server restart between sequential method blocks). Two fixes
are now built in and on by default:

  1. Contexts are precomputed once, up front (benchmark/precompute_contexts.py),
     so Locust does a plain dict lookup per request instead of live pruning --
     this removes the CPU-bound scoring bottleneck from the load generator
     entirely, for all three methods, at every concurrency level.
  2. All (method, ratio, concurrency) cells, across all repeats, are pooled
     into one list and shuffled (--seed for reproducibility) before running,
     so elapsed run time is decorrelated from method identity -- no cell
     spends its whole life in the same "early" or "late" part of the run
     just because of which method it belongs to.

Combined with --repeats >= 2 (default 3), this also lets
analyze_sweep_results.py report a mean and 95% CI per cell instead of a
single point estimate.

REQUIRES: a running vLLM server (see BENCHMARK_INSTRUCTIONS.md for exact
setup), reachable at --host. This script does not start vLLM itself.

Usage (from repo root, with the vLLM server already running):
    python benchmark/run_full_sweep.py \
        --host http://localhost:8000 \
        --run-time 90 \
        --repeats 3 \
        --out-dir results/sweep_v2

Runtime estimate: 35 cells x 3 repeats x (90s run + ~10s overhead)
= ~175 minutes (~3 hours) total at the defaults. Reduce --repeats to 2 or
--run-time to 60 to trade statistical confidence for GPU cost/time if
needed; --repeats 1 reproduces the original (uncontrolled-for-variance,
but now order-randomized and pruning-decoupled) single-run behavior. Safe
to interrupt and resume -- already-completed cells (locust stats CSV
present and non-empty) are skipped automatically, independent of shuffle
order.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RATIOS = [0.3, 0.5, 0.7]
CONCURRENCY = [1, 10, 25, 50, 100]
METHODS = ["raw", "lspm", "naive"]  # raw ignores ratio (single condition)
DEFAULT_CONTEXTS_PATH = ROOT / "benchmark" / "precomputed_contexts.json"


def build_cells():
    cells = []
    for users in CONCURRENCY:
        cells.append(("raw", None, users))
    for method in ("lspm", "naive"):
        for ratio in RATIOS:
            for users in CONCURRENCY:
                cells.append((method, ratio, users))
    return cells


def ensure_precomputed_contexts(path: Path):
    if path.exists():
        print(f"Using existing precomputed contexts: {path}")
        return
    print(f"Precomputed contexts not found at {path} -- generating now (one-time, CPU, ~seconds)...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmark" / "precompute_contexts.py"), "--out", str(path)],
        cwd=str(ROOT),
    )
    if result.returncode != 0 or not path.exists():
        print("ERROR: failed to precompute contexts. Aborting rather than falling back to "
              "live in-process pruning, since that reintroduces the confound this script "
              "exists to remove. Run benchmark/precompute_contexts.py manually to debug.")
        sys.exit(1)


def cell_tag(method, ratio, users):
    if method == "raw":
        return f"raw_c{users}"
    return f"{method}_r{ratio}_c{users}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="vLLM server base URL, e.g. http://localhost:8000")
    ap.add_argument("--run-time", type=int, default=90, help="Seconds per locust cell")
    ap.add_argument("--spawn-rate", type=int, default=10, help="Locust user spawn rate")
    ap.add_argument("--out-dir", default="results/sweep_v2")
    ap.add_argument("--metrics-path", default="/metrics", help="vLLM metrics path, appended to --host")
    ap.add_argument("--force", action="store_true", help="Re-run cells even if output already exists")
    ap.add_argument("--repeats", type=int, default=3,
                     help="Independent repeats per cell, for mean/CI reporting (default 3; use 1 to "
                          "reproduce single-run behavior)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed, for a reproducible run order")
    ap.add_argument("--no-randomize", action="store_true",
                     help="Disable cell-order shuffling (NOT recommended -- reintroduces the "
                          "elapsed-time/method-identity confound this script exists to fix; only "
                          "for debugging)")
    ap.add_argument("--contexts-path", default=str(DEFAULT_CONTEXTS_PATH),
                     help="Precomputed-contexts JSON path (auto-generated if missing)")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_url = args.host.rstrip("/") + args.metrics_path

    contexts_path = Path(args.contexts_path)
    ensure_precomputed_contexts(contexts_path)

    base_cells = build_cells()
    schedule = [(method, ratio, users, rep) for rep in range(1, args.repeats + 1) for (method, ratio, users) in base_cells]
    if args.no_randomize:
        print("WARNING: --no-randomize set. Cell order is NOT decorrelated from elapsed run time.")
    else:
        random.Random(args.seed).shuffle(schedule)

    total = len(schedule)
    print(f"Running {total} cells ({len(base_cells)} (method,ratio,concurrency) combinations "
          f"x {args.repeats} repeat(s), {'shuffled' if not args.no_randomize else 'in fixed block order'} "
          f"with seed={args.seed}), {args.run_time}s each. Estimated total time (uncached): "
          f"{total * (args.run_time + 15) / 60:.1f} min.\n")

    skipped, run_count = 0, 0
    for i, (method, ratio, users, rep) in enumerate(schedule, start=1):
        rep_dir = out_dir / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        tag = cell_tag(method, ratio, users)
        stats_path = rep_dir / f"locust_{tag}_stats.csv"
        metrics_csv = rep_dir / f"metrics_{tag}.csv"

        if not args.force and stats_path.exists() and stats_path.stat().st_size > 0:
            print(f"[{i}/{total}] rep{rep}/{tag} -- already done, skipping (use --force to re-run)")
            skipped += 1
            continue

        print(f"[{i}/{total}] rep={rep} method={method} ratio={ratio} concurrency={users} -> {tag}")

        env = os.environ.copy()
        env["RAG_MODE"] = method
        if ratio is not None:
            env["COMPRESSION_RATIO"] = str(ratio)
        env["PRECOMPUTED_CONTEXTS_PATH"] = str(contexts_path)

        metrics_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "benchmark" / "metrics_scraper.py"),
             "--url", metrics_url, "--interval", "1.0",
             "--duration", str(args.run_time + 10), "--out", str(metrics_csv)],
        )

        time.sleep(1)  # let the scraper start polling before load begins

        locust_csv_prefix = str(rep_dir / f"locust_{tag}")
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
            print(f"  WARNING: locust exited with code {result.returncode} for cell rep{rep}/{tag}")

        metrics_proc.wait(timeout=args.run_time + 30)
        run_count += 1
        print(f"  done: {locust_csv_prefix}_stats.csv, {metrics_csv}\n")

    print(f"\nSweep complete. {run_count} cell(s) run, {skipped} skipped (already present). "
          f"All raw output in {out_dir}/rep1..rep{args.repeats}/.")
    print("Next: python benchmark/analyze_sweep_results.py --sweep-dir", args.out_dir, "--repeats", args.repeats)


if __name__ == "__main__":
    main()
