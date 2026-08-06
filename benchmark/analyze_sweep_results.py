"""
Aggregates the raw output of run_full_sweep.py (locust *_stats.csv files +
metrics_*.csv KV-cache occupancy logs) into:
  1. results/sweep_summary.csv - one row per (ratio, concurrency) cell with
     throughput (req/s, tokens/s), p50/p95/p99 latency, and mean/peak
     KV-cache occupancy.
  2. results/sweep_comparison.png - throughput and KV-cache occupancy vs.
     concurrency, one line per compression ratio.

This is the script that turns your GPU run's raw output into the exact
numbers needed for the paper's Section 5.4 (replacing the analytical
projection with a measured result) and a real Table 5 / Figure 5.

Usage (after run_full_sweep.py has finished):
    python benchmark/analyze_sweep_results.py --sweep-dir results/sweep
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIOS = [0.2, 0.5, 0.8, 1.0]
CONCURRENCY = [1, 10, 25, 50, 100]
BLUE, ORANGE, GREEN, RED, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#555555"
COLORS = {0.2: RED, 0.5: ORANGE, 0.8: GREEN, 1.0: GRAY}


def read_locust_stats(path: Path):
    """Locust's *_stats.csv has one row per endpoint name plus an
    'Aggregated' row; we want the aggregated row's summary stats."""
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), None)
    if agg is None:
        return None
    return {
        "requests_per_s": float(agg.get("Requests/s", 0) or 0),
        "p50_ms": float(agg.get("50%", 0) or 0),
        "p95_ms": float(agg.get("95%", 0) or 0),
        "p99_ms": float(agg.get("99%", 0) or 0),
        "failure_count": int(float(agg.get("Failure Count", 0) or 0)),
        "request_count": int(float(agg.get("Request Count", 0) or 0)),
    }


def read_metrics_csv(path: Path):
    if not path.exists():
        return None, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    vals = [float(r["gpu_cache_usage_perc"]) for r in rows if r.get("gpu_cache_usage_perc") not in (None, "", "None")]
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/sweep")
    ap.add_argument("--out-csv", default="results/sweep_summary.csv")
    ap.add_argument("--out-fig", default="results/sweep_comparison.png")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_rows = []
    missing = []

    for ratio in RATIOS:
        for users in CONCURRENCY:
            tag = f"raw_c{users}" if ratio == 1.0 else f"r{ratio}_c{users}"
            stats_path = sweep_dir / f"locust_{tag}_stats.csv"
            metrics_path = sweep_dir / f"metrics_{tag}.csv"

            stats = read_locust_stats(stats_path)
            mean_kv, peak_kv = read_metrics_csv(metrics_path)

            if stats is None:
                missing.append(str(stats_path))
                continue

            out_rows.append({
                "ratio": ratio,
                "concurrency": users,
                "requests_per_s": stats["requests_per_s"],
                "p50_ms": stats["p50_ms"],
                "p95_ms": stats["p95_ms"],
                "p99_ms": stats["p99_ms"],
                "failure_count": stats["failure_count"],
                "request_count": stats["request_count"],
                "mean_kv_cache_usage_pct": round(mean_kv * 100, 2) if mean_kv is not None else "",
                "peak_kv_cache_usage_pct": round(peak_kv * 100, 2) if peak_kv is not None else "",
            })

    if missing:
        print(f"WARNING: {len(missing)} expected result file(s) not found (cell may not have run yet):")
        for m in missing[:10]:
            print("  -", m)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not out_rows:
        print("No results found. Did you run run_full_sweep.py first?")
        return

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.out_csv}")

    # --- Plot: throughput and KV-cache occupancy vs. concurrency, one line per ratio ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ratio in RATIOS:
        cell_rows = [r for r in out_rows if r["ratio"] == ratio]
        cell_rows.sort(key=lambda r: r["concurrency"])
        if not cell_rows:
            continue
        xs = [r["concurrency"] for r in cell_rows]
        label = "Raw (no pruning)" if ratio == 1.0 else f"LSPM r={ratio}"
        axes[0].plot(xs, [r["requests_per_s"] for r in cell_rows], marker="o",
                     color=COLORS[ratio], label=label)
        kv_ys = [r["mean_kv_cache_usage_pct"] for r in cell_rows if r["mean_kv_cache_usage_pct"] != ""]
        kv_xs = [r["concurrency"] for r in cell_rows if r["mean_kv_cache_usage_pct"] != ""]
        if kv_ys:
            axes[1].plot(kv_xs, kv_ys, marker="o", color=COLORS[ratio], label=label)

    axes[0].set_xlabel("Concurrent requests")
    axes[0].set_ylabel("Throughput (req/s)")
    axes[0].set_title("Throughput vs. concurrency")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("Mean KV-cache occupancy (%)")
    axes[1].set_title("KV-cache occupancy vs. concurrency")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out_fig}")


if __name__ == "__main__":
    main()
