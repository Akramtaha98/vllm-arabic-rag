"""
Aggregates the raw output of run_full_sweep.py (locust *_stats.csv files +
metrics_*.csv KV-cache occupancy logs) into:
  1. results/sweep_summary.csv - one row per (method, ratio, concurrency)
     cell with throughput (req/s), end-to-end latency percentiles, TTFT
     percentiles, mean tokens/sec, and mean/peak KV-cache occupancy.
  2. results/sweep_comparison.png - three panels (throughput, TTFT,
     KV-cache occupancy) vs. concurrency, one line per (method, ratio).

This is the script that turns your GPU run's raw output into the exact
numbers needed for the paper's systems-benchmark section (replacing the
analytical KV-cache projection with a measured result) and a real
Table/Figure.

Usage (after run_full_sweep.py has finished):
    python benchmark/analyze_sweep_results.py --sweep-dir results/sweep
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIOS = [0.3, 0.5, 0.7]
CONCURRENCY = [1, 10, 25, 50, 100]
CONDITIONS = [("raw", None)] + [(m, r) for m in ("lspm", "naive") for r in RATIOS]

COLORS = {
    ("raw", None): "#555555",
    ("lspm", 0.3): "#0072B2", ("lspm", 0.5): "#56B4E9", ("lspm", 0.7): "#009E73",
    ("naive", 0.3): "#D55E00", ("naive", 0.5): "#E69F00", ("naive", 0.7): "#CC79A7",
}


def cell_tag(method, ratio, users):
    if method == "raw":
        return f"raw_c{users}"
    return f"{method}_r{ratio}_c{users}"


def endpoint_name(method, ratio):
    if method == "raw":
        return "raw"
    return f"{method}_r{ratio}"


def read_stats_rows(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r.get("Name"): r for r in rows}


def _f(row, key, default=0.0):
    if row is None:
        return default
    v = row.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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

    for method, ratio in CONDITIONS:
        for users in CONCURRENCY:
            tag = cell_tag(method, ratio, users)
            stats_path = sweep_dir / f"locust_{tag}_stats.csv"
            metrics_path = sweep_dir / f"metrics_{tag}.csv"

            rows_by_name = read_stats_rows(stats_path)
            if not rows_by_name:
                missing.append(str(stats_path))
                continue

            ep = endpoint_name(method, ratio)
            chat_row = rows_by_name.get(f"chat[{ep}]") or rows_by_name.get("Aggregated")
            ttft_row = rows_by_name.get(f"ttft_ms[{ep}]")
            tps_row = rows_by_name.get(f"tokens_per_sec[{ep}]")
            mean_kv, peak_kv = read_metrics_csv(metrics_path)

            out_rows.append({
                "method": method,
                "ratio": ratio if ratio is not None else "",
                "concurrency": users,
                "requests_per_s": round(_f(chat_row, "Requests/s"), 3),
                "e2e_p50_ms": _f(chat_row, "50%"),
                "e2e_p95_ms": _f(chat_row, "95%"),
                "e2e_p99_ms": _f(chat_row, "99%"),
                "ttft_p50_ms": _f(ttft_row, "50%"),
                "ttft_p95_ms": _f(ttft_row, "95%"),
                "tokens_per_sec_avg": round(_f(tps_row, "Average Response Time"), 2),
                "failure_count": int(_f(chat_row, "Failure Count")),
                "request_count": int(_f(chat_row, "Request Count")),
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

    # --- Plot: throughput, TTFT, KV-cache occupancy vs. concurrency ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for method, ratio in CONDITIONS:
        cell_rows = [r for r in out_rows if r["method"] == method and r["ratio"] == (ratio if ratio is not None else "")]
        cell_rows.sort(key=lambda r: r["concurrency"])
        if not cell_rows:
            continue
        xs = [r["concurrency"] for r in cell_rows]
        label = "Raw (no pruning)" if method == "raw" else f"{'LSPM' if method == 'lspm' else 'Naive truncation'} r={ratio}"
        color = COLORS[(method, ratio)]
        axes[0].plot(xs, [r["requests_per_s"] for r in cell_rows], marker="o", color=color, label=label)
        axes[1].plot(xs, [r["ttft_p50_ms"] for r in cell_rows], marker="o", color=color, label=label)
        kv_ys = [r["mean_kv_cache_usage_pct"] for r in cell_rows if r["mean_kv_cache_usage_pct"] != ""]
        kv_xs = [r["concurrency"] for r in cell_rows if r["mean_kv_cache_usage_pct"] != ""]
        if kv_ys:
            axes[2].plot(kv_xs, kv_ys, marker="o", color=color, label=label)

    axes[0].set_xlabel("Concurrent requests")
    axes[0].set_ylabel("Throughput (req/s)")
    axes[0].set_title("Throughput vs. concurrency")
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("Median TTFT (ms)")
    axes[1].set_title("TTFT vs. concurrency")
    axes[2].set_xlabel("Concurrent requests")
    axes[2].set_ylabel("Mean KV-cache occupancy (%)")
    axes[2].set_title("KV-cache occupancy vs. concurrency")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out_fig}")


if __name__ == "__main__":
    main()
