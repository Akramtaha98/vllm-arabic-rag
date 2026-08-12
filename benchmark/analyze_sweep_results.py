"""
Aggregates the raw output of run_full_sweep.py (locust *_stats.csv files +
metrics_*.csv KV-cache occupancy logs) into:
  1. results/sweep_summary.csv - one row per (method, ratio, concurrency)
     cell with throughput (req/s), TTFT percentiles, mean tokens/sec, and
     mean/peak KV-cache occupancy, each as mean +/- 95% CI across repeats
     if --repeats > 1 was used.
  2. results/sweep_comparison.png - three panels (throughput, TTFT,
     KV-cache occupancy) vs. concurrency, one line per (method, ratio),
     with shaded 95% CI bands if repeats are available.

Supports both the original single-run layout (--sweep-dir results/sweep,
files directly in that directory) and the corrected multi-repeat layout
(--sweep-dir results/sweep_v2, files under rep1/, rep2/, ... subdirectories,
as written by the corrected run_full_sweep.py -- see paper Section 5.8/8).

Usage (after run_full_sweep.py has finished):
    python benchmark/analyze_sweep_results.py --sweep-dir results/sweep_v2 --repeats 3
    python benchmark/analyze_sweep_results.py --sweep-dir results/sweep            # single-run, legacy
"""
import argparse
import csv
import math
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


def mean_ci95(values):
    """Returns (mean, half-width of 95% CI). Uses a normal approximation
    (t-distribution collapses close to it for n>=3); with n=1 the
    half-width is reported as 0.0 (a point estimate, not a CI) rather than
    fabricating an interval from a single observation."""
    values = [v for v in values if v is not None]
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    # t critical values for small n (two-sided 95%); falls back to 1.96 for n>=30
    t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t = t_table.get(n, 1.96)
    half_width = t * sd / math.sqrt(n)
    return mean, half_width


def collect_cell_across_reps(rep_dirs, method, ratio, users):
    """Reads one (method, ratio, users) cell from every rep directory found
    and returns per-metric lists of per-rep values, ready for mean_ci95."""
    tag = cell_tag(method, ratio, users)
    ep = endpoint_name(method, ratio)
    metrics = {
        "requests_per_s": [], "ttft_p50_ms": [], "ttft_p95_ms": [],
        "tokens_per_sec_avg": [], "failure_count": [], "request_count": [],
        "mean_kv": [], "peak_kv": [],
    }
    for rep_dir in rep_dirs:
        stats_path = rep_dir / f"locust_{tag}_stats.csv"
        metrics_path = rep_dir / f"metrics_{tag}.csv"
        rows_by_name = read_stats_rows(stats_path)
        if not rows_by_name:
            continue
        chat_row = rows_by_name.get(f"chat[{ep}]") or rows_by_name.get("Aggregated")
        ttft_row = rows_by_name.get(f"ttft_ms[{ep}]")
        tps_row = rows_by_name.get(f"tokens_per_sec[{ep}]")
        mean_kv, peak_kv = read_metrics_csv(metrics_path)

        metrics["requests_per_s"].append(_f(chat_row, "Requests/s"))
        metrics["ttft_p50_ms"].append(_f(ttft_row, "50%"))
        metrics["ttft_p95_ms"].append(_f(ttft_row, "95%"))
        metrics["tokens_per_sec_avg"].append(_f(tps_row, "Average Response Time"))
        metrics["failure_count"].append(_f(chat_row, "Failure Count"))
        metrics["request_count"].append(_f(chat_row, "Request Count"))
        metrics["mean_kv"].append(mean_kv * 100 if mean_kv is not None else None)
        metrics["peak_kv"].append(peak_kv * 100 if peak_kv is not None else None)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/sweep_v2")
    ap.add_argument("--repeats", type=int, default=None,
                     help="Number of rep1..repN subdirectories to look for. If omitted, "
                          "auto-detects repN subdirectories under --sweep-dir; if none exist, "
                          "falls back to treating --sweep-dir itself as a single-run (legacy) layout.")
    ap.add_argument("--out-csv", default="results/sweep_summary.csv")
    ap.add_argument("--out-fig", default="results/sweep_comparison.png")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if args.repeats is not None:
        rep_dirs = [sweep_dir / f"rep{i}" for i in range(1, args.repeats + 1)]
        rep_dirs = [d for d in rep_dirs if d.exists()]
    else:
        rep_dirs = sorted(d for d in sweep_dir.glob("rep*") if d.is_dir())
    legacy_single_run = len(rep_dirs) == 0
    if legacy_single_run:
        print(f"No repN subdirectories found under {sweep_dir} -- treating as a single-run "
              "(legacy) layout. CI columns will be 0 (point estimates only).")
        rep_dirs = [sweep_dir]
    else:
        print(f"Found {len(rep_dirs)} repeat(s): {[d.name for d in rep_dirs]}")

    out_rows = []
    missing = []

    for method, ratio in CONDITIONS:
        for users in CONCURRENCY:
            per_rep = collect_cell_across_reps(rep_dirs, method, ratio, users)
            if not per_rep["requests_per_s"]:
                missing.append(cell_tag(method, ratio, users))
                continue

            rps_mean, rps_ci = mean_ci95(per_rep["requests_per_s"])
            ttft50_mean, ttft50_ci = mean_ci95(per_rep["ttft_p50_ms"])
            ttft95_mean, ttft95_ci = mean_ci95(per_rep["ttft_p95_ms"])
            tps_mean, tps_ci = mean_ci95(per_rep["tokens_per_sec_avg"])
            mean_kv_mean, mean_kv_ci = mean_ci95(per_rep["mean_kv"])
            peak_kv_mean, peak_kv_ci = mean_ci95(per_rep["peak_kv"])

            out_rows.append({
                "method": method,
                "ratio": ratio if ratio is not None else "",
                "concurrency": users,
                "n_reps": len(per_rep["requests_per_s"]),
                "requests_per_s_mean": round(rps_mean, 3) if rps_mean is not None else "",
                "requests_per_s_ci95": round(rps_ci, 3) if rps_ci is not None else "",
                "ttft_p50_ms_mean": round(ttft50_mean, 1) if ttft50_mean is not None else "",
                "ttft_p50_ms_ci95": round(ttft50_ci, 1) if ttft50_ci is not None else "",
                "ttft_p95_ms_mean": round(ttft95_mean, 1) if ttft95_mean is not None else "",
                "ttft_p95_ms_ci95": round(ttft95_ci, 1) if ttft95_ci is not None else "",
                "tokens_per_sec_avg_mean": round(tps_mean, 2) if tps_mean is not None else "",
                "tokens_per_sec_avg_ci95": round(tps_ci, 2) if tps_ci is not None else "",
                "total_failures": int(sum(per_rep["failure_count"])),
                "total_requests": int(sum(per_rep["request_count"])),
                "mean_kv_cache_usage_pct_mean": round(mean_kv_mean, 2) if mean_kv_mean is not None else "",
                "mean_kv_cache_usage_pct_ci95": round(mean_kv_ci, 2) if mean_kv_ci is not None else "",
                "peak_kv_cache_usage_pct_mean": round(peak_kv_mean, 2) if peak_kv_mean is not None else "",
                "peak_kv_cache_usage_pct_ci95": round(peak_kv_ci, 2) if peak_kv_ci is not None else "",
            })

    if missing:
        print(f"WARNING: {len(missing)} expected cell(s) not found (may not have run yet):")
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
    print(f"Wrote {len(out_rows)} rows to {args.out_csv} (n_reps per cell: "
          f"{sorted(set(r['n_reps'] for r in out_rows))})")

    # --- Plot: throughput, TTFT, KV-cache occupancy vs. concurrency, with CI bands ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 13))
    for method, ratio in CONDITIONS:
        cell_rows = [r for r in out_rows if r["method"] == method and r["ratio"] == (ratio if ratio is not None else "")]
        cell_rows.sort(key=lambda r: r["concurrency"])
        if not cell_rows:
            continue
        xs = [r["concurrency"] for r in cell_rows]
        label = "Raw (no pruning)" if method == "raw" else f"{'LSPM' if method == 'lspm' else 'Naive truncation'} r={ratio}"
        color = COLORS[(method, ratio)]

        rps = [r["requests_per_s_mean"] for r in cell_rows]
        rps_ci = [r["requests_per_s_ci95"] or 0 for r in cell_rows]
        axes[0].plot(xs, rps, marker="o", color=color, label=label, linewidth=1.8, markersize=5)
        axes[0].fill_between(xs, [a - b for a, b in zip(rps, rps_ci)], [a + b for a, b in zip(rps, rps_ci)], color=color, alpha=0.15)

        ttft = [r["ttft_p50_ms_mean"] for r in cell_rows]
        ttft_ci = [r["ttft_p50_ms_ci95"] or 0 for r in cell_rows]
        axes[1].plot(xs, ttft, marker="o", color=color, label=label, linewidth=1.8, markersize=5)
        axes[1].fill_between(xs, [a - b for a, b in zip(ttft, ttft_ci)], [a + b for a, b in zip(ttft, ttft_ci)], color=color, alpha=0.15)

        kv_rows = [(r["concurrency"], r["mean_kv_cache_usage_pct_mean"], r["mean_kv_cache_usage_pct_ci95"] or 0)
                   for r in cell_rows if r["mean_kv_cache_usage_pct_mean"] != ""]
        if kv_rows:
            kx, ky, kci = zip(*kv_rows)
            axes[2].plot(kx, ky, marker="o", color=color, label=label, linewidth=1.8, markersize=5)
            axes[2].fill_between(kx, [a - b for a, b in zip(ky, kci)], [a + b for a, b in zip(ky, kci)], color=color, alpha=0.15)

    axes[0].set_xlabel("Concurrent requests", fontsize=12)
    axes[0].set_ylabel("Throughput (req/s)", fontsize=12)
    axes[0].set_title("Throughput vs. concurrency", fontsize=13)
    axes[1].set_xlabel("Concurrent requests", fontsize=12)
    axes[1].set_ylabel("Median TTFT (ms)", fontsize=12)
    axes[1].set_title("TTFT vs. concurrency", fontsize=13)
    axes[2].set_xlabel("Concurrent requests", fontsize=12)
    axes[2].set_ylabel("Mean KV-cache occupancy (%)", fontsize=12)
    axes[2].set_title("KV-cache occupancy vs. concurrency", fontsize=13)
    for ax in axes:
        ax.tick_params(labelsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out_fig}")


if __name__ == "__main__":
    main()
