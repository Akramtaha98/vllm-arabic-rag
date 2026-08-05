"""
Plot throughput / KV-cache-usage comparison charts from Locust CSV output
(baseline vs. LSPM-pruned runs), for the paper's results section (4.1).

Usage:
    python benchmark/plot_results.py \
        --baseline results/baseline_stats_history.csv \
        --pruned results/lspm_r05_stats_history.csv \
        --out results/throughput_comparison.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--out", default="results/throughput_comparison.png")
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    pruned = pd.read_csv(args.pruned)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(base["User Count"], base["Requests/s"], label="Baseline (raw context)", marker="o")
    axes[0].plot(pruned["User Count"], pruned["Requests/s"], label="LSPM (pruned context)", marker="o")
    axes[0].set_xlabel("Concurrent Users")
    axes[0].set_ylabel("Throughput (req/s)")
    axes[0].set_title("Throughput vs. Concurrency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(base["User Count"], base["50%"], label="Baseline p50 latency (ms)", marker="o")
    axes[1].plot(pruned["User Count"], pruned["50%"], label="LSPM p50 latency (ms)", marker="o")
    axes[1].set_xlabel("Concurrent Users")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency vs. Concurrency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
