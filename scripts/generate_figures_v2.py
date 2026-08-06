"""
Round-2 figures, generated from the real expanded pilot (data/eval_set_expanded.jsonl)
and its statistical analysis (results/fidelity_summary.csv, results/lspm_vs_naive_paired.csv).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#555555"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def load_summary():
    with open(ROOT / "results/fidelity_summary.csv") as f:
        rows = list(csv.DictReader(f))
    data = defaultdict(dict)  # (method, ratio) -> {metric: (mean, lo, hi)}
    for r in rows:
        key = (r["method"], float(r["ratio"]))
        if r["metric"] == "char_reduction":
            data[key]["char_reduction"] = float(r["mean"])
        else:
            lo = float(r["ci95_lo"]) if r["ci95_lo"] != "" else None
            hi = float(r["ci95_hi"]) if r["ci95_hi"] != "" else None
            data[key][r["metric"]] = (float(r["mean"]), lo, hi)
    return data


def fig4_ratio_sweep():
    data = load_summary()
    ratios = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    metrics = [("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore-F1"), ("bleu", "BLEU (/100)")]

    for ax, (metric, label) in zip(axes, metrics):
        for method, color, marker in [("lspm", BLUE, "o"), ("naive", ORANGE, "s")]:
            means, los, his = [], [], []
            for r in ratios:
                m, lo, hi = data[(method, r)][metric]
                if metric == "bleu":
                    m, lo, hi = m / 100, (lo / 100 if lo is not None else None), (hi / 100 if hi is not None else None)
                means.append(m)
                los.append(m - lo if lo is not None else 0)
                his.append(hi - m if hi is not None else 0)
            label_name = "LSPM (semantic cross-encoder)" if method == "lspm" else "Naive length-matched truncation"
            ax.errorbar(ratios, means, yerr=[los, his], marker=marker, color=color,
                         label=label_name, capsize=4, linewidth=1.6, markersize=6)
        ax.set_xlabel("Compression ratio r")
        ax.set_title(label, fontsize=11)
        ax.set_xticks(ratios)
        ax.set_ylim(0.5, 1.05)

    axes[0].set_ylabel("Score (0-1, higher = more similar\nto raw-context answer)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Fidelity vs. compression ratio: LSPM vs. naive truncation baseline\n(n=8 questions per cell, 95% bootstrap CI, live Llama-3.1-8B-Instruct endpoint)",
                 fontsize=11.5, y=1.14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_ratio_sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig5_paired_comparison():
    """
    Redesigned as three side-by-side panels (one per metric, each on its own
    natural scale), matching fig4_ratio_sweep's small-multiples layout,
    instead of the previous single-axis grouped bar chart that placed a
    rescaled "BLEU/100" alongside ROUGE-L and BERTScore-F1 on one shared
    0-1 axis. Sharing an axis across metrics with different native ranges
    invited a misleading visual comparison of bar heights across metrics;
    giving BLEU its own 0-100 panel removes that entirely.
    """
    with open(ROOT / "results/lspm_vs_naive_paired.csv") as f:
        rows = list(csv.DictReader(f))

    ratios = sorted({float(r["ratio"]) for r in rows})
    metrics = [
        ("rougeL", "ROUGE-L", (-0.08, 0.09)),
        ("bertscore_f1", "BERTScore-F1", (-0.012, 0.012)),
        ("bleu", "BLEU (0-100 scale)", (-10, 10)),
    ]
    colors = [BLUE, GREEN, ORANGE]
    x = range(len(ratios))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    for ax, (metric, label, ylim), color in zip(axes, metrics, colors):
        vals, pvals = [], []
        for r in ratios:
            row = next(rr for rr in rows if float(rr["ratio"]) == r and rr["metric"] == metric)
            vals.append(float(row["mean_diff_lspm_minus_naive"]))
            pvals.append(float(row["bootstrap_p_value"]))
        bars = ax.bar(list(x), vals, width=0.5, color=color, alpha=0.82)
        for b, p in zip(bars, pvals):
            marker = "n.s." if p >= 0.05 else f"p={p:.3f}"
            offset = (ylim[1] - ylim[0]) * 0.03
            y_text = b.get_height() + offset if b.get_height() >= 0 else b.get_height() - offset
            va = "bottom" if b.get_height() >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, y_text, marker, ha="center", va=va, fontsize=8)
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"r={r}" for r in ratios])
        ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=11)

    axes[0].set_ylabel("Mean difference (LSPM − naive truncation)")
    fig.suptitle(
        "Paired comparison: does semantic scoring beat naive truncation?\n"
        "(n=8 paired questions per ratio; bootstrap p-values shown; none significant at α=0.05)",
        fontsize=11.5, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_lspm_vs_naive_paired.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig4_ratio_sweep()
    fig5_paired_comparison()
    print("Wrote fig4_ratio_sweep.png and fig5_lspm_vs_naive_paired.png to", FIG_DIR)
