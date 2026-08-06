"""
Generate publication-quality figures from the real pilot data collected for
the paper: tokenization disparity, semantic fidelity, and the system
architecture diagram. APA-ish styling: colorblind-safe palette, serif-free,
300 DPI.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette (Okabe-Ito)
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


def fig1_tokenization_disparity():
    def load_ratios(fname):
        with open(ROOT / "results" / fname) as f:
            rows = list(csv.DictReader(f))
        return [float(r["ratio_ar_over_en"]) for r in rows]

    qwen_ratios = load_ratios("tokenization_disparity.csv")
    llama_ratios = load_ratios("tokenization_disparity_llama.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(
        [qwen_ratios, llama_ratios],
        tick_labels=["Qwen2.5-7B\ntokenizer", "Llama-3.1-8B\ntokenizer"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black"},
    )
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_ylabel("Arabic / English token-count ratio")
    ax.set_title(
        f"Arabic-English tokenization disparity\n(n={len(qwen_ratios)} parallel sentence pairs)",
        fontsize=12,
    )
    # Label the parity line directly (outside the box/whisker/outlier region)
    # instead of a legend box, which previously overlapped an outlier marker.
    ax.set_xlim(0.35, 2.65)
    ax.text(2.62, 1.0, "Parity (AR = EN tokens)", ha="right", va="bottom",
            fontsize=8.5, color=GRAY, style="italic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_tokenization_disparity.png", bbox_inches="tight")
    plt.close(fig)


def fig2_fidelity_metrics():
    with open(ROOT / "results/fidelity.csv") as f:
        rows = list(csv.DictReader(f))

    rouge = [float(r["rougeL_pruned_vs_raw"]) for r in rows]
    bert = [float(r["bertscore_f1_pruned_vs_raw"]) for r in rows]
    bleu = [float(r["bleu_pruned_vs_raw"]) / 100 for r in rows]  # normalize to 0-1

    metrics = ["BLEU / 100", "ROUGE-L", "BERTScore-F1"]
    means = [sum(bleu) / len(bleu), sum(rouge) / len(rouge), sum(bert) / len(bert)]
    mins = [min(bleu), min(rouge), min(bert)]
    maxs = [max(bleu), max(rouge), max(bert)]
    err_low = [m - mn for m, mn in zip(means, mins)]
    err_high = [mx - m for m, mx in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(metrics))
    bars = ax.bar(x, means, yerr=[err_low, err_high], capsize=5,
                   color=[BLUE, GREEN, ORANGE], alpha=0.75, width=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0-1, higher = more similar\nto raw-context answer)")
    ax.set_title(
        f"Semantic fidelity of pruned vs. raw-context answers\n(n={len(rows)}, compression ratio=0.5)",
        fontsize=12,
    )
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_semantic_fidelity.png", bbox_inches="tight")
    plt.close(fig)


def fig3_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 3)

    boxes = [
        (0.3, "User\nQuery", GRAY),
        (2.2, "Vector DB\n(Chroma)", BLUE),
        (4.4, "LSPM\nMiddleware", ORANGE),
        (6.6, "vLLM Server\n(PagedAttention)", GREEN),
        (9.0, "Streamed\nAnswer", GRAY),
    ]
    box_w, box_h = 1.7, 1.2
    y = 0.9

    for x, label, color in boxes:
        rect = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.18,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=9.5, color="black")

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + box_w
        x1 = boxes[i + 1][0]
        arrow = FancyArrowPatch((x0, y + box_h / 2), (x1, y + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=14, color=GRAY, linewidth=1.2)
        ax.add_patch(arrow)

    ax.text(4.4 + box_w / 2, y - 0.35, "sentence-level cross-encoder scoring +\nfixed/dynamic compression ratio",
            ha="center", fontsize=8, color=ORANGE, style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_architecture.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_tokenization_disparity()
    fig2_fidelity_metrics()
    fig3_architecture()
    print("Figures written to", FIG_DIR)
