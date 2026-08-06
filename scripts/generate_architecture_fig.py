"""
Redesign of Figure 1 (system architecture / pipeline diagram) for a more
professional, publication-grade look: tighter canvas, restrained navy/slate
palette instead of washed-out pastel fills, the paper's actual contribution
(LSPM) visually distinguished from off-the-shelf components, a connected
annotation instead of floating italic text, and a labeled process-phase
bracket. Replaces results/figures/fig3_architecture.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Restrained, professional palette: slate/navy family with a single warm
# accent reserved for the paper's actual contribution (LSPM).
NAVY = "#1F3B57"
SLATE = "#44607A"
SLATE_FILL = "#EEF2F6"
SLATE_EDGE = "#8CA3B8"
ACCENT = "#C0622A"       # warm accent, used only for the LSPM box + its label
ACCENT_FILL = "#FBEEE4"
TEXT_DARK = "#1A1A1A"
ARROW_GRAY = "#5B6B7A"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(10.4, 3.15))
ax.axis("off")

box_w, box_h = 1.68, 1.05
y = 1.15
gap = 0.62

boxes = [
    ("User\nQuery", SLATE, SLATE_FILL, SLATE_EDGE, False),
    ("Vector DB\n(Chroma)", SLATE, SLATE_FILL, SLATE_EDGE, False),
    ("LSPM\nMiddleware", NAVY, ACCENT_FILL, ACCENT, True),
    ("vLLM Server\n(PagedAttention)", SLATE, SLATE_FILL, SLATE_EDGE, False),
    ("Streamed\nAnswer", SLATE, SLATE_FILL, SLATE_EDGE, False),
]

xs = []
x = 0.15
for _ in boxes:
    xs.append(x)
    x += box_w + gap
total_w = x - gap + 0.15

ax.set_xlim(0, total_w)
ax.set_ylim(0.0, 2.85)

for (label, text_color, fill, edge, is_contribution), bx in zip(boxes, xs):
    lw = 2.0 if is_contribution else 1.1
    rect = FancyBboxPatch(
        (bx, y), box_w, box_h,
        boxstyle="round,pad=0.045,rounding_size=0.09",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        path_effects=[pe.withSimplePatchShadow(offset=(0.6, -0.6), shadow_rgbFace="#B9C3CC", alpha=0.35)],
    )
    ax.add_patch(rect)
    ax.text(bx + box_w / 2, y + box_h / 2, label, ha="center", va="center",
             fontsize=9.8, color=TEXT_DARK, fontweight=("bold" if is_contribution else "normal"))

# Arrows between consecutive boxes
for i in range(len(boxes) - 1):
    x0 = xs[i] + box_w
    x1 = xs[i + 1]
    arrow = FancyArrowPatch((x0, y + box_h / 2), (x1, y + box_h / 2),
                             arrowstyle="-|>", mutation_scale=13,
                             color=ARROW_GRAY, linewidth=1.3)
    ax.add_patch(arrow)

# Leader line + annotation chip under the LSPM box (replaces floating text)
lspm_x = xs[2] + box_w / 2
leader_top = y
leader_bottom = 0.66
ax.plot([lspm_x, lspm_x], [leader_top, leader_bottom], color=ACCENT, linewidth=1.0, linestyle=(0, (2, 1.5)))

ann_w, ann_h = 3.55, 0.62
ann_x = lspm_x - ann_w / 2
ann = FancyBboxPatch(
    (ann_x, leader_bottom - ann_h), ann_w, ann_h,
    boxstyle="round,pad=0.04,rounding_size=0.07",
    linewidth=1.0, edgecolor=ACCENT, facecolor="white",
)
ax.add_patch(ann)
ax.text(lspm_x, leader_bottom - ann_h / 2,
         "sentence-level cross-encoder scoring\n+ fixed / dynamic compression ratio",
         ha="center", va="center", fontsize=7.8, color=ACCENT, style="italic")

# Phase bracket labeling "Retrieval" vs "Generation / Serving" above the boxes
bracket_y = y + box_h + 0.30
def phase_bracket(x_start, x_end, label):
    ax.annotate(
        "", xy=(x_end, bracket_y), xytext=(x_start, bracket_y),
        arrowprops=dict(arrowstyle="-", color=SLATE, linewidth=1.0),
    )
    ax.plot([x_start, x_start], [bracket_y - 0.05, bracket_y], color=SLATE, linewidth=1.0)
    ax.plot([x_end, x_end], [bracket_y - 0.05, bracket_y], color=SLATE, linewidth=1.0)
    ax.text((x_start + x_end) / 2, bracket_y + 0.09, label, ha="center", va="bottom",
             fontsize=8.3, color=SLATE, fontweight="bold")

phase_bracket(xs[1], xs[2] + box_w, "Retrieval + pruning")
phase_bracket(xs[3], xs[3] + box_w, "Generation")

fig.tight_layout(pad=0.3)
out_path = FIG_DIR / "fig3_architecture.png"
fig.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close(fig)
print("Wrote", out_path)
