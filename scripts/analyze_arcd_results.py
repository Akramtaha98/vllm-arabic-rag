"""
Statistical analysis of the expanded ARCD ground-truth accuracy experiment
(data/arcd_results.jsonl, 140 questions x 7 method/ratio cells = 980 real
NIM-hosted Llama-3.1-8B-Instruct generations; an earlier n=40/280-cell run
is also supported for backward compatibility).

Unlike the paper's original pilot -- which only measured similarity of a
pruned-context answer to the *unpruned-context answer* (ROUGE-L/BERTScore/
BLEU) -- this scores every answer against ARCD's real human-written gold
answer string(s) using Arabic-normalized Exact Match (EM) and token-F1
(eval/arabic_em_f1.py), i.e. ground-truth correctness, not self-similarity.

For each of LSPM and naive-truncation at each ratio in {0.3, 0.5, 0.7}:
  - mean EM, mean F1, with percentile bootstrap 95% CIs (10,000 resamples)
  - paired comparison vs. the raw/unpruned answer (fidelity loss relative
    to no pruning) -- Wilcoxon signed-rank test + paired bootstrap CI on
    the mean difference
  - paired comparison LSPM vs. naive truncation at the same ratio (the
    paper's core novelty claim: does semantic pruning beat naive length
    truncation?) -- same paired tests

Outputs:
  results/arcd_stats_summary.csv   -- one row per method x ratio
  results/arcd_stats_pairwise.csv  -- one row per pairwise comparison
  results/arcd_stats_report.txt    -- human-readable narrative summary
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.arabic_em_f1 import score_prediction

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "arcd_results.jsonl"
RNG_SEED = 42
N_BOOT = 10000

recs = [json.loads(l) for l in open(IN_PATH, encoding="utf-8") if l.strip()]
by_id = {}
for r in recs:
    key = (r["method"], r["ratio"])
    by_id.setdefault(key, {})[r["id"]] = r

# Score every cell against its gold answer(s).
scored = {}  # (method, ratio) -> {qid: {"em":..., "f1":...}}
for key, id_map in by_id.items():
    scored[key] = {}
    for qid, r in id_map.items():
        scored[key][qid] = score_prediction(r["answer"], r["gold_answers"])

question_ids = sorted({r["id"] for r in recs})
assert len(question_ids) in (40, 140), f"expected 40 or 140 questions, found {len(question_ids)}"

METHODS_RATIOS = [
    ("raw", 1.0),
    ("lspm", 0.3), ("naive", 0.3),
    ("lspm", 0.5), ("naive", 0.5),
    ("lspm", 0.7), ("naive", 0.7),
]

rng = np.random.default_rng(RNG_SEED)


def bootstrap_ci(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_ci(diffs, n_boot=N_BOOT):
    return bootstrap_ci(diffs, n_boot=n_boot)


summary_rows = []
for method, ratio in METHODS_RATIOS:
    em_vals = [scored[(method, ratio)][qid]["em"] for qid in question_ids]
    f1_vals = [scored[(method, ratio)][qid]["f1"] for qid in question_ids]
    em_lo, em_hi = bootstrap_ci(em_vals)
    f1_lo, f1_hi = bootstrap_ci(f1_vals)
    summary_rows.append({
        "method": method, "ratio": ratio, "n": len(em_vals),
        "mean_em": np.mean(em_vals), "em_ci_lo": em_lo, "em_ci_hi": em_hi,
        "mean_f1": np.mean(f1_vals), "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
    })

pairwise_rows = []


def paired_compare(name_a, key_a, name_b, key_b, metric):
    vals_a = np.array([scored[key_a][qid][metric] for qid in question_ids])
    vals_b = np.array([scored[key_b][qid][metric] for qid in question_ids])
    diffs = vals_a - vals_b
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = paired_bootstrap_ci(diffs)
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    n_tied = int((diffs == 0).sum())
    if n_pos + n_neg == 0:
        sign_p = 1.0
    else:
        sign_p = sstats.binomtest(min(n_pos, n_neg), n_pos + n_neg, 0.5).pvalue
    if np.allclose(diffs, 0):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = sstats.wilcoxon(vals_a, vals_b, zero_method="wilcox")
        except ValueError:
            wilcoxon_p = float("nan")
    pairwise_rows.append({
        "comparison": f"{name_a} vs {name_b}", "metric": metric,
        "mean_a": float(vals_a.mean()), "mean_b": float(vals_b.mean()),
        "mean_diff_a_minus_b": mean_diff, "diff_ci_lo": ci_lo, "diff_ci_hi": ci_hi,
        "n_pos": n_pos, "n_neg": n_neg, "n_tied": n_tied,
        "sign_test_p": sign_p, "wilcoxon_p": wilcoxon_p,
    })


for ratio in [0.3, 0.5, 0.7]:
    for metric in ["em", "f1"]:
        # Core novelty claim: LSPM vs naive truncation, same ratio.
        paired_compare(f"lspm_r{ratio}", ("lspm", ratio), f"naive_r{ratio}", ("naive", ratio), metric)
        # Fidelity loss relative to unpruned raw answer.
        paired_compare(f"lspm_r{ratio}", ("lspm", ratio), "raw", ("raw", 1.0), metric)
        paired_compare(f"naive_r{ratio}", ("naive", ratio), "raw", ("raw", 1.0), metric)

# --- Holm-Bonferroni correction across the natural 9-test F1 family ---
# (3 ratios x {lspm-vs-naive, naive-vs-raw, lspm-vs-raw}), matching the
# family reported in the paper's Table 5. Uncorrected multiple significance
# tests are not by themselves evidence; see Section 5.4/6 of the paper for
# why this correction was added after an initial (invalid) reading of the
# uncorrected naive-vs-raw p-values as an independent finding.
F1_FAMILY_ORDER = []
for ratio in [0.3, 0.5, 0.7]:
    F1_FAMILY_ORDER.append(f"lspm_r{ratio} vs naive_r{ratio}")
for ratio in [0.3, 0.5, 0.7]:
    F1_FAMILY_ORDER.append(f"naive_r{ratio} vs raw")
for ratio in [0.3, 0.5, 0.7]:
    F1_FAMILY_ORDER.append(f"lspm_r{ratio} vs raw")

f1_rows_by_comparison = {r["comparison"]: r for r in pairwise_rows if r["metric"] == "f1"}


def holm_bonferroni(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(max((n - rank) * pvals[idx], prev), 1.0)
        adj[idx] = val
        prev = val
    return adj


holm_pvals = [f1_rows_by_comparison[c]["wilcoxon_p"] for c in F1_FAMILY_ORDER]
holm_adj = holm_bonferroni(holm_pvals)
for comp, adj_p in zip(F1_FAMILY_ORDER, holm_adj):
    f1_rows_by_comparison[comp]["holm_adjusted_p"] = float(adj_p)
for row in pairwise_rows:
    row.setdefault("holm_adjusted_p", "")


# --- TOST equivalence test (predefined margin, ±0.05 F1) ---
def tost_paired(vals_a, vals_b, margin, alpha=0.05):
    diffs = np.asarray(vals_a) - np.asarray(vals_b)
    n = len(diffs)
    mean_d = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return mean_d, 0.0 if abs(mean_d) < margin else 1.0
    t1 = (mean_d - (-margin)) / se
    p1 = 1 - sstats.t.cdf(t1, df=n - 1)
    t2 = (mean_d - margin) / se
    p2 = sstats.t.cdf(t2, df=n - 1)
    return float(mean_d), float(max(p1, p2))


TOST_MARGIN = 0.05
tost_rows = []
for ratio in [0.3, 0.5, 0.7]:
    for name_a, key_a in [(f"lspm_r{ratio}", ("lspm", ratio)), (f"naive_r{ratio}", ("naive", ratio))]:
        vals_a = [scored[key_a][qid]["f1"] for qid in question_ids]
        vals_b = [scored[("raw", 1.0)][qid]["f1"] for qid in question_ids]
        mean_d, tost_p = tost_paired(vals_a, vals_b, TOST_MARGIN)
        tost_rows.append({
            "comparison": f"{name_a} vs raw", "metric": "f1", "margin": TOST_MARGIN,
            "mean_diff": mean_d, "tost_p": tost_p,
            "equivalent": tost_p < 0.05,
        })

# --- Write outputs ---
import csv

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "arcd_stats_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

with open(RESULTS_DIR / "arcd_stats_pairwise.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
    w.writeheader()
    w.writerows(pairwise_rows)

with open(RESULTS_DIR / "arcd_stats_tost.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(tost_rows[0].keys()))
    w.writeheader()
    w.writerows(tost_rows)

lines = []
lines.append("ARCD ground-truth accuracy experiment: statistical summary")
lines.append(f"n = {len(question_ids)} questions, 6-document noisy retrieval pool (1 gold + 5 distractors),")
lines.append("scored with Arabic-normalized Exact Match (EM) and token-F1 against ARCD's real gold answers.")
lines.append("Bootstrap CIs: 10,000 percentile resamples, seed=42.")
lines.append("")
lines.append("-- Per-cell means --")
for row in summary_rows:
    lines.append(
        f"{row['method']:6s} r={row['ratio']:.1f}  "
        f"EM={row['mean_em']:.3f} [{row['em_ci_lo']:.3f}, {row['em_ci_hi']:.3f}]   "
        f"F1={row['mean_f1']:.3f} [{row['f1_ci_lo']:.3f}, {row['f1_ci_hi']:.3f}]"
    )
lines.append("")
lines.append("-- Core novelty claim: LSPM vs. naive truncation (same ratio) --")
for row in pairwise_rows:
    if row["comparison"].startswith("lspm_r") and "naive_r" in row["comparison"]:
        sig = "significant" if row["wilcoxon_p"] < 0.05 else "not significant"
        holm_str = f"  holm-adj p={row['holm_adjusted_p']:.3f}" if row["metric"] == "f1" else ""
        lines.append(
            f"{row['comparison']:22s} [{row['metric']}]  "
            f"mean_diff={row['mean_diff_a_minus_b']:+.3f} "
            f"[{row['diff_ci_lo']:+.3f}, {row['diff_ci_hi']:+.3f}]  "
            f"sign p={row['sign_test_p']:.3f}  wilcoxon p={row['wilcoxon_p']:.3f}{holm_str}  ({sig} at alpha=0.05, uncorrected)"
        )
lines.append("")
lines.append("-- Fidelity loss vs. raw (unpruned) answer --")
for row in pairwise_rows:
    if row["comparison"].endswith("vs raw"):
        holm_str = f"  holm-adj p={row['holm_adjusted_p']:.3f}" if row["metric"] == "f1" else ""
        lines.append(
            f"{row['comparison']:22s} [{row['metric']}]  "
            f"mean_diff={row['mean_diff_a_minus_b']:+.3f} "
            f"[{row['diff_ci_lo']:+.3f}, {row['diff_ci_hi']:+.3f}]  "
            f"wilcoxon p={row['wilcoxon_p']:.3f}{holm_str}"
        )
lines.append("")
lines.append(f"-- Holm-Bonferroni correction across the 9-test F1 family (n={len(question_ids)}) --")
for comp in F1_FAMILY_ORDER:
    row = f1_rows_by_comparison[comp]
    sig = "SIGNIFICANT" if row["holm_adjusted_p"] < 0.05 else "n.s."
    lines.append(f"{comp:26s} raw p={row['wilcoxon_p']:.4f}  holm-adjusted p={row['holm_adjusted_p']:.4f}  ({sig})")
lines.append("")
lines.append(f"-- TOST equivalence test (margin=+/-{TOST_MARGIN} F1, n={len(question_ids)}) --")
for row in tost_rows:
    verdict = "EQUIVALENT" if row["equivalent"] else "inconclusive"
    lines.append(f"{row['comparison']:22s} mean_diff={row['mean_diff']:+.4f}  TOST p={row['tost_p']:.4f}  ({verdict})")

report = "\n".join(lines)
with open(RESULTS_DIR / "arcd_stats_report.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(report)
print(f"\nWrote results/arcd_stats_summary.csv, results/arcd_stats_pairwise.csv, results/arcd_stats_report.txt")
