"""
Statistical analysis of the end-to-end retrieval evaluation
(data/e2e_retrieval_results.jsonl, 60 questions x {raw, lspm_r0.3,
naive_r0.3} = 180 real NIM-hosted Llama-3.1-8B-Instruct generations).

Unlike the paper's main ARCD re-test (Section 4.4/5.4), which fixes a
6-document pool that always contains the gold, answer-bearing paragraph
(recall@6 = 100% by construction), this evaluation retrieves the top-6
documents for each question from a 234-paragraph corpus with a real TF-IDF
retriever (build_e2e_retrieval_eval.py); recall@6 here is 86.7% (52/60), so
8 of the 60 questions never had the answer-bearing paragraph in context at
all, for any method.

Reports, for each method (raw, LSPM r=0.3, naive truncation r=0.3):
  - overall mean EM/F1 across all 60 questions (the realistic, unconditional
    number a production deployment would actually see)
  - mean EM/F1 restricted to the 52 questions where retrieval succeeded
    (comparable in spirit to the main ARCD re-test's answer-guaranteed
    setting)
  - mean EM/F1 restricted to the 8 questions where retrieval failed
    (descriptive only -- n=8 is too small for inferential claims)
Plus the paired LSPM-vs-naive and each-vs-raw comparisons (Wilcoxon signed-
rank, paired bootstrap CI), both overall and on the retrieval-success subset.

Outputs:
  results/e2e_retrieval_summary.csv
  results/e2e_retrieval_pairwise.csv
  results/e2e_retrieval_report.txt
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.arabic_em_f1 import score_prediction

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "e2e_retrieval_results.jsonl"
EVAL_SET_PATH = ROOT / "data" / "e2e_retrieval_eval_set.json"
RNG_SEED = 42
N_BOOT = 10000

recs = [json.loads(l) for l in open(IN_PATH, encoding="utf-8") if l.strip()]
eval_set = json.load(open(EVAL_SET_PATH, encoding="utf-8"))
gold_in_topk_by_id = {item["id"]: item["gold_in_topk"] for item in eval_set}

by_method = {}
for r in recs:
    by_method.setdefault(r["method"], {})[r["id"]] = r

question_ids = sorted({r["id"] for r in recs})
assert len(question_ids) == 60, f"expected 60 questions, found {len(question_ids)}"
n_hit = sum(1 for qid in question_ids if gold_in_topk_by_id[qid])
n_miss = len(question_ids) - n_hit
hit_ids = [qid for qid in question_ids if gold_in_topk_by_id[qid]]
miss_ids = [qid for qid in question_ids if not gold_in_topk_by_id[qid]]

scored = {}  # method -> {qid: {"em":..., "f1":...}}
for method, id_map in by_method.items():
    scored[method] = {}
    for qid, r in id_map.items():
        scored[method][qid] = score_prediction(r["answer"], r["gold_answers"])

METHODS = ["raw", "lspm", "naive"]
rng = np.random.default_rng(RNG_SEED)


def bootstrap_ci(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci(method, ids, metric):
    vals = [scored[method][qid][metric] for qid in ids]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = bootstrap_ci(vals)
    return float(np.mean(vals)), lo, hi, len(vals)


summary_rows = []
for method in METHODS:
    for subset_name, ids in [("all", question_ids), ("retrieval_hit", hit_ids), ("retrieval_miss", miss_ids)]:
        em_mean, em_lo, em_hi, n = mean_ci(method, ids, "em")
        f1_mean, f1_lo, f1_hi, _ = mean_ci(method, ids, "f1")
        summary_rows.append({
            "method": method, "subset": subset_name, "n": n,
            "mean_em": em_mean, "em_ci_lo": em_lo, "em_ci_hi": em_hi,
            "mean_f1": f1_mean, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
        })

pairwise_rows = []


def paired_compare(name_a, method_a, name_b, method_b, metric, ids, subset_name):
    if len(ids) < 2:
        return
    vals_a = np.array([scored[method_a][qid][metric] for qid in ids])
    vals_b = np.array([scored[method_b][qid][metric] for qid in ids])
    diffs = vals_a - vals_b
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci(diffs)
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
        "comparison": f"{name_a} vs {name_b}", "metric": metric, "subset": subset_name, "n": len(ids),
        "mean_a": float(vals_a.mean()), "mean_b": float(vals_b.mean()),
        "mean_diff_a_minus_b": mean_diff, "diff_ci_lo": ci_lo, "diff_ci_hi": ci_hi,
        "n_pos": n_pos, "n_neg": n_neg, "n_tied": n_tied,
        "sign_test_p": sign_p, "wilcoxon_p": wilcoxon_p,
    })


for subset_name, ids in [("all", question_ids), ("retrieval_hit", hit_ids)]:
    for metric in ["em", "f1"]:
        paired_compare("lspm", "lspm", "naive", "naive", metric, ids, subset_name)
        paired_compare("lspm", "lspm", "raw", "raw", metric, ids, subset_name)
        paired_compare("naive", "naive", "raw", "raw", metric, ids, subset_name)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "e2e_retrieval_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

with open(RESULTS_DIR / "e2e_retrieval_pairwise.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
    w.writeheader()
    w.writerows(pairwise_rows)

lines = []
lines.append("End-to-end retrieval evaluation: statistical summary")
lines.append(f"n = {len(question_ids)} questions, top-6 documents from a real TF-IDF retriever over a")
lines.append("234-paragraph corpus (recall@6 is NOT guaranteed). Scored with Arabic-normalized EM/F1.")
lines.append(f"Retrieval recall@6: {n_hit}/{len(question_ids)} = {100*n_hit/len(question_ids):.1f}% "
             f"(gold paragraph retrieved); {n_miss}/{len(question_ids)} = {100*n_miss/len(question_ids):.1f}% miss.")
lines.append("Bootstrap CIs: 10,000 percentile resamples, seed=42.")
lines.append("")
lines.append("-- Per-cell means (subset = all / retrieval_hit / retrieval_miss) --")
for row in summary_rows:
    lines.append(
        f"{row['method']:6s} [{row['subset']:15s} n={row['n']:2d}]  "
        f"EM={row['mean_em']:.3f} [{row['em_ci_lo']:.3f}, {row['em_ci_hi']:.3f}]   "
        f"F1={row['mean_f1']:.3f} [{row['f1_ci_lo']:.3f}, {row['f1_ci_hi']:.3f}]"
    )
lines.append("")
lines.append("-- Paired comparisons --")
for row in pairwise_rows:
    lines.append(
        f"[{row['subset']:14s} n={row['n']:2d}] {row['comparison']:14s} [{row['metric']}]  "
        f"mean_diff={row['mean_diff_a_minus_b']:+.3f} [{row['diff_ci_lo']:+.3f}, {row['diff_ci_hi']:+.3f}]  "
        f"sign p={row['sign_test_p']:.3f}  wilcoxon p={row['wilcoxon_p']:.3f}"
    )

report = "\n".join(lines)
with open(RESULTS_DIR / "e2e_retrieval_report.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(report)
print(f"\nWrote results/e2e_retrieval_summary.csv, results/e2e_retrieval_pairwise.csv, results/e2e_retrieval_report.txt")
