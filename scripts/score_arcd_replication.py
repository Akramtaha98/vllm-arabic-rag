"""
Scores data/arcd_replication_results.jsonl (the fresh-seed disjoint
replication, Section 4.10) with the SAME single scorer and SAME
statistical methodology used for every other number in the paper
(scripts/score_arcd.py; bootstrap 95% CI seed=42, 10,000 resamples;
paired Wilcoxon signed-rank + sign test; Holm-Bonferroni within the
3-comparison family; TOST equivalence margin=+/-0.05 F1).

This is the 3-condition, r=0.3-only analogue of Table 2's 7-condition
version. Run after run_arcd_replication.py reports all 420 cells complete.

Usage:
    python3 scripts/score_arcd_replication.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from score_arcd import score  # noqa: E402

RNG_SEED = 42
N_BOOT = 10000


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def score_file(path):
    recs = load(path)
    by_id = {}
    for r in recs:
        by_id.setdefault((r["method"], r["ratio"]), {})[r["id"]] = r
    scored = {}
    for key, id_map in by_id.items():
        scored[key] = {}
        for qid, r in id_map.items():
            scored[key][qid] = score(r["answer"], r["gold_answers"])
    return scored, sorted({r["id"] for r in recs})


def bootstrap_ci(values, seed=RNG_SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_compare(scored, qids, key_a, key_b, metric):
    vals_a = np.array([scored[key_a][qid][metric] for qid in qids])
    vals_b = np.array([scored[key_b][qid][metric] for qid in qids])
    diffs = vals_a - vals_b
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci(diffs)
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    sign_p = 1.0 if n_pos + n_neg == 0 else sstats.binomtest(min(n_pos, n_neg), n_pos + n_neg, 0.5).pvalue
    if np.allclose(diffs, 0):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = sstats.wilcoxon(vals_a, vals_b, zero_method="wilcox")
        except ValueError:
            wilcoxon_p = float("nan")
    return {
        "mean_a": float(vals_a.mean()), "mean_b": float(vals_b.mean()),
        "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_pos": n_pos, "n_neg": n_neg, "sign_p": sign_p, "wilcoxon_p": wilcoxon_p,
    }


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


def tost_paired(vals_a, vals_b, margin):
    diffs = np.asarray(vals_a) - np.asarray(vals_b)
    n = len(diffs)
    mean_d = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float(mean_d), 0.0 if abs(mean_d) < margin else 1.0
    t1 = (mean_d - (-margin)) / se
    p1 = 1 - sstats.t.cdf(t1, df=n - 1)
    t2 = (mean_d - margin) / se
    p2 = sstats.t.cdf(t2, df=n - 1)
    return float(mean_d), float(max(p1, p2))


def main():
    path = ROOT / "data" / "arcd_replication_results.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run run_arcd_replication.py first.")

    scored, qids = score_file(path)
    expected_cells = {("raw", 1.0), ("lspm", 0.3), ("naive", 0.3)}
    missing = expected_cells - set(scored.keys())
    if missing:
        raise SystemExit(f"Missing conditions: {missing}. Run run_arcd_replication.py to completion first.")
    n = len(qids)
    print(f"=== Fresh-seed disjoint replication, r=0.3 (n={n}) ===\n")

    for key in [("raw", 1.0), ("lspm", 0.3), ("naive", 0.3)]:
        em = [scored[key][q]["em"] for q in qids]
        f1 = [scored[key][q]["f1"] for q in qids]
        em_lo, em_hi = bootstrap_ci(em)
        f1_lo, f1_hi = bootstrap_ci(f1)
        print(f"{key}: EM={np.mean(em):.4f} [{em_lo:.4f},{em_hi:.4f}]  "
              f"F1={np.mean(f1):.4f} [{f1_lo:.4f},{f1_hi:.4f}]")

    print("\n-- F1, 3-test family (Holm-Bonferroni) --")
    family_order = ["lspm vs naive", "naive vs raw", "lspm vs raw"]
    pairs = [(("lspm", 0.3), ("naive", 0.3)), (("naive", 0.3), ("raw", 1.0)), (("lspm", 0.3), ("raw", 1.0))]
    results = [paired_compare(scored, qids, a, b, "f1") for a, b in pairs]
    holm = holm_bonferroni([r["wilcoxon_p"] for r in results])
    for name, r, adj in zip(family_order, results, holm):
        print(f"{name:16s} mean_diff={r['mean_diff']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
              f"raw_p={r['wilcoxon_p']:.6f} holm_p={adj:.6f} "
              f"(sign test: {r['n_pos']}+/{r['n_neg']}-, p={r['sign_p']:.6f})")

    print("\n-- EM, lspm vs naive (uncorrected) --")
    c = paired_compare(scored, qids, ("lspm", 0.3), ("naive", 0.3), "em")
    print(f"mean_diff={c['mean_diff']:+.4f} [{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] wilcoxon_p={c['wilcoxon_p']:.6f}")

    print("\n-- TOST equivalence vs raw (margin=+/-0.05 F1) --")
    for name, key in [("lspm", ("lspm", 0.3)), ("naive", ("naive", 0.3))]:
        va = [scored[key][q]["f1"] for q in qids]
        vb = [scored[("raw", 1.0)][q]["f1"] for q in qids]
        mean_d, tost_p = tost_paired(va, vb, 0.05)
        print(f"{name} vs raw: mean_diff={mean_d:+.4f} tost_p={tost_p:.6f} "
              f"({'EQUIVALENT' if tost_p < 0.05 else 'inconclusive'})")

    print(f"\nCross-check: does this replicate the original r=0.3 direction "
          f"(LSPM > naive, both > raw baseline gap)? Compare the mean_diff/holm_p "
          f"above to the original Table 2 r=0.3 row and paste both into the chat "
          f"once you have this -- that pairing is what goes into the new Section 4.10.")


if __name__ == "__main__":
    main()
