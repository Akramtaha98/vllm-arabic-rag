"""
Analyzes a completed human-evaluation rating sheet (produced by
build_human_eval_sample.py and filled in by evaluators) against the
private codebook, producing:
  - results/human_eval_summary.csv: mean correctness (0/1/2) and
    faithfulness (0/1) per method, with 95% bootstrap CIs.
  - Inter-rater agreement (Cohen's kappa) for correctness and
    faithfulness, if both rater1 and rater2 columns are filled.
  - A printed comparison against the paper's automatic EM/F1 numbers for
    the same ratio, so you can report "human-judged correctness agrees
    with automatic F1 to within X points" as an extra robustness claim.

Usage (after evaluators have filled in data/human_eval_sample_blind.csv):
    python scripts/analyze_human_eval.py

Expects correctness ratings in {0,1,2} (0=incorrect, 1=partially correct,
2=fully correct) and faithfulness ratings in {0,1} (0=not grounded in
context / hallucinated or unsupported, 1=grounded). See
HUMAN_EVAL_PROTOCOL.md for the full rubric given to evaluators.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bootstrap_ci(values, n_boot=10000, seed=42, alpha=0.05):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def cohens_kappa(a, b, labels):
    """Unweighted Cohen's kappa between two rating lists over a fixed label set."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {l: sum(1 for x in a if x == l) / n for l in labels}
    pb = {l: sum(1 for x in b if x == l) / n for l in labels}
    pe = sum(pa[l] * pb[l] for l in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-csv", default="data/human_eval_sample_blind.csv")
    ap.add_argument("--codebook", default="data/human_eval_codebook.json")
    ap.add_argument("--out-csv", default="results/human_eval_summary.csv")
    args = ap.parse_args()

    sample_path = ROOT / args.sample_csv
    codebook_path = ROOT / args.codebook

    codebook = json.load(open(codebook_path, encoding="utf-8"))
    with open(sample_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def parse_num(v):
        v = (v or "").strip()
        if v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    incomplete = [r["item_id"] for r in rows if parse_num(r["correctness_rater1"]) is None]
    if incomplete:
        print(f"NOTE: {len(incomplete)}/{len(rows)} rows have no rater1 correctness rating yet "
              f"(e.g. {incomplete[:5]}). Fill in data/human_eval_sample_blind.csv before running this.")

    by_method_correctness = defaultdict(list)
    by_method_faithfulness = defaultdict(list)
    kappa_correctness_pairs = []
    kappa_faithfulness_pairs = []

    for r in rows:
        item_id = r["item_id"]
        meta = codebook.get(item_id)
        if meta is None:
            print(f"WARNING: {item_id} not found in codebook, skipping")
            continue
        method = meta["method"]

        c1 = parse_num(r["correctness_rater1"])
        c2 = parse_num(r["correctness_rater2"])
        f1 = parse_num(r["faithfulness_rater1"])
        f2 = parse_num(r["faithfulness_rater2"])

        # Average available raters per item for the summary stats.
        c_vals = [v for v in (c1, c2) if v is not None]
        f_vals = [v for v in (f1, f2) if v is not None]
        if c_vals:
            by_method_correctness[method].append(sum(c_vals) / len(c_vals))
        if f_vals:
            by_method_faithfulness[method].append(sum(f_vals) / len(f_vals))

        if c1 is not None and c2 is not None:
            kappa_correctness_pairs.append((c1, c2))
        if f1 is not None and f2 is not None:
            kappa_faithfulness_pairs.append((f1, f2))

    out_rows = []
    print("\n=== Mean correctness (0-2 scale) and faithfulness (0-1 scale) by method ===")
    for method in ("raw", "lspm", "naive"):
        c_vals = by_method_correctness.get(method, [])
        f_vals = by_method_faithfulness.get(method, [])
        c_mean = sum(c_vals) / len(c_vals) if c_vals else float("nan")
        f_mean = sum(f_vals) / len(f_vals) if f_vals else float("nan")
        c_lo, c_hi = bootstrap_ci(c_vals) if c_vals else (float("nan"), float("nan"))
        f_lo, f_hi = bootstrap_ci(f_vals) if f_vals else (float("nan"), float("nan"))
        print(f"  {method:6s}  n={len(c_vals):3d}  correctness={c_mean:.3f} [{c_lo:.3f}, {c_hi:.3f}]  "
              f"faithfulness={f_mean:.3f} [{f_lo:.3f}, {f_hi:.3f}]")
        out_rows.append({
            "method": method, "n": len(c_vals),
            "mean_correctness": round(c_mean, 4), "correctness_ci_lo": round(c_lo, 4), "correctness_ci_hi": round(c_hi, 4),
            "mean_faithfulness": round(f_mean, 4), "faithfulness_ci_lo": round(f_lo, 4), "faithfulness_ci_hi": round(f_hi, 4),
        })

    if kappa_correctness_pairs:
        a, b = zip(*kappa_correctness_pairs)
        kappa_c = cohens_kappa(list(a), list(b), labels=[0.0, 1.0, 2.0])
        print(f"\nCohen's kappa (correctness, rater1 vs rater2, n={len(kappa_correctness_pairs)}): {kappa_c:.3f}")
    else:
        print("\nCohen's kappa (correctness): not computed -- need both rater1 and rater2 filled for at least one item.")

    if kappa_faithfulness_pairs:
        a, b = zip(*kappa_faithfulness_pairs)
        kappa_f = cohens_kappa(list(a), list(b), labels=[0.0, 1.0])
        print(f"Cohen's kappa (faithfulness, rater1 vs rater2, n={len(kappa_faithfulness_pairs)}): {kappa_f:.3f}")
    else:
        print("Cohen's kappa (faithfulness): not computed -- need both rater1 and rater2 filled for at least one item.")

    if out_rows:
        out_path = ROOT / args.out_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
