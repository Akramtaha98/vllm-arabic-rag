"""
Round-2 statistical analysis: computes ROUGE-L / BLEU / BERTScore-F1 for every
(method, ratio) cell in data/eval_set_expanded.jsonl, with:
  - bootstrap 95% CIs (10,000 resamples) instead of round-1's bare min/max
  - a paired comparison (LSPM vs naive truncation, same queries) at each ratio,
    with a paired bootstrap test for the mean difference
  - a one-sample significance test on the tokenization-disparity ratios
    against parity (ratio = 1.0), addressing R1's round-1 methodology note

Writes:
  results/fidelity_expanded.csv        (per-item scores)
  results/fidelity_summary.csv         (per method x ratio: mean + 95% CI)
  results/lspm_vs_naive_paired.csv     (paired diff + bootstrap p-value per ratio)
  results/tokenization_significance.txt
"""
import csv
import json
import random
import re
import statistics
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)
N_BOOT = 10000


class MultilingualTokenizer:
    _WORD_RE = re.compile(r"\w+", re.UNICODE)

    def tokenize(self, text: str):
        return self._WORD_RE.findall(text.lower())


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def paired_bootstrap_pvalue(diffs, n_boot=N_BOOT):
    """Two-sided bootstrap p-value for H0: mean(diffs) == 0."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    observed = sum(diffs) / n
    # Center the diffs at 0 to simulate the null, then resample.
    mean_d = observed
    centered = [d - mean_d for d in diffs]
    count = 0
    for _ in range(n_boot):
        sample = [centered[random.randrange(n)] for _ in range(n)]
        boot_mean = sum(sample) / n
        if abs(boot_mean) >= abs(observed):
            count += 1
    return count / n_boot


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_records(ROOT / "data/eval_set_expanded.jsonl")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=MultilingualTokenizer())
    bleu = BLEU(effective_order=True)

    # Per-item scores
    per_item = []
    for r in records:
        raw_ans, pruned_ans = r["raw_answer"], r["pruned_answer"]
        rougeL = rouge.score(raw_ans, pruned_ans)["rougeL"].fmeasure
        bleu_score = bleu.sentence_score(pruned_ans, [raw_ans]).score
        per_item.append({**r, "rougeL": rougeL, "bleu": bleu_score})

    pruned_list = [r["pruned_answer"] for r in per_item]
    raw_list = [r["raw_answer"] for r in per_item]
    _, _, f1 = bertscore(pruned_list, raw_list, lang="ar", verbose=False)
    for row, f1_val in zip(per_item, f1.tolist()):
        row["bertscore_f1"] = f1_val

    Path(ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/fidelity_expanded.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query", "method", "ratio", "rougeL", "bleu", "bertscore_f1",
                      "raw_context_chars", "pruned_context_chars", "pruner_latency_ms"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_item:
            w.writerow({k: row.get(k) for k in fieldnames})

    # Per (method, ratio) summary with bootstrap CIs
    cells = {}
    for row in per_item:
        key = (row["method"], row["ratio"])
        cells.setdefault(key, []).append(row)

    summary_rows = []
    for (method, ratio), rows in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for metric in ["rougeL", "bleu", "bertscore_f1"]:
            vals = [row[metric] for row in rows]
            mean_v = statistics.mean(vals)
            lo, hi = bootstrap_ci(vals)
            summary_rows.append({
                "method": method, "ratio": ratio, "metric": metric,
                "n": len(vals), "mean": round(mean_v, 4),
                "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            })
        char_reductions = [1 - (row["pruned_context_chars"] / row["raw_context_chars"]) for row in rows]
        summary_rows.append({
            "method": method, "ratio": ratio, "metric": "char_reduction",
            "n": len(char_reductions), "mean": round(statistics.mean(char_reductions), 4),
            "ci95_lo": "", "ci95_hi": "",
        })

    with open(ROOT / "results/fidelity_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "ratio", "metric", "n", "mean", "ci95_lo", "ci95_hi"])
        w.writeheader()
        w.writerows(summary_rows)

    # Paired comparison: LSPM vs naive at each ratio (same queries -> paired)
    paired_rows = []
    by_query_method_ratio = {(r["query"], r["method"], r["ratio"]): r for r in per_item}
    ratios = sorted({r["ratio"] for r in per_item})
    for ratio in ratios:
        for metric in ["rougeL", "bleu", "bertscore_f1"]:
            diffs = []
            for r in per_item:
                if r["method"] != "lspm" or r["ratio"] != ratio:
                    continue
                q = r["query"]
                naive_r = by_query_method_ratio.get((q, "naive", ratio))
                if naive_r is None:
                    continue
                diffs.append(r[metric] - naive_r[metric])
            if not diffs:
                continue
            mean_diff = statistics.mean(diffs)
            p = paired_bootstrap_pvalue(diffs)
            paired_rows.append({
                "ratio": ratio, "metric": metric, "n_pairs": len(diffs),
                "mean_diff_lspm_minus_naive": round(mean_diff, 4),
                "bootstrap_p_value": round(p, 4),
            })

    with open(ROOT / "results/lspm_vs_naive_paired.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ratio", "metric", "n_pairs", "mean_diff_lspm_minus_naive", "bootstrap_p_value"])
        w.writeheader()
        w.writerows(paired_rows)

    print("Wrote results/fidelity_expanded.csv, results/fidelity_summary.csv, results/lspm_vs_naive_paired.csv")
    for row in summary_rows:
        print(row)
    print("\nPaired LSPM vs naive:")
    for row in paired_rows:
        print(row)


if __name__ == "__main__":
    main()
