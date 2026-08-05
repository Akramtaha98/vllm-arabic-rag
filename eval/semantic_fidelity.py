"""
Semantic Fidelity Preservation evaluation.
--------------------------------------------
Compares model answers generated from the RAW (unpruned) context vs. the
PRUNED (LSPM) context, using ROUGE-L, BLEU, and BERTScore, to show that
pruning does not materially degrade answer quality.

Usage:
    python eval/semantic_fidelity.py --data data/eval_set.jsonl --out results/fidelity.csv

Input JSONL, one record per QA item:
    {
      "query": "...",
      "raw_answer": "<answer generated from unpruned context>",
      "pruned_answer": "<answer generated from pruned context>",
      "reference_answer": "<gold/reference answer, optional but recommended>"
    }

If reference_answer is present, both raw_answer and pruned_answer are scored
against it (absolute quality). Additionally, pruned_answer is scored against
raw_answer directly (fidelity/consistency of pruning w.r.t. the unpruned
baseline), which is the core "did pruning break anything" metric.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore


def load_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/fidelity.csv")
    ap.add_argument("--lang", default="ar", help="BERTScore language code")
    args = ap.parse_args()

    records = load_records(args.data)
    if not records:
        print("No records found.")
        return

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    bleu = BLEU()

    rows = []
    for r in records:
        query = r["query"]
        raw_ans = r["raw_answer"]
        pruned_ans = r["pruned_answer"]
        ref = r.get("reference_answer")

        # Fidelity: pruned vs raw (core metric — did pruning change the answer?)
        rougeL_vs_raw = rouge.score(raw_ans, pruned_ans)["rougeL"].fmeasure
        bleu_vs_raw = bleu.sentence_score(pruned_ans, [raw_ans]).score

        row = {
            "query": query,
            "rougeL_pruned_vs_raw": rougeL_vs_raw,
            "bleu_pruned_vs_raw": bleu_vs_raw,
        }

        if ref:
            rougeL_raw_vs_ref = rouge.score(ref, raw_ans)["rougeL"].fmeasure
            rougeL_pruned_vs_ref = rouge.score(ref, pruned_ans)["rougeL"].fmeasure
            row["rougeL_raw_vs_ref"] = rougeL_raw_vs_ref
            row["rougeL_pruned_vs_ref"] = rougeL_pruned_vs_ref

        rows.append(row)

    # BERTScore in batch (much faster than per-example)
    pruned_list = [r["pruned_answer"] for r in records]
    raw_list = [r["raw_answer"] for r in records]
    _, _, f1 = bertscore(pruned_list, raw_list, lang=args.lang, verbose=False)
    for row, f1_val in zip(rows, f1.tolist()):
        row["bertscore_f1_pruned_vs_raw"] = f1_val

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {len(rows)} items. Results: {args.out}")
    print(f"Mean ROUGE-L (pruned vs raw): {statistics.mean(r['rougeL_pruned_vs_raw'] for r in rows):.4f}")
    print(f"Mean BERTScore-F1 (pruned vs raw): {statistics.mean(r['bertscore_f1_pruned_vs_raw'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
