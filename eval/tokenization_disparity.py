"""
Tokenization Disparity Analysis
--------------------------------
Empirically measures how many more tokens Arabic text yields vs. an
equivalent-meaning English text, for a given tokenizer. Produces a CSV
and summary stats to justify Arabic-first context pruning in the paper
(Part 3, "Tokenization Disparity Analysis").

Usage:
    python eval/tokenization_disparity.py --pairs data/parallel_pairs.jsonl \
        --tokenizer Qwen/Qwen2.5-7B-Instruct --out results/tokenization_disparity.csv

Input format (JSONL), one aligned pair per line:
    {"ar": "...", "en": "..."}
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def load_pairs(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((obj["ar"], obj["en"]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSONL file of {ar, en} parallel sentence/document pairs")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/tokenization_disparity.csv")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs = load_pairs(args.pairs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ratios = []

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ar_tokens", "en_tokens", "ratio_ar_over_en", "ar_chars", "en_chars"])
        for ar, en in pairs:
            ar_n = len(tok.encode(ar))
            en_n = len(tok.encode(en))
            ratio = ar_n / en_n if en_n else float("nan")
            ratios.append(ratio)
            writer.writerow([ar_n, en_n, f"{ratio:.4f}", len(ar), len(en)])

    if ratios:
        print(f"Pairs analyzed: {len(ratios)}")
        print(f"Mean AR/EN token ratio: {statistics.mean(ratios):.3f}")
        print(f"Median AR/EN token ratio: {statistics.median(ratios):.3f}")
        print(f"Stdev: {statistics.pstdev(ratios):.3f}")
        print(f"Results written to: {args.out}")
    else:
        print("No pairs found.")


if __name__ == "__main__":
    main()
