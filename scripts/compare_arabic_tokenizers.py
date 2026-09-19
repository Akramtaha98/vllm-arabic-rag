"""
Real, CPU-only token-count comparison of Arabic-specific tokenizers vs. the
generator's Llama-3.1 tokenizer, on the same 30-pair hand-verified parallel
corpus (data/parallel_pairs.jsonl) already used for the paper's aggregate
tokenization-disparity measurement (Section 3.3/4.1). Produces real numbers
for Reviewer 3's tokenizer-comparison request, in place of pure discussion.
"""
import json
import statistics
from transformers import AutoTokenizer

pairs = [json.loads(l) for l in open("data/parallel_pairs.jsonl", encoding="utf-8") if l.strip()]
print(f"Loaded {len(pairs)} pairs")

tokenizers = {
    "Llama-3.1 (generator, used throughout this paper)": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "AraBERTv2 (Arabic-specific)": "aubmindlab/bert-base-arabertv2",
    "CAMeLBERT-mix (Arabic-specific)": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "AraBERT-asafaya (Arabic-specific)": "asafaya/bert-base-arabic",
}

results = {}
for label, name in tokenizers.items():
    tok = AutoTokenizer.from_pretrained(name)
    ar_counts = []
    en_counts = []
    for p in pairs:
        ar_counts.append(len(tok.encode(p["ar"], add_special_tokens=False)))
        en_counts.append(len(tok.encode(p["en"], add_special_tokens=False)))
    ratios = [a / e for a, e in zip(ar_counts, en_counts)]
    results[label] = {
        "mean_ar_tokens": statistics.mean(ar_counts),
        "mean_en_tokens": statistics.mean(en_counts),
        "mean_ar_en_ratio": statistics.mean(ratios),
        "median_ar_en_ratio": statistics.median(ratios),
    }
    print(f"{label:50s}  AR={statistics.mean(ar_counts):6.2f}  EN={statistics.mean(en_counts):6.2f}  "
          f"ratio(mean)={statistics.mean(ratios):.3f}  ratio(median)={statistics.median(ratios):.3f}")

with open("/tmp/tokenizer_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
