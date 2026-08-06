"""
Analytical (not measured) KV-cache memory savings estimate.

Addresses R4/DA round-1 finding: no systems-level number connects the real,
measured token/char reduction (Section 5) to an actual KV-cache byte figure,
because no GPU was available to run vLLM directly. This script computes a
back-of-envelope estimate from three real, verifiable inputs -- it does NOT
substitute for the GPU throughput benchmark specified in Section 4.3/8, and
the paper must present it as an analytical projection, clearly separated
from the empirical results.

Inputs (all real, all cited):
  1. Llama-3.1-8B-Instruct architecture config, fetched directly from the
     model's primary-source config.json on Hugging Face (num_hidden_layers=32,
     num_key_value_heads=8, hidden_size=4096, num_attention_heads=32 ->
     head_dim=128). Standard GQA KV-cache formula:
         bytes/token = 2 (K,V) x num_layers x num_kv_heads x head_dim x dtype_bytes
  2. The real, measured mean raw retrieved-context size (chars) from the
     expanded fidelity pilot (data/eval_set_expanded.jsonl).
  3. The real, measured Arabic chars-per-token ratio from the tokenization-
     disparity corpus (results/tokenization_disparity_llama.csv), used only
     to convert the pilot's char-based context size into an approximate
     token count (no separate tokenizer call needed for this back-of-envelope
     step).
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. KV-cache bytes/token (verified via Hugging Face config.json, see docstring)
NUM_LAYERS = 32
NUM_KV_HEADS = 8
HIDDEN_SIZE = 4096
NUM_ATTN_HEADS = 32
HEAD_DIM = HIDDEN_SIZE / NUM_ATTN_HEADS  # 128
DTYPE_BYTES = 2  # bf16/fp16, vLLM default serving dtype
KV_BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES  # 131072 = 128 KiB

# 2. mean raw retrieved-context size (chars), real, from the expanded pilot
rows = [json.loads(l) for l in open(ROOT / "data/eval_set_expanded.jsonl", encoding="utf-8")]
uniq_by_query = {r["query"]: r["raw_context_chars"] for r in rows}
mean_raw_chars = statistics.mean(uniq_by_query.values())

# 3. Arabic chars-per-token, real, aggregate over the 30-pair tokenization corpus
with open(ROOT / "results/tokenization_disparity_llama.csv") as f:
    tok_rows = list(csv.DictReader(f))
total_ar_chars = sum(float(r["ar_chars"]) for r in tok_rows)
total_ar_tokens = sum(float(r["ar_tokens"]) for r in tok_rows)
chars_per_token = total_ar_chars / total_ar_tokens

est_raw_tokens = mean_raw_chars / chars_per_token

# Real, measured char-reduction per ratio (from results/fidelity_summary.csv)
char_reduction_by_ratio = {0.3: 0.6729, 0.5: 0.5029, 0.7: 0.3422}

results = {
    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
    "kv_kib_per_token": KV_BYTES_PER_TOKEN / 1024,
    "mean_raw_context_chars": round(mean_raw_chars, 1),
    "arabic_chars_per_token": round(chars_per_token, 3),
    "estimated_raw_context_tokens": round(est_raw_tokens, 1),
    "per_ratio": [],
}

for ratio, reduction in sorted(char_reduction_by_ratio.items()):
    tokens_saved = est_raw_tokens * reduction
    kv_saved_kib = tokens_saved * results["kv_kib_per_token"]
    results["per_ratio"].append({
        "ratio": ratio,
        "measured_char_reduction": reduction,
        "estimated_tokens_saved_per_request": round(tokens_saved, 1),
        "estimated_kv_cache_saved_per_request_kib": round(kv_saved_kib, 1),
        "estimated_kv_cache_saved_per_request_mib": round(kv_saved_kib / 1024, 2),
    })

for conc in [10, 50, 100]:
    r05 = char_reduction_by_ratio[0.5]
    tokens_saved = est_raw_tokens * r05
    kv_saved_mib = tokens_saved * results["kv_kib_per_token"] / 1024
    results.setdefault("concurrency_projection_at_r0.5", []).append({
        "concurrency": conc,
        "aggregate_kv_cache_saved_mib": round(kv_saved_mib * conc, 1),
        "aggregate_kv_cache_saved_gib": round(kv_saved_mib * conc / 1024, 2),
    })

out_path = ROOT / "results/kv_cache_analytical_estimate.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nWrote {out_path}")
