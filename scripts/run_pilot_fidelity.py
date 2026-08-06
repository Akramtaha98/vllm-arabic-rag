"""
One-off pilot runner: generates raw-vs-pruned answers for a small real
question set against the live NVIDIA NIM backend, for the semantic-fidelity
preliminary experiment reported in the paper. Not part of the reusable
benchmark suite (that's eval/semantic_fidelity.py, which this feeds).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from middleware.pruning import SemanticPruner
from middleware.vllm_client import VLLMClient

ROOT = Path(__file__).resolve().parent.parent
CORPUS = json.load(open(ROOT / "data/pilot_corpus.json", encoding="utf-8"))
QUESTIONS = json.load(open(ROOT / "data/pilot_questions.json", encoding="utf-8"))

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
API_KEY = os.environ["VLLM_API_KEY"]

SYSTEM_PROMPT = "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك وبشكل مختصر."

client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)
pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


def naive_retrieve(query, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in CORPUS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


out_path = ROOT / "data/eval_set.jsonl"
f_out = open(out_path, "w", encoding="utf-8")

for i, q in enumerate(QUESTIONS):
    docs = naive_retrieve(q, top_k=6)
    raw_context = " ".join(docs)

    prune_result = pruner.prune(q, docs, compression_ratio=0.5)
    pruned_context = prune_result.pruned_text

    raw_prompt = f"السياق: {raw_context}\n\nالسؤال: {q}"
    pruned_prompt = f"السياق: {pruned_context}\n\nالسؤال: {q}"

    print(f"[{i+1}/{len(QUESTIONS)}] {q}", flush=True)
    t0 = time.time()
    raw_resp = client.chat(SYSTEM_PROMPT, raw_prompt, temperature=0.0, max_tokens=200)
    print(f"  raw answer ({time.time()-t0:.1f}s): {raw_resp.text}", flush=True)

    t0 = time.time()
    pruned_resp = client.chat(SYSTEM_PROMPT, pruned_prompt, temperature=0.0, max_tokens=200)
    print(f"  pruned answer ({time.time()-t0:.1f}s): {pruned_resp.text}", flush=True)

    record = {
        "query": q,
        "raw_answer": raw_resp.text,
        "pruned_answer": pruned_resp.text,
        "raw_context_chars": len(raw_context),
        "pruned_context_chars": len(pruned_context),
        "raw_context_sentences": prune_result.original_sentence_count,
        "pruned_context_sentences": prune_result.kept_sentence_count,
    }
    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    f_out.flush()

f_out.close()
print(f"\nDone. Wrote records to {out_path}", flush=True)
