"""
Round-2 expanded pilot runner, built to answer the peer-review round-1
findings (review_round1.md):
  - R1/DA: single ratio, no dose-response curve  -> now sweeps r in {0.3, 0.5, 0.7}
  - R1/R4/DA: no baseline comparison             -> adds a naive length-matched
    truncation baseline (same ratio, no cross-encoder scoring) run through the
    identical live-LLM pipeline
  - R4: LSPM's own overhead never measured       -> records pruner latency_ms
    per item (already tracked by SemanticPruner.prune, just wasn't surfaced)

Reuses the round-1 raw answers (ratio-independent) and the r=0.5 LSPM answers
already collected in data/eval_set.jsonl, so only the NEW cells of the
ratio x method matrix are actually called against the live endpoint:
  LSPM   @ r=0.3, r=0.7            (r=0.5 reused)      -> 8 x 2 = 16 new calls
  naive  @ r=0.3, r=0.5, r=0.7                          -> 8 x 3 = 24 new calls
  raw                                                    -> reused, 0 new calls
Total new live calls: 40 (vs. 98 if run from scratch).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from middleware.pruning import SemanticPruner, split_sentences
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

RATIOS = [0.3, 0.5, 0.7]


def naive_truncate(query, documents, compression_ratio):
    """Length-matched baseline: keep the first r*N sentences in original
    document order, WITHOUT any relevance scoring. Same interface shape as
    SemanticPruner.prune's relevant fields so downstream code is uniform."""
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    n = len(all_sentences)
    num_to_keep = max(1, int(round(n * compression_ratio)))
    num_to_keep = min(num_to_keep, n)
    kept = all_sentences[:num_to_keep]
    return {
        "pruned_text": " ".join(kept),
        "original_sentence_count": n,
        "kept_sentence_count": len(kept),
    }


def naive_retrieve(query, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in CORPUS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# Load round-1 results to reuse raw answers + r=0.5 LSPM answers.
round1_path = ROOT / "data/eval_set.jsonl"
round1_by_query = {}
if round1_path.exists():
    with open(round1_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                round1_by_query[rec["query"]] = rec

out_path = ROOT / "data/eval_set_expanded.jsonl"

# Resumable: this environment kills long-running foreground calls after ~3
# minutes, so re-invocations must skip (query, method, ratio) cells already
# written rather than starting over.
done_cells = set()
if out_path.exists():
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done_cells.add((rec["query"], rec["method"], rec["ratio"]))
f_out = open(out_path, "a", encoding="utf-8")
print(f"Resuming: {len(done_cells)} cells already done.", flush=True)

n_new_calls = 0
t_start = time.time()

for qi, q in enumerate(QUESTIONS):
    docs = naive_retrieve(q, top_k=6)
    raw_context = " ".join(docs)

    # --- raw answer: reuse round-1 if present, else fetch fresh ---
    r1 = round1_by_query.get(q)
    if r1 is not None:
        raw_answer = r1["raw_answer"]
        print(f"[{qi+1}/{len(QUESTIONS)}] {q}  (raw answer reused from round 1)", flush=True)
    else:
        print(f"[{qi+1}/{len(QUESTIONS)}] {q}  (fetching raw answer)", flush=True)
        raw_prompt = f"السياق: {raw_context}\n\nالسؤال: {q}"
        raw_resp = client.chat(SYSTEM_PROMPT, raw_prompt, temperature=0.0, max_tokens=200)
        raw_answer = raw_resp.text
        n_new_calls += 1

    for ratio in RATIOS:
        # ---------- LSPM (semantic cross-encoder pruning) ----------
        if (q, "lspm", ratio) in done_cells:
            print(f"  [LSPM  r={ratio}] already done, skipping", flush=True)
        elif ratio == 0.5 and r1 is not None:
            pruned_answer = r1["pruned_answer"]
            pruned_chars = r1["pruned_context_chars"]
            orig_sent = r1["raw_context_sentences"]
            kept_sent = r1["pruned_context_sentences"]
            latency_ms = None  # not recorded in round 1
            print(f"  [LSPM  r={ratio}] reused from round 1", flush=True)
            f_out.write(json.dumps({
                "query": q, "method": "lspm", "ratio": ratio,
                "raw_answer": raw_answer, "pruned_answer": pruned_answer,
                "raw_context_chars": len(raw_context), "pruned_context_chars": pruned_chars,
                "raw_context_sentences": orig_sent, "pruned_context_sentences": kept_sent,
                "pruner_latency_ms": latency_ms,
            }, ensure_ascii=False) + "\n")
            f_out.flush()
        else:
            prune_result = pruner.prune(q, docs, compression_ratio=ratio)
            pruned_prompt = f"السياق: {prune_result.pruned_text}\n\nالسؤال: {q}"
            t0 = time.time()
            resp = client.chat(SYSTEM_PROMPT, pruned_prompt, temperature=0.0, max_tokens=200)
            n_new_calls += 1
            print(f"  [LSPM  r={ratio}] ({time.time()-t0:.1f}s) {resp.text[:60]}...", flush=True)
            f_out.write(json.dumps({
                "query": q, "method": "lspm", "ratio": ratio,
                "raw_answer": raw_answer, "pruned_answer": resp.text,
                "raw_context_chars": len(raw_context), "pruned_context_chars": prune_result.pruned_char_count,
                "raw_context_sentences": prune_result.original_sentence_count,
                "pruned_context_sentences": prune_result.kept_sentence_count,
                "pruner_latency_ms": prune_result.latency_ms,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

        # ---------- naive truncation baseline ----------
        if (q, "naive", ratio) in done_cells:
            print(f"  [naive r={ratio}] already done, skipping", flush=True)
        else:
            nb = naive_truncate(q, docs, ratio)
            naive_prompt = f"السياق: {nb['pruned_text']}\n\nالسؤال: {q}"
            t0 = time.time()
            nresp = client.chat(SYSTEM_PROMPT, naive_prompt, temperature=0.0, max_tokens=200)
            n_new_calls += 1
            print(f"  [naive r={ratio}] ({time.time()-t0:.1f}s) {nresp.text[:60]}...", flush=True)

            f_out.write(json.dumps({
                "query": q, "method": "naive", "ratio": ratio,
                "raw_answer": raw_answer, "pruned_answer": nresp.text,
                "raw_context_chars": len(raw_context), "pruned_context_chars": len(nb["pruned_text"]),
                "raw_context_sentences": nb["original_sentence_count"], "pruned_context_sentences": nb["kept_sentence_count"],
                "pruner_latency_ms": None,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

f_out.close()
print(f"\nDone. {n_new_calls} new live calls in {time.time()-t_start:.1f}s. Wrote {out_path}", flush=True)
