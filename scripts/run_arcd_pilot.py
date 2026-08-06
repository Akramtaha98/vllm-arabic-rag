"""
Runs the expanded, ARCD-based baseline re-test: for each of the 40 sampled
ARCD questions (data/arcd_eval_set.json), against the real 6-document
noisy retrieval pool (1 gold + 5 distractor paragraphs from other
articles), generates:
  - 1 raw (unpruned, all 6 docs) answer
  - 3 LSPM (semantic cross-encoder) pruned answers, r in {0.3, 0.5, 0.7}
  - 3 naive length-matched truncation answers, r in {0.3, 0.5, 0.7}
= 7 live calls/question x 40 questions = 280 total live calls against the
NVIDIA NIM-hosted Llama-3.1-8B-Instruct endpoint.

Resumable: writes to data/arcd_results.jsonl incrementally and skips
(question_id, method, ratio) cells already present, so it can be safely
re-invoked across multiple shorter runs if the process is interrupted.

Retrieval uses the SAME naive keyword-overlap retriever as the original
pilot (not a full vector DB) so the pruning/generation comparison is
apples-to-apples with the paper's existing methodology; the only change is
a larger, noisier, real-answer-bearing corpus in place of the original
10-document synthetic knowledge base.
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
EVAL_SET = json.load(open(ROOT / "data/arcd_eval_set.json", encoding="utf-8"))

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
API_KEY = os.environ["VLLM_API_KEY"]

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)
pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

RATIOS = [0.3, 0.5, 0.7]
MAX_CALLS_PER_INVOCATION = int(os.environ.get("MAX_CALLS_PER_INVOCATION", "9999"))
# Free-tier NIM allowance rate-limits hard after a burst of rapid calls
# (observed: HTTP 429 starting around call ~150-200 in a tight loop). A
# fixed inter-call delay plus retry-with-backoff on 429 keeps us under the
# limit instead of silently persisting error text as if it were an answer.
INTER_CALL_DELAY_S = float(os.environ.get("INTER_CALL_DELAY_S", "1.5"))
MAX_RETRIES = 6


def chat_with_retry(system_prompt, user_prompt, **kwargs):
    """Wraps client.chat with exponential backoff on HTTP 429 (and other
    error responses, which VLLMClient.chat() surfaces as a normal
    VLLMResponse whose .text starts with '[vLLM error <code>]' rather than
    raising). Never returns an error-text response silently -- raises
    RuntimeError if all retries are exhausted, so a failed cell is skipped
    and reported instead of written into the results file as a fake answer.
    """
    delay = 5.0
    last_resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        resp = client.chat(system_prompt, user_prompt, **kwargs)
        last_resp = resp
        if not resp.text.startswith("[vLLM error"):
            time.sleep(INTER_CALL_DELAY_S)
            return resp
        print(f"    (attempt {attempt}/{MAX_RETRIES}) {resp.text[:80]} -- backing off {delay:.0f}s", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"Exhausted {MAX_RETRIES} retries; last response: {last_resp.text[:200]}")


def naive_retrieve_order(query, documents, top_k=6):
    """Keyword-overlap ranking over the already-assembled 6-doc pool (same
    mechanism as the original pilot's retriever), returns documents
    re-ordered by relevance score (does not change which 6 are in the
    pool -- that's fixed by build_arcd_eval_set.py -- only their order,
    consistent with 'the retriever ranks a fixed candidate set')."""
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def naive_truncate(documents, compression_ratio):
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


out_path = ROOT / "data" / "arcd_results.jsonl"
done_cells = set()
if out_path.exists():
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done_cells.add((rec["id"], rec["method"], rec["ratio"]))
f_out = open(out_path, "a", encoding="utf-8")
print(f"Resuming: {len(done_cells)} cells already done.", flush=True)

n_new_calls = 0
t_start = time.time()

for qi, item in enumerate(EVAL_SET):
    if n_new_calls >= MAX_CALLS_PER_INVOCATION:
        print(f"\nHit MAX_CALLS_PER_INVOCATION={MAX_CALLS_PER_INVOCATION}, stopping this invocation early "
              f"(re-run the same command to continue).", flush=True)
        break

    qid = item["id"]
    question = item["question"]
    docs = naive_retrieve_order(question, item["documents"], top_k=6)
    raw_context = " ".join(docs)

    print(f"[{qi+1}/{len(EVAL_SET)}] {question}", flush=True)

    # ---------- raw (unpruned) ----------
    if (qid, "raw", 1.0) in done_cells:
        raw_answer = None
        for line in open(out_path, encoding="utf-8"):
            rec = json.loads(line)
            if rec["id"] == qid and rec["method"] == "raw":
                raw_answer = rec["answer"]
                break
        print("  [raw] already done, skipping", flush=True)
    else:
        prompt = f"السياق: {raw_context}\n\nالسؤال: {question}"
        try:
            resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
        except RuntimeError as e:
            print(f"  [raw] SKIPPED after retries: {e}", flush=True)
            resp = None
        if resp is not None:
            raw_answer = resp.text
            n_new_calls += 1
            print(f"  [raw] {raw_answer[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": "raw", "ratio": 1.0,
                "answer": raw_answer, "gold_answers": item["gold_answers"],
                "context_chars": len(raw_context), "context_sentences": len(split_sentences(raw_context)),
                "pruner_latency_ms": None,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

    if n_new_calls >= MAX_CALLS_PER_INVOCATION:
        continue

    for ratio in RATIOS:
        # ---------- LSPM ----------
        if (qid, "lspm", ratio) not in done_cells and n_new_calls < MAX_CALLS_PER_INVOCATION:
            pr = pruner.prune(question, docs, compression_ratio=ratio)
            prompt = f"السياق: {pr.pruned_text}\n\nالسؤال: {question}"
            try:
                resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
            except RuntimeError as e:
                print(f"  [lspm  r={ratio}] SKIPPED after retries: {e}", flush=True)
                resp = None
            if resp is not None:
                n_new_calls += 1
                print(f"  [lspm  r={ratio}] {resp.text[:60]}", flush=True)
                f_out.write(json.dumps({
                    "id": qid, "question": question, "method": "lspm", "ratio": ratio,
                    "answer": resp.text, "gold_answers": item["gold_answers"],
                    "context_chars": pr.pruned_char_count, "context_sentences": pr.kept_sentence_count,
                    "pruner_latency_ms": pr.latency_ms,
                }, ensure_ascii=False) + "\n")
                f_out.flush()
        elif (qid, "lspm", ratio) in done_cells:
            print(f"  [lspm  r={ratio}] already done, skipping", flush=True)

        # ---------- naive truncation ----------
        if (qid, "naive", ratio) not in done_cells and n_new_calls < MAX_CALLS_PER_INVOCATION:
            nb = naive_truncate(docs, ratio)
            prompt = f"السياق: {nb['pruned_text']}\n\nالسؤال: {question}"
            try:
                resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
            except RuntimeError as e:
                print(f"  [naive r={ratio}] SKIPPED after retries: {e}", flush=True)
                resp = None
            if resp is not None:
                n_new_calls += 1
                print(f"  [naive r={ratio}] {resp.text[:60]}", flush=True)
                f_out.write(json.dumps({
                    "id": qid, "question": question, "method": "naive", "ratio": ratio,
                    "answer": resp.text, "gold_answers": item["gold_answers"],
                    "context_chars": len(nb["pruned_text"]), "context_sentences": nb["kept_sentence_count"],
                    "pruner_latency_ms": None,
                }, ensure_ascii=False) + "\n")
                f_out.flush()
        elif (qid, "naive", ratio) in done_cells:
            print(f"  [naive r={ratio}] already done, skipping", flush=True)

f_out.close()
total_expected = len(EVAL_SET) * 7
total_done = len(done_cells) + n_new_calls
print(f"\nThis invocation: {n_new_calls} new live calls in {time.time()-t_start:.1f}s.", flush=True)
print(f"Overall progress: {total_done}/{total_expected} cells complete.", flush=True)
