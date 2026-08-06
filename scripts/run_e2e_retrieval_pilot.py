"""
Runs the end-to-end retrieval evaluation pilot: for each of the 60 sampled
questions (data/e2e_retrieval_eval_set.json), whose 6-document context is
the output of a real TF-IDF retriever over a 234-paragraph corpus (recall@6
is NOT guaranteed -- see build_e2e_retrieval_eval.py), generates:
  - 1 raw (unpruned, all 6 retrieved docs) answer
  - 1 LSPM answer at r = 0.3 (the ratio with the paper's established,
    Holm-Bonferroni-significant head-to-head advantage over naive
    truncation on the answer-guaranteed ARCD re-test, Section 5.4)
  - 1 naive length-matched truncation answer at r = 0.3
= 3 live calls/question x 60 questions = 180 total live calls against the
NVIDIA NIM-hosted Llama-3.1-8B-Instruct endpoint.

Resumable: writes to data/e2e_retrieval_results.jsonl incrementally and
skips (question_id, method) cells already present.

Retrieval order is NOT re-ranked here (unlike run_arcd_pilot.py, which
re-ranks its fixed, answer-guaranteed pool with a naive keyword-overlap
scorer): the 6 documents are already in the real TF-IDF retriever's ranked
order from build_e2e_retrieval_eval.py, so "naive truncation" here means
keeping the first r*N sentences of that real retrieval order, exactly
mirroring what a production pipeline would see.
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
EVAL_SET = json.load(open(ROOT / "data/e2e_retrieval_eval_set.json", encoding="utf-8"))

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
API_KEY = os.environ["VLLM_API_KEY"]

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)
pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

RATIO = 0.3
MAX_CALLS_PER_INVOCATION = int(os.environ.get("MAX_CALLS_PER_INVOCATION", "9999"))
INTER_CALL_DELAY_S = float(os.environ.get("INTER_CALL_DELAY_S", "1.5"))
MAX_RETRIES = 6


def chat_with_retry(system_prompt, user_prompt, **kwargs):
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


out_path = ROOT / "data" / "e2e_retrieval_results.jsonl"
done_cells = set()
if out_path.exists():
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done_cells.add((rec["id"], rec["method"]))
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
    docs = item["retrieved_documents"]
    raw_context = " ".join(docs)

    print(f"[{qi+1}/{len(EVAL_SET)}] {question} (gold_in_topk={item['gold_in_topk']})", flush=True)

    methods_to_run = []
    if (qid, "raw") not in done_cells:
        methods_to_run.append("raw")
    if (qid, "lspm") not in done_cells:
        methods_to_run.append("lspm")
    if (qid, "naive") not in done_cells:
        methods_to_run.append("naive")

    for method in methods_to_run:
        if n_new_calls >= MAX_CALLS_PER_INVOCATION:
            break

        if method == "raw":
            prompt = f"السياق: {raw_context}\n\nالسؤال: {question}"
            context_chars = len(raw_context)
            context_sentences = len(split_sentences(raw_context))
            pruner_latency_ms = None
        elif method == "lspm":
            pr = pruner.prune(question, docs, compression_ratio=RATIO)
            prompt = f"السياق: {pr.pruned_text}\n\nالسؤال: {question}"
            context_chars = pr.pruned_char_count
            context_sentences = pr.kept_sentence_count
            pruner_latency_ms = pr.latency_ms
        else:  # naive
            nb = naive_truncate(docs, RATIO)
            prompt = f"السياق: {nb['pruned_text']}\n\nالسؤال: {question}"
            context_chars = len(nb["pruned_text"])
            context_sentences = nb["kept_sentence_count"]
            pruner_latency_ms = None

        try:
            resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
        except RuntimeError as e:
            print(f"  [{method}] SKIPPED after retries: {e}", flush=True)
            resp = None

        if resp is not None:
            n_new_calls += 1
            print(f"  [{method}] {resp.text[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": method, "ratio": RATIO,
                "answer": resp.text, "gold_answers": item["gold_answers"],
                "gold_in_topk": item["gold_in_topk"], "gold_rank": item["gold_rank"],
                "context_chars": context_chars, "context_sentences": context_sentences,
                "pruner_latency_ms": pruner_latency_ms,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

f_out.close()
total_expected = len(EVAL_SET) * 3
total_done = len(done_cells) + n_new_calls
print(f"\nThis invocation: {n_new_calls} new live calls in {time.time()-t_start:.1f}s.", flush=True)
print(f"Overall progress: {total_done}/{total_expected} cells complete.", flush=True)
