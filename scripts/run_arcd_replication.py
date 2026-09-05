"""
Fresh-seed disjoint replication run (paper Section 4.10).

Identical protocol to scripts/run_arcd_pilot.py -- same system prompt, same
naive keyword-overlap retriever, same LSPM cross-encoder pruner, same
Llama-3.1-8B-Instruct generator, same temperature=0.0/max_tokens=64 -- but:
  - reads data/arcd_eval_set_replication.json (140 FRESH questions, none of
    which appear in the original data/arcd_results.jsonl), built by
    build_arcd_eval_set_replication.py;
  - only runs the 3 cells the replication actually needs: raw, lspm r=0.3,
    naive r=0.3 (420 live calls total instead of the original 980);
  - writes to data/arcd_replication_results.jsonl, in the same schema as
    data/arcd_results.jsonl, so scripts/score_arcd.py scores it unchanged.

The generator MUST stay Llama-3.1-8B-Instruct to match the original test --
substituting Claude or a ChatGPT model here would make this a different,
third-generator generalization experiment (like the existing Qwen2.5 check),
not an independent confirmation of the original result.

Two ways to point this at a real Llama-3.1-8B-Instruct endpoint:

  1) NVIDIA NIM hosted API (no GPU needed, cheapest, matches how the
     original 140-question test was actually run):
       export VLLM_API_KEY=nvapi-...
       python run_arcd_replication.py

  2) Self-hosted on a rented GPU (e.g. RTX 4090), matching the paper's own
     Section 4.6/4.7 GPU-benchmark setup:
       vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct \\
           --max-model-len 8192 --dtype bfloat16
       export VLLM_API_URL=http://localhost:8000/v1/chat/completions
       export VLLM_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B-Instruct
       python run_arcd_replication.py
     (no VLLM_API_KEY needed for a local server; leave it unset)

Resumable: writes incrementally and skips (id, method, ratio) cells already
present, so it is safe to Ctrl-C and re-run the same command later.
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
EVAL_SET_PATH = ROOT / "data" / "arcd_eval_set_replication.json"
if not EVAL_SET_PATH.exists():
    raise SystemExit(
        f"{EVAL_SET_PATH} not found. Run build_arcd_eval_set_replication.py first."
    )
EVAL_SET = json.load(open(EVAL_SET_PATH, encoding="utf-8"))

# Default: NVIDIA NIM hosted API, same endpoint the original pilot used.
# Override VLLM_API_URL/VLLM_MODEL_NAME to point at a self-hosted vLLM
# server instead (see module docstring, option 2).
API_URL = os.environ.get("VLLM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
API_KEY = os.environ.get("VLLM_API_KEY")  # required for NIM, leave unset for local vLLM

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)
pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

RATIO = 0.3  # this replication only needs the r=0.3 headline result
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


def naive_retrieve_order(query, documents, top_k=6):
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


out_path = ROOT / "data" / "arcd_replication_results.jsonl"
done_cells = set()
if out_path.exists():
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done_cells.add((rec["id"], rec["method"], rec["ratio"]))
f_out = open(out_path, "a", encoding="utf-8")
print(f"Model: {MODEL}  Endpoint: {API_URL}", flush=True)
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
        print("  [raw] already done, skipping", flush=True)
    else:
        prompt = f"السياق: {raw_context}\n\nالسؤال: {question}"
        try:
            resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
        except RuntimeError as e:
            print(f"  [raw] SKIPPED after retries: {e}", flush=True)
            resp = None
        if resp is not None:
            n_new_calls += 1
            print(f"  [raw] {resp.text[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": "raw", "ratio": 1.0,
                "answer": resp.text, "gold_answers": item["gold_answers"],
                "context_chars": len(raw_context), "context_sentences": len(split_sentences(raw_context)),
                "pruner_latency_ms": None,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

    if n_new_calls >= MAX_CALLS_PER_INVOCATION:
        continue

    # ---------- LSPM r=0.3 ----------
    if (qid, "lspm", RATIO) not in done_cells and n_new_calls < MAX_CALLS_PER_INVOCATION:
        pr = pruner.prune(question, docs, compression_ratio=RATIO)
        prompt = f"السياق: {pr.pruned_text}\n\nالسؤال: {question}"
        try:
            resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
        except RuntimeError as e:
            print(f"  [lspm  r={RATIO}] SKIPPED after retries: {e}", flush=True)
            resp = None
        if resp is not None:
            n_new_calls += 1
            print(f"  [lspm  r={RATIO}] {resp.text[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": "lspm", "ratio": RATIO,
                "answer": resp.text, "gold_answers": item["gold_answers"],
                "context_chars": pr.pruned_char_count, "context_sentences": pr.kept_sentence_count,
                "pruner_latency_ms": pr.latency_ms,
            }, ensure_ascii=False) + "\n")
            f_out.flush()
    elif (qid, "lspm", RATIO) in done_cells:
        print(f"  [lspm  r={RATIO}] already done, skipping", flush=True)

    # ---------- naive truncation r=0.3 ----------
    if (qid, "naive", RATIO) not in done_cells and n_new_calls < MAX_CALLS_PER_INVOCATION:
        nb = naive_truncate(docs, RATIO)
        prompt = f"السياق: {nb['pruned_text']}\n\nالسؤال: {question}"
        try:
            resp = chat_with_retry(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
        except RuntimeError as e:
            print(f"  [naive r={RATIO}] SKIPPED after retries: {e}", flush=True)
            resp = None
        if resp is not None:
            n_new_calls += 1
            print(f"  [naive r={RATIO}] {resp.text[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": "naive", "ratio": RATIO,
                "answer": resp.text, "gold_answers": item["gold_answers"],
                "context_chars": len(nb["pruned_text"]), "context_sentences": nb["kept_sentence_count"],
                "pruner_latency_ms": None,
            }, ensure_ascii=False) + "\n")
            f_out.flush()
    elif (qid, "naive", RATIO) in done_cells:
        print(f"  [naive r={RATIO}] already done, skipping", flush=True)

f_out.close()
total_expected = len(EVAL_SET) * 3
total_done = len(done_cells) + n_new_calls
print(f"\nThis invocation: {n_new_calls} new live calls in {time.time()-t_start:.1f}s.", flush=True)
print(f"Overall progress: {total_done}/{total_expected} cells complete.", flush=True)
if total_done >= total_expected:
    print("\nAll cells complete. Score it with:")
    print("  python3 scripts/score_arcd_replication.py")

