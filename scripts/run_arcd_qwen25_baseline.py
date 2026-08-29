"""
Second-generator generalization check: reruns the exact same ARCD re-test
(scripts/run_arcd_pilot.py's pipeline -- same 140-question eval set, same
6-document retrieval pool, same LSPM/naive pruning at r in {0.3, 0.5, 0.7},
same raw/unpruned condition, same decoding settings) against a SECOND,
architecturally different generator family: Qwen2.5-7B-Instruct, via the
same NVIDIA NIM hosted API used throughout this project.

Reviewer concern this answers (Reviewer 2, generalization): the paper's
one experimentally established result was demonstrated on exactly one
generator family (Llama-3.1-8B-Instruct). This does not "solve" the
generalization gap in full (still one retriever, one dataset), but it
directly tests whether the LSPM-over-naive-truncation head-to-head result
replicates on a second, unrelated model family -- the single highest-value
partial answer available without new infrastructure.

Run this AFTER scripts/run_arcd_pilot.py (needs the same eval set; does not
depend on its output otherwise -- this issues its own independent set of
live calls).

Output: data/arcd_qwen25_results.jsonl, same schema as data/arcd_results.jsonl
plus a "generator" field so scripts/analyze_arcd_results.py (or a small
extension of it) can filter by generator when scoring.

Usage:
    export VLLM_API_KEY=...        # your existing NIM key
    python scripts/run_arcd_qwen25_baseline.py

Resumable exactly like run_arcd_pilot.py: writes incrementally, skips
(question_id, method, ratio) cells already present in the output file.
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

# Model id confirmed against NVIDIA's own NIM API reference page
# (docs.api.nvidia.com/nim/reference/qwen-qwen2_5-7b-instruct) as of this
# writing. If this 404s when you run it, check https://build.nvidia.com
# for the exact current model string and update MODEL below -- do not
# guess a different id silently.
MODEL = "qwen/qwen2.5-7b-instruct"
GENERATOR_LABEL = "qwen2.5-7b-instruct"

API_KEY = os.environ["VLLM_API_KEY"]

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)
pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

RATIOS = [0.3, 0.5, 0.7]
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


def main():
    out_path = ROOT / "data" / "arcd_qwen25_results.jsonl"
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
            print(f"\nHit MAX_CALLS_PER_INVOCATION={MAX_CALLS_PER_INVOCATION}, stopping "
                  f"(re-run the same command to continue).", flush=True)
            break

        qid = item["id"]
        question = item["question"]
        docs = naive_retrieve_order(question, item["documents"], top_k=6)
        raw_context = " ".join(docs)

        print(f"[{qi+1}/{len(EVAL_SET)}] {question}", flush=True)

        # ---------- raw ----------
        if (qid, "raw", 1.0) not in done_cells and n_new_calls < MAX_CALLS_PER_INVOCATION:
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
                    "generator": GENERATOR_LABEL,
                    "answer": resp.text, "gold_answers": item["gold_answers"],
                    "context_chars": len(raw_context), "context_sentences": len(split_sentences(raw_context)),
                    "pruner_latency_ms": None,
                }, ensure_ascii=False) + "\n")
                f_out.flush()
        elif (qid, "raw", 1.0) in done_cells:
            print("  [raw] already done, skipping", flush=True)

        if n_new_calls >= MAX_CALLS_PER_INVOCATION:
            continue

        for ratio in RATIOS:
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
                        "generator": GENERATOR_LABEL,
                        "answer": resp.text, "gold_answers": item["gold_answers"],
                        "context_chars": pr.pruned_char_count, "context_sentences": pr.kept_sentence_count,
                        "pruner_latency_ms": pr.latency_ms,
                    }, ensure_ascii=False) + "\n")
                    f_out.flush()
            elif (qid, "lspm", ratio) in done_cells:
                print(f"  [lspm  r={ratio}] already done, skipping", flush=True)

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
                        "generator": GENERATOR_LABEL,
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
    print("\nNext: score data/arcd_qwen25_results.jsonl with the same Arabic-normalized "
          "EM/F1 scorer used for the Llama results (outputs/score_arcd.py from this session, "
          "or scripts/analyze_arcd_results.py extended to accept an input-file argument), and "
          "report the LSPM-vs-naive comparison side by side with the Llama-3.1 numbers already "
          "in the paper -- same direction = replicates; different direction or non-significant "
          "= a genuine, reportable generalization limitation, not something to hide.")


if __name__ == "__main__":
    main()
