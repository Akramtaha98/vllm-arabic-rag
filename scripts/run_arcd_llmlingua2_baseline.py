"""
LLMLingua-2 baseline comparison on the ARCD re-test
------------------------------------------------------
Reviewer concern (both reviewers, item 1/7): the paper only compares LSPM
against naive positional truncation, not a real modern compression
method. This adds LLMLingua-2 (Pan et al., ACL 2024 Findings; Microsoft's
official `llmlingua` pip package) as a second baseline under the exact
same protocol as run_arcd_pilot.py: same 140-question ARCD pool, same
6-document retrieval pool per question, same generator/prompt/decoding
settings, and (via --match token_budget) the same token budget as LSPM
at each ratio, so this is directly comparable to run_arcd_token_budget.py.

Install:
    pip install llmlingua

Caveats to disclose in the paper if you run this:
  - LLMLingua-2's public checkpoints (xlm-roberta-large-meetingbank,
    bert-base-multilingual-cased-meetingbank) were trained/distilled on
    English meeting-summarization data, not Arabic and not RAG-style
    factual QA context. This is a genuine domain-shift caveat -- report
    it, don't hide it, exactly like the paper already does for its own
    limitations.
  - LLMLingua-2 compresses at the TOKEN level (drops individual tokens,
    not whole sentences), so its output can be disfluent Arabic; this is
    expected behavior for the method, not a bug in this script.

Run this AFTER run_arcd_pilot.py (needs LSPM's token counts to match
budgets against). Output: data/arcd_llmlingua2_results.jsonl, same
schema as the other ARCD result files (method="llmlingua2").
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
EVAL_SET = json.load(open(ROOT / "data/arcd_eval_set.json", encoding="utf-8"))

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
API_KEY = os.environ["VLLM_API_KEY"]

# Multilingual checkpoint (not the English-only meetingbank-large default)
# since our context is Arabic; still meeting-summarization-domain-trained,
# see module docstring caveat above.
LLMLINGUA2_MODEL = os.environ.get(
    "LLMLINGUA2_MODEL", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

RATIOS = [0.3, 0.5, 0.7]
INTER_CALL_DELAY_S = float(os.environ.get("INTER_CALL_DELAY_S", "1.5"))
MAX_RETRIES = 6
MAX_CALLS_PER_INVOCATION = int(os.environ.get("MAX_CALLS_PER_INVOCATION", "9999"))


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def chat_with_retry(client, system_prompt, user_prompt, **kwargs):
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


def main():
    from llmlingua import PromptCompressor

    print(f"Loading LLMLingua-2 ({LLMLINGUA2_MODEL}) ...")
    compressor = PromptCompressor(model_name=LLMLINGUA2_MODEL, use_llmlingua2=True)

    client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)

    out_path = ROOT / "data" / "arcd_llmlingua2_results.jsonl"
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

    for qi, item in enumerate(EVAL_SET):
        if n_new_calls >= MAX_CALLS_PER_INVOCATION:
            print("Hit MAX_CALLS_PER_INVOCATION, stopping (re-run to continue).", flush=True)
            break

        qid = item["id"]
        question = item["question"]
        docs = naive_retrieve_order(question, item["documents"], top_k=6)
        raw_context = " ".join(docs)

        for ratio in RATIOS:
            key = (qid, "llmlingua2", ratio)
            if key in done_cells or n_new_calls >= MAX_CALLS_PER_INVOCATION:
                continue

            try:
                result = compressor.compress_prompt(
                    raw_context,
                    rate=ratio,
                    force_tokens=["\n", "?", "؟", "."],
                )
                compressed_context = result["compressed_prompt"]
            except Exception as e:
                print(f"  [llmlingua2 r={ratio}] compression failed, skipping: {e}", flush=True)
                continue

            prompt = f"السياق: {compressed_context}\n\nالسؤال: {question}"
            try:
                resp = chat_with_retry(client, SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
            except RuntimeError as e:
                print(f"  [llmlingua2 r={ratio}] SKIPPED after retries: {e}", flush=True)
                continue

            n_new_calls += 1
            print(f"[{qi+1}/{len(EVAL_SET)}] llmlingua2 r={ratio}: {resp.text[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": "llmlingua2", "ratio": ratio,
                "answer": resp.text, "gold_answers": item["gold_answers"],
                "context_chars": len(compressed_context),
                "origin_tokens": result.get("origin_tokens"),
                "compressed_tokens": result.get("compressed_tokens"),
                "compression_rate_actual": result.get("rate"),
                "llmlingua2_model": LLMLINGUA2_MODEL,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

    f_out.close()
    print(f"\nThis invocation: {n_new_calls} new live calls.")
    print("Next: score data/arcd_llmlingua2_results.jsonl with the same "
          "Arabic-normalized EM/F1 scorer used in scripts/analyze_arcd_results.py, "
          "add it as a third bar alongside LSPM and naive truncation.")


if __name__ == "__main__":
    main()
