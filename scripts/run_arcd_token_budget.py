"""
Token-budget-matched rerun of the ARCD baseline comparison
------------------------------------------------------------
Reviewer concern (both IJIES reviewers, item 2/8): LSPM and naive
truncation in run_arcd_pilot.py are matched by RETAINED SENTENCE COUNT
(same compression ratio r applied to both), not by retained TOKEN COUNT.
Since LSPM and naive truncation select different sentences, they can end
up with different token counts at the "same" ratio -- so the reported F1
advantage doesn't cleanly mean "better answer quality at equal KV-cache
cost." This script fixes that: for every (question, ratio) cell, it
first prunes with LSPM as normal, counts LSPM's ACTUAL token count with
the real generator tokenizer, then builds naive truncation to hit that
SAME token budget (not the same sentence-count ratio) before generating.

Run this AFTER run_arcd_pilot.py has already produced data/arcd_results.jsonl
(reuses its "raw" and "lspm" cells directly -- LSPM's selection doesn't
change, only naive's target changes -- and only issues new live calls for
the token-budget-matched naive condition).

Requires a HuggingFace account with access to meta-llama/Llama-3.1-8B-Instruct
(gated repo; accept the license on huggingface.co once, then
`huggingface-cli login` or set HF_TOKEN) so token counts use the ACTUAL
generator tokenizer rather than an approximation. If you don't want to
deal with the gated repo, set TOKENIZER_NAME below to a compatible
non-gated tokenizer (e.g. "NousResearch/Meta-Llama-3.1-8B-Instruct" is a
common unofficial mirror with an identical tokenizer) -- note this in
the paper's reproducibility section if you do.

Output: data/arcd_token_budget_results.jsonl (same schema as
arcd_results.jsonl, method="naive_tokbudget"), plus prints a
per-question token-count reconciliation table so you can sanity-check
that naive_tokbudget's token counts now track lspm's rather than
sentence-count-matched naive's.
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

TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "meta-llama/Llama-3.1-8B-Instruct")

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)

RATIOS = [0.3, 0.5, 0.7]
INTER_CALL_DELAY_S = float(os.environ.get("INTER_CALL_DELAY_S", "1.5"))
MAX_RETRIES = 6
MAX_CALLS_PER_INVOCATION = int(os.environ.get("MAX_CALLS_PER_INVOCATION", "9999"))


def load_tokenizer():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {TOKENIZER_NAME} (one-time download, tokenizer files only)")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def naive_truncate_to_token_budget(tok, documents, target_tokens: int):
    """Greedily keep leading sentences (same positional-truncation policy as
    the original naive baseline) until adding the next whole sentence would
    exceed target_tokens, so the two conditions are compared at matched
    (or naive-below-target, never-above-target) prompt token counts."""
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))

    kept = []
    running_tokens = 0
    for s in all_sentences:
        s_tokens = count_tokens(tok, s)
        if kept and running_tokens + s_tokens > target_tokens:
            break
        kept.append(s)
        running_tokens += s_tokens
        if not kept[:-1] and running_tokens > target_tokens:
            # first sentence alone already exceeds budget; keep it anyway
            # (min_sentences=1 policy, matches SemanticPruner's own floor)
            break

    return {
        "pruned_text": " ".join(kept),
        "original_sentence_count": len(all_sentences),
        "kept_sentence_count": len(kept),
        "token_count": running_tokens,
    }


def load_existing_results():
    path = ROOT / "data" / "arcd_results.jsonl"
    by_key = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_key[(rec["id"], rec["method"], rec["ratio"])] = rec
    return by_key


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
    tok = load_tokenizer()
    client = VLLMClient(api_url=API_URL, model_name=MODEL, api_key=API_KEY, timeout_s=60)
    pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    existing = load_existing_results()

    out_path = ROOT / "data" / "arcd_token_budget_results.jsonl"
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
    reconciliation_rows = []

    for qi, item in enumerate(EVAL_SET):
        if n_new_calls >= MAX_CALLS_PER_INVOCATION:
            print("Hit MAX_CALLS_PER_INVOCATION, stopping (re-run to continue).", flush=True)
            break

        qid = item["id"]
        question = item["question"]
        docs = naive_retrieve_order(question, item["documents"], top_k=6)

        for ratio in RATIOS:
            lspm_rec = existing.get((qid, "lspm", ratio))
            if lspm_rec is None:
                # LSPM cell not yet generated by run_arcd_pilot.py -- prune locally
                # just to get the token budget (doesn't need a live call).
                pr = pruner.prune(question, docs, compression_ratio=ratio)
                lspm_pruned_text = pr.pruned_text
            else:
                lspm_pruned_text = None  # unknown text, but we still need a token count
                pr = pruner.prune(question, docs, compression_ratio=ratio)
                lspm_pruned_text = pr.pruned_text

            target_tokens = count_tokens(tok, lspm_pruned_text)

            key = (qid, "naive_tokbudget", ratio)
            if key in done_cells:
                continue
            if n_new_calls >= MAX_CALLS_PER_INVOCATION:
                break

            nb = naive_truncate_to_token_budget(tok, docs, target_tokens)
            reconciliation_rows.append((qid, ratio, target_tokens, nb["token_count"]))

            prompt = f"السياق: {nb['pruned_text']}\n\nالسؤال: {question}"
            try:
                resp = chat_with_retry(client, SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=64)
            except RuntimeError as e:
                print(f"  [naive_tokbudget r={ratio}] SKIPPED: {e}", flush=True)
                continue

            n_new_calls += 1
            print(f"[{qi+1}/{len(EVAL_SET)}] naive_tokbudget r={ratio} "
                  f"(target={target_tokens} actual={nb['token_count']} toks): {resp.text[:60]}", flush=True)
            f_out.write(json.dumps({
                "id": qid, "question": question, "method": "naive_tokbudget", "ratio": ratio,
                "answer": resp.text, "gold_answers": item["gold_answers"],
                "context_chars": len(nb["pruned_text"]), "context_sentences": nb["kept_sentence_count"],
                "lspm_target_tokens": target_tokens, "actual_tokens": nb["token_count"],
                "pruner_latency_ms": None,
            }, ensure_ascii=False) + "\n")
            f_out.flush()

    f_out.close()

    if reconciliation_rows:
        print("\n--- token-budget reconciliation (first 10) ---")
        for row in reconciliation_rows[:10]:
            print(f"  q={row[0]} r={row[1]}: LSPM tokens={row[2]}, naive_tokbudget tokens={row[3]}")
    print(f"\nThis invocation: {n_new_calls} new live calls.")
    print("Next: extend scripts/analyze_arcd_results.py (or copy it) to also load "
          "method=='naive_tokbudget' and report EM/F1 for LSPM vs. naive_tokbudget "
          "alongside the existing LSPM vs. naive comparison -- this is the number the "
          "reviewers actually asked for.")


if __name__ == "__main__":
    main()
