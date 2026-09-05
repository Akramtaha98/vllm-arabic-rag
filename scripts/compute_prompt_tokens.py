"""
Re-derives the exact prompt text of every (question, method, ratio) cell of the
ARCD ground-truth re-test (data/arcd_results.jsonl) by replaying the
deterministic pipeline of scripts/run_arcd_pilot.py, then counts real prompt
tokens with the Llama-3.1-8B-Instruct tokenizer.

Correctness is verified, not assumed: for every cell the reconstructed context's
character count and sentence count are compared against the values that
run_arcd_pilot.py recorded live in arcd_results.jsonl. Any mismatch is reported.

No live LLM calls are made. Everything runs on CPU from data already in the repo.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from middleware.pruning import SemanticPruner, split_sentences  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKENIZER_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct"
CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RATIOS = [0.3, 0.5, 0.7]

SYSTEM_PROMPT = (
    "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك. "
    "أجب بعبارة قصيرة ومباشرة تحتوي فقط على الإجابة المطلوبة، دون شرح إضافي."
)


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
    return {"pruned_text": " ".join(kept), "kept_sentence_count": len(kept)}


def main():
    eval_set = json.load(open(ROOT / "data/arcd_eval_set.json", encoding="utf-8"))
    recs = [json.loads(l) for l in open(ROOT / "data/arcd_results.jsonl", encoding="utf-8") if l.strip()]
    recorded = {(r["id"], r["method"], r["ratio"]): r for r in recs}

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    pruner = SemanticPruner(model_name=CROSS_ENCODER, device="cpu")

    def n_tokens(context, question):
        """Full prompt as sent to the generator: system message + user message,
        rendered through the model's real chat template."""
        user = f"السياق: {context}\n\nالسؤال: {question}"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return len(tok.encode(text, add_special_tokens=False))

    outpath = ROOT / "results" / "arcd_prompt_tokens.jsonl"
    done_qids = set()
    if outpath.exists():
        for line in open(outpath, encoding="utf-8"):
            if line.strip():
                done_qids.add(json.loads(line)["id"])
    f_out = open(outpath, "a", encoding="utf-8")
    print(f"Resuming: {len(done_qids)} questions already done.", flush=True)

    out = []
    mismatches = []
    for qi, item in enumerate(eval_set):
        qid = item["id"]
        if qid in done_qids:
            continue
        question = item["question"]
        docs = naive_retrieve_order(question, item["documents"], top_k=6)
        raw_context = " ".join(docs)

        cells = [("raw", 1.0, raw_context, len(split_sentences(raw_context)))]
        for ratio in RATIOS:
            pr = pruner.prune(question, docs, compression_ratio=ratio)
            cells.append(("lspm", ratio, pr.pruned_text, pr.kept_sentence_count))
            nb = naive_truncate(docs, ratio)
            cells.append(("naive", ratio, nb["pruned_text"], nb["kept_sentence_count"]))

        for method, ratio, ctx, n_sent in cells:
            rec = recorded.get((qid, method, ratio))
            if rec is None:
                mismatches.append((qid, method, ratio, "MISSING_RECORD"))
                continue
            if len(ctx) != rec["context_chars"] or n_sent != rec["context_sentences"]:
                mismatches.append((qid, method, ratio,
                                   f"chars {len(ctx)} vs {rec['context_chars']}, "
                                   f"sent {n_sent} vs {rec['context_sentences']}"))
            row = {
                "id": qid, "method": method, "ratio": ratio,
                "context_chars": len(ctx), "context_sentences": n_sent,
                "prompt_tokens": n_tokens(ctx, question),
                "context_tokens": len(tok.encode(ctx, add_special_tokens=False)),
                "verified_against_live_record": rec is not None
                and len(ctx) == rec["context_chars"]
                and n_sent == rec["context_sentences"],
            }
            out.append(row)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_out.flush()
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(eval_set)} questions done", flush=True)

    f_out.close()
    print(f"\nWrote {len(out)} cells -> {outpath}")
    print(f"Verification mismatches vs. live-recorded values: {len(mismatches)}")
    for m in mismatches[:20]:
        print("  ", m)


if __name__ == "__main__":
    main()
