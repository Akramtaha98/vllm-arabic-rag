"""
Internal helper (not part of the released reproducibility scripts): scores a
slice of the ARCD eval set with one cross-encoder backend and writes raw
per-question sentence lists + scores to a JSON file, so the full-140
backend comparison can be run in time-bounded chunks and combined afterward.
Used to build the results reported by compare_cross_encoder_backends_full140.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, split_sentences

ROOT = Path(__file__).resolve().parent.parent


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def main():
    model_name = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    out_path = sys.argv[4]

    eval_set = json.load(open(ROOT / "data" / "arcd_eval_set.json", encoding="utf-8"))
    subset = eval_set[start:end]
    print(f"Scoring [{start}:{end}] of {len(eval_set)} with {model_name}...", flush=True)

    pruner = SemanticPruner(model_name=model_name, device="cpu")
    results = []
    for i, item in enumerate(subset):
        docs = naive_retrieve_order(item["question"], item["documents"], top_k=6)
        all_sentences = []
        for doc in docs:
            all_sentences.extend(split_sentences(doc))
        scores = pruner.score(item["question"], all_sentences) if all_sentences else []
        results.append({
            "id": item["id"],
            "sentences": all_sentences,
            "scores": scores,
            "gold_answers": item["gold_answers"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(subset)}", flush=True)

    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Wrote {len(results)} records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
