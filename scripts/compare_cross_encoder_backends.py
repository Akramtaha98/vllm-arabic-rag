"""
Real, CPU-feasible check of cross-encoder backend sensitivity for Reviewer
3's backend-variance request: compares sentence-selection agreement between
the paper's main backend (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) and
the reference implementation's configurable alternative (BAAI/bge-reranker-v2-m3)
on a 20-question subset of the ARCD evaluation set at r = 0.5.

This is a selection-agreement check (Jaccard overlap of kept-sentence sets,
and whether the gold-answer substring survives pruning under each backend),
not a full re-run of the paper's F1 evaluation or serving-cost benchmark on
the alternate backend -- see Section 5, item (4b) for exactly what this
does and does not establish.

Usage (from repo root):
    python scripts/compare_cross_encoder_backends.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner

ROOT = Path(__file__).resolve().parent.parent
N_QUESTIONS = 20
RATIO = 0.5


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def contains_gold(kept_sentences, gold_answers):
    joined = " ".join(kept_sentences)
    return any(g.strip() and g.strip() in joined for g in gold_answers)


def run_backend(model_name, subset):
    pruner = SemanticPruner(model_name=model_name, device="cpu")
    out = []
    for item in subset:
        docs = naive_retrieve_order(item["question"], item["documents"], top_k=6)
        res = pruner.prune(item["question"], docs, compression_ratio=RATIO)
        out.append({"id": item["id"], "kept_sentences": res.kept_sentences, "gold_answers": item["gold_answers"]})
    return out


def main():
    eval_set = json.load(open(ROOT / "data" / "arcd_eval_set.json", encoding="utf-8"))
    subset = eval_set[:N_QUESTIONS]

    print("Running backend A: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (paper's main backend)...")
    a = run_backend("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", subset)

    print("Running backend B: BAAI/bge-reranker-v2-m3 (alternate backend)...")
    b = run_backend("BAAI/bge-reranker-v2-m3", subset)

    jaccards, agree, n_a_gold, n_b_gold = [], 0, 0, 0
    for ra, rb in zip(a, b):
        ka, kb = set(ra["kept_sentences"]), set(rb["kept_sentences"])
        union = ka | kb
        j = len(ka & kb) / len(union) if union else 1.0
        jaccards.append(j)
        ag, bg = contains_gold(ra["kept_sentences"], ra["gold_answers"]), contains_gold(rb["kept_sentences"], rb["gold_answers"])
        n_a_gold += ag
        n_b_gold += bg
        agree += (ag == bg)

    print(f"\nN = {len(a)} questions, ratio r = {RATIO}")
    print(f"Mean Jaccard overlap of kept-sentence sets: {statistics.mean(jaccards):.3f}")
    print(f"Median Jaccard: {statistics.median(jaccards):.3f}")
    print(f"Exact-match cells (Jaccard=1.0): {sum(1 for j in jaccards if j == 1.0)}/{len(jaccards)}")
    print(f"mmarco-mMiniLMv2 retains gold-answer substring: {n_a_gold}/{len(a)}")
    print(f"BGE-reranker-v2-m3 retains gold-answer substring: {n_b_gold}/{len(a)}")
    print(f"Backends agree on gold-presence outcome: {agree}/{len(a)}")


if __name__ == "__main__":
    main()
