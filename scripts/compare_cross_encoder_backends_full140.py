"""
Extension of compare_cross_encoder_backends.py from a 20-question subset to
the full 140-question ARCD evaluation set, at the same three compression
ratios used throughout the paper (r = 0.3, 0.5, 0.7), in response to a
submission-readiness audit's request to run the full 140 questions rather
than a 20-question subset.

Scope, stated plainly, exactly as for the 20-question version this extends:
this is a selection-agreement and gold-answer-retention check using a
substring-match proxy against the released gold answer, not a full,
generation-based re-run of the paper's token-F1 evaluation on the
BGE-reranker-v2-m3 backend. That would require serving Llama-3.1-8B-Instruct
on a GPU to generate real answers over BGE-pruned contexts for all 420
(question, ratio) cells, which is not available in this environment. The
20-question, r=0.5-only result already in Section 5, item (4b) is superseded
in coverage by this result but not contradicted by it.

Usage (from repo root):
    python scripts/compare_cross_encoder_backends_full140.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, split_sentences

ROOT = Path(__file__).resolve().parent.parent
RATIOS = [0.3, 0.5, 0.7]


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def contains_gold(kept_sentences, gold_answers):
    joined = " ".join(kept_sentences)
    return any(g.strip() and g.strip() in joined for g in gold_answers)


def select_top_k(all_sentences, scores, ratio, min_sentences=1):
    num_to_keep = max(min_sentences, int(round(len(all_sentences) * ratio)))
    num_to_keep = min(num_to_keep, len(all_sentences))
    ranked_idx = sorted(range(len(all_sentences)), key=lambda i: scores[i], reverse=True)
    keep_idx = set(ranked_idx[:num_to_keep])
    return [all_sentences[i] for i in range(len(all_sentences)) if i in keep_idx]


def score_all(model_name, eval_set, label):
    print(f"Loading {label} ({model_name})...", flush=True)
    pruner = SemanticPruner(model_name=model_name, device="cpu")
    out = []
    t0 = time.time()
    for i, item in enumerate(eval_set):
        docs = naive_retrieve_order(item["question"], item["documents"], top_k=6)
        all_sentences = []
        for doc in docs:
            all_sentences.extend(split_sentences(doc))
        scores = pruner.score(item["question"], all_sentences) if all_sentences else []
        out.append((item, all_sentences, scores))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] scored {i + 1}/{len(eval_set)} questions ({elapsed:.0f}s elapsed)...", flush=True)
    print(f"  [{label}] done in {time.time() - t0:.0f}s total.", flush=True)
    return out


def main():
    eval_set = json.load(open(ROOT / "data" / "arcd_eval_set.json", encoding="utf-8"))
    print(f"Loaded {len(eval_set)} ARCD questions.", flush=True)

    a_scored = score_all("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", eval_set, "backend A (mmarco-mMiniLMv2, paper's main backend)")
    b_scored = score_all("BAAI/bge-reranker-v2-m3", eval_set, "backend B (BGE-reranker-v2-m3, alternate backend)")

    for ratio in RATIOS:
        jaccards, agree, n_a_gold, n_b_gold = [], 0, 0, 0
        for (item_a, sents_a, scores_a), (item_b, sents_b, scores_b) in zip(a_scored, b_scored):
            assert item_a["id"] == item_b["id"]
            if not sents_a or not sents_b:
                continue
            ka_list = select_top_k(sents_a, scores_a, ratio)
            kb_list = select_top_k(sents_b, scores_b, ratio)
            ka, kb = set(ka_list), set(kb_list)
            union = ka | kb
            j = len(ka & kb) / len(union) if union else 1.0
            jaccards.append(j)
            ag = contains_gold(ka_list, item_a["gold_answers"])
            bg = contains_gold(kb_list, item_b["gold_answers"])
            n_a_gold += ag
            n_b_gold += bg
            agree += (ag == bg)

        n = len(jaccards)
        print(f"\n=== r = {ratio} (N = {n}) ===")
        print(f"Mean Jaccard overlap of kept-sentence sets: {statistics.mean(jaccards):.3f}")
        print(f"Median Jaccard: {statistics.median(jaccards):.3f}")
        print(f"Exact-match cells (Jaccard=1.0): {sum(1 for j in jaccards if j == 1.0)}/{n}")
        print(f"mmarco-mMiniLMv2 retains gold-answer substring: {n_a_gold}/{n}")
        print(f"BGE-reranker-v2-m3 retains gold-answer substring: {n_b_gold}/{n}")
        print(f"Backends agree on gold-presence outcome: {agree}/{n}")


if __name__ == "__main__":
    main()
