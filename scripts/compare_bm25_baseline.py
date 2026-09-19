"""
Real, CPU-feasible check of a classical lexical baseline for Reviewer 3's
Comment 5 ("additional baseline comparisons... or reference ongoing
work/plans"): compares sentence-selection agreement between LSPM's
cross-encoder (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) and a BM25
sentence-ranker on all 140 questions of the ARCD evaluation set, at the
same three compression ratios used throughout the paper (r = 0.3, 0.5, 0.7).

Scope, stated plainly: this is a selection-agreement and gold-answer-
retention check using the same substring-match proxy methodology as
Section 5, item (4b)'s cross-encoder-backend comparison, not a full,
generation-based re-run of the paper's token-F1 evaluation with BM25
substituted for the cross-encoder. That would require serving Llama-3.1
-8B-Instruct on a GPU to generate real answers over BM25-pruned contexts,
which is not available in this environment; a full BM25 quality/serving
comparison remains future work exactly like the extra baselines the
reviewer named (DAC, ACC-RAG, RECOMP).

No external BM25 package is required; Okapi BM25 (k1=1.5, b=0.75) is
implemented directly below over each question's own candidate-pool
sentences (a fresh, per-question corpus), mirroring how the cross-encoder
scores sentences only within that question's own pool.

Usage (from repo root):
    python scripts/compare_bm25_baseline.py
"""
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, split_sentences

ROOT = Path(__file__).resolve().parent.parent
RATIOS = [0.3, 0.5, 0.7]
K1, B = 1.5, 0.75


def naive_retrieve_order(query, documents, top_k=6):
    q_tokens = set(query.split())
    scored = [(len(q_tokens & set(doc.split())), doc) for doc in documents]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def tokenize(text):
    return text.split()


def bm25_scores(query_tokens, sentence_token_lists):
    n = len(sentence_token_lists)
    if n == 0:
        return []
    avgdl = sum(len(s) for s in sentence_token_lists) / n
    df = Counter()
    for s in sentence_token_lists:
        for t in set(s):
            df[t] += 1
    idf = {t: math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0) for t in df}

    scores = []
    for s in sentence_token_lists:
        tf = Counter(s)
        dl = len(s)
        score = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            f = tf[t]
            score += idf.get(t, 0.0) * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avgdl))
        scores.append(score)
    return scores


def bm25_prune(query, documents, compression_ratio):
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    if not all_sentences:
        return []
    n = len(all_sentences)
    k = max(1, round(compression_ratio * n))
    q_tokens = tokenize(query)
    sent_tokens = [tokenize(s) for s in all_sentences]
    scores = bm25_scores(q_tokens, sent_tokens)
    ranked_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    keep_idx = sorted(ranked_idx)  # restore original order, same convention as LSPM
    return [all_sentences[i] for i in keep_idx]


def contains_gold(kept_sentences, gold_answers):
    joined = " ".join(kept_sentences)
    return any(g.strip() and g.strip() in joined for g in gold_answers)


def select_top_k(all_sentences, scores, ratio, min_sentences=1):
    num_to_keep = max(min_sentences, int(round(len(all_sentences) * ratio)))
    num_to_keep = min(num_to_keep, len(all_sentences))
    ranked_idx = sorted(range(len(all_sentences)), key=lambda i: scores[i], reverse=True)
    keep_idx = set(ranked_idx[:num_to_keep])
    return [all_sentences[i] for i in range(len(all_sentences)) if i in keep_idx]


def main():
    eval_set = json.load(open(ROOT / "data" / "arcd_eval_set.json", encoding="utf-8"))
    print(f"Loaded {len(eval_set)} ARCD questions.", flush=True)

    # Score each question's sentence pool with the cross-encoder ONCE and reuse
    # across all three ratios (matches SemanticPruner.prune()'s own selection
    # logic exactly; avoids rerunning the cross-encoder 3x per question).
    pruner = SemanticPruner(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")

    per_question = []
    for i, item in enumerate(eval_set):
        docs = naive_retrieve_order(item["question"], item["documents"], top_k=6)
        all_sentences = []
        for doc in docs:
            all_sentences.extend(split_sentences(doc))
        ce_scores = pruner.score(item["question"], all_sentences) if all_sentences else []
        per_question.append((item, all_sentences, ce_scores, docs))
        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{len(eval_set)} questions...", flush=True)

    out_records = []
    for ratio in RATIOS:
        jaccards, agree, n_lspm_gold, n_bm25_gold = [], 0, 0, 0
        for item, all_sentences, ce_scores, docs in per_question:
            if not all_sentences:
                continue
            lspm_kept = select_top_k(all_sentences, ce_scores, ratio)
            bm25_kept = bm25_prune(item["question"], docs, ratio)

            ka, kb = set(lspm_kept), set(bm25_kept)
            union = ka | kb
            j = len(ka & kb) / len(union) if union else 1.0
            jaccards.append(j)

            lg = contains_gold(lspm_kept, item["gold_answers"])
            bg = contains_gold(bm25_kept, item["gold_answers"])
            n_lspm_gold += lg
            n_bm25_gold += bg
            agree += (lg == bg)
            out_records.append({
                "question_id": item["id"], "ratio": ratio, "jaccard": j,
                "lspm_retains_gold": lg, "bm25_retains_gold": bg,
            })

        n = len(jaccards)
        print(f"\n=== r = {ratio} (N = {n}) ===")
        print(f"Mean Jaccard overlap of kept-sentence sets (LSPM vs BM25): {statistics.mean(jaccards):.3f}")
        print(f"Median Jaccard: {statistics.median(jaccards):.3f}")
        print(f"Exact-match cells (Jaccard=1.0): {sum(1 for j in jaccards if j == 1.0)}/{n}")
        print(f"LSPM (cross-encoder) retains gold-answer substring: {n_lspm_gold}/{n}")
        print(f"BM25 retains gold-answer substring: {n_bm25_gold}/{n}")
        print(f"Methods agree on gold-presence outcome: {agree}/{n}")

    out_path = ROOT / "results" / "bm25_baseline_comparison.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_records)} rows to {out_path}")


if __name__ == "__main__":
    main()
