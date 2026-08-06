"""
Builds a genuinely end-to-end retrieval evaluation set: unlike
build_arcd_eval_set.py (which fixes a 6-document pool that always contains
the gold, answer-bearing paragraph -- recall@6 = 100% by construction), this
script retrieves the top-6 documents for each question from a much larger
candidate corpus using a real, standard sparse retriever (TF-IDF cosine
similarity, scikit-learn), so the gold paragraph may or may not be retrieved
at all. This directly addresses the limitation disclosed throughout the
paper (Section 4.4, 6, 7, 8): the ARCD re-test evaluates pruning conditional
on successful retrieval, not retrieval itself.

Corpus: all 234 distinct context paragraphs across ARCD's 78 validation-split
articles (a real article can contribute more than one distinct paragraph, so
retrieval must discriminate between paragraphs from the same and different
articles, not just between articles).

Retriever: TF-IDF (word n-grams 1-2, Arabic-aware whitespace tokenization)
fit once on the full 234-paragraph corpus; each question is transformed with
the same vectorizer and ranked against all 234 candidates by cosine
similarity. This is a real, standard, reproducible sparse-retrieval baseline
-- not the paper's existing "naive keyword-overlap re-ranker" (which only
ever re-orders an already answer-guaranteed 6-document pool) and not a mock.

Writes data/e2e_retrieval_eval_set.json:
    [
      {"id", "question", "gold_answers": [...], "gold_context": str,
       "retrieved_documents": [doc, ...6], "gold_in_topk": bool,
       "gold_rank": int or null},
      ...
    ]
"""
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "e2e_retrieval_eval_set.json"
N_QUESTIONS = 60
TOP_K = 6
SEED = 123

random.seed(SEED)

print("Loading ARCD (hsseinmz/arcd) from Hugging Face...")
ds = load_dataset("hsseinmz/arcd", split="validation")
print(f"Loaded {len(ds)} validation examples.")

all_examples = list(ds)

# Corpus: every distinct context paragraph in the validation split.
contexts = []
seen_ctx = set()
for ex in all_examples:
    if ex["context"] not in seen_ctx:
        seen_ctx.add(ex["context"])
        contexts.append(ex["context"])
print(f"Corpus: {len(contexts)} distinct context paragraphs.")

vectorizer = TfidfVectorizer()
corpus_matrix = vectorizer.fit_transform(contexts)
context_to_index = {ctx: i for i, ctx in enumerate(contexts)}

# Sample N_QUESTIONS questions (fresh sample, different seed/pool from the
# 140-question ARCD re-test in build_arcd_eval_set.py, so this is an
# independent check rather than the same items re-scored).
candidates = [ex for ex in all_examples if ex["answers"]["text"]]
random.shuffle(candidates)
sampled = candidates[:N_QUESTIONS]
print(f"Sampled {len(sampled)} questions for the end-to-end retrieval eval.")

eval_set = []
n_recall_hits = 0
for ex in sampled:
    question = ex["question"]
    gold_context = ex["context"]
    gold_idx = context_to_index[gold_context]

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, corpus_matrix)[0]
    ranked_indices = np.argsort(-sims)

    top_k_indices = ranked_indices[:TOP_K].tolist()
    retrieved_documents = [contexts[i] for i in top_k_indices]

    gold_in_topk = gold_idx in top_k_indices
    if gold_in_topk:
        n_recall_hits += 1
        gold_rank = top_k_indices.index(gold_idx) + 1
    else:
        # Full rank among all 234 candidates (1-indexed), for diagnostics.
        gold_rank = int(np.where(ranked_indices == gold_idx)[0][0]) + 1

    eval_set.append({
        "id": ex["id"],
        "title": ex["title"],
        "question": question,
        "gold_answers": list(dict.fromkeys(ex["answers"]["text"])),
        "gold_context": gold_context,
        "retrieved_documents": retrieved_documents,
        "gold_in_topk": gold_in_topk,
        "gold_rank": gold_rank,
    })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_set, f, ensure_ascii=False, indent=2)

print(f"\nWrote {len(eval_set)} eval items to {OUT_PATH}")
print(f"Recall@{TOP_K} (gold paragraph retrieved): {n_recall_hits}/{len(eval_set)} = {100*n_recall_hits/len(eval_set):.1f}%")
