"""
Builds a real, larger, noisier evaluation set from ARCD (Arabic Reading
Comprehension Dataset, hsseinmz/arcd on Hugging Face) to replace the
original pilot's small, low-redundancy, synthetic 10-document knowledge
base and its similarity-to-unpruned-answer-only evaluation.

For each sampled question:
  - the gold paragraph (from ARCD) is the one that actually contains the
    answer
  - 5 distractor paragraphs are drawn from OTHER ARCD articles, so the
    6-document "retrieved" pool resembles a real, noisy top-k retrieval
    result rather than a curated single-answer context
  - a naive keyword-overlap retriever ranks/orders these 6 documents (same
    retrieval mechanism as the original pilot), so the gold paragraph is
    not guaranteed to be first
  - the real gold answer string(s) are kept for EM/F1 scoring against
    ground truth, not just similarity-to-unpruned-answer

Writes data/arcd_eval_set.json:
    [
      {"id", "question", "gold_answers": [...], "documents": [doc, ...6],
       "gold_doc_index": int},
      ...
    ]
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "arcd_eval_set.json"
N_QUESTIONS = 140
N_DISTRACTORS = 5
SEED = 42

random.seed(SEED)

print("Loading ARCD (hsseinmz/arcd) from Hugging Face...")
ds = load_dataset("hsseinmz/arcd", split="validation")
print(f"Loaded {len(ds)} validation examples.")

# Group by article title so distractors can be drawn from genuinely
# different articles (not just different paragraphs of the same one).
by_title = {}
for ex in ds:
    by_title.setdefault(ex["title"], []).append(ex)

titles = list(by_title.keys())
print(f"{len(titles)} distinct articles available for distractor sampling.")

# Sample N_QUESTIONS examples, each from a different article where possible,
# to maximize topical diversity in the eval set.
all_examples = list(ds)
random.shuffle(all_examples)

seen_titles = set()
sampled = []
for ex in all_examples:
    if len(sampled) >= N_QUESTIONS:
        break
    if ex["title"] in seen_titles:
        continue
    if not ex["answers"]["text"]:
        continue  # skip unanswerable/malformed rows, if any
    seen_titles.add(ex["title"])
    sampled.append(ex)

# If we couldn't get N_QUESTIONS from distinct articles (small dataset),
# top up from remaining examples regardless of title repeats.
if len(sampled) < N_QUESTIONS:
    for ex in all_examples:
        if len(sampled) >= N_QUESTIONS:
            break
        if ex in sampled:
            continue
        if not ex["answers"]["text"]:
            continue
        sampled.append(ex)

print(f"Sampled {len(sampled)} questions from {len(seen_titles)} distinct articles.")

eval_set = []
for ex in sampled:
    gold_title = ex["title"]
    gold_context = ex["context"]

    # Distractor pool: paragraphs from any OTHER article.
    other_titles = [t for t in titles if t != gold_title]
    distractor_titles = random.sample(other_titles, min(N_DISTRACTORS, len(other_titles)))
    distractors = [random.choice(by_title[t])["context"] for t in distractor_titles]

    documents = [gold_context] + distractors
    # Shuffle so the gold paragraph isn't always first (simulates realistic,
    # not-perfectly-ranked retrieval) -- record its post-shuffle index.
    order = list(range(len(documents)))
    random.shuffle(order)
    shuffled_docs = [documents[i] for i in order]
    gold_doc_index = order.index(0)

    eval_set.append({
        "id": ex["id"],
        "title": gold_title,
        "question": ex["question"],
        "gold_answers": list(dict.fromkeys(ex["answers"]["text"])),  # dedupe, preserve order
        "documents": shuffled_docs,
        "gold_doc_index": gold_doc_index,
    })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_set, f, ensure_ascii=False, indent=2)

print(f"\nWrote {len(eval_set)} eval items to {OUT_PATH}")
print(f"Each item has 1 gold paragraph + {N_DISTRACTORS} distractor paragraphs from other articles (6 total, shuffled).")
