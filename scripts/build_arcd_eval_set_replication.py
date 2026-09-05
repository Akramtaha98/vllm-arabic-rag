"""
Builds a FRESH, DISJOINT ARCD evaluation set for the independent
fresh-seed replication (paper Section 4.10), reusing the exact same
sampling/distractor procedure as build_arcd_eval_set.py, but:

  - excludes every question id already used in data/arcd_eval_set.json
    (the original 140-question pilot/re-test set), so none of these
    questions were seen during the original r=0.3 result;
  - uses a DIFFERENT random seed (2026, not 42), so the distractor draws
    and document shuffles are independently sampled, not just a
    re-filtering of the same draw;
  - draws N_QUESTIONS=140 fresh questions from the 562 remaining ARCD
    validation examples not already used (702 total - 140 used).

ARCD only has 78 distinct source articles in total, and the original set
already touched all 78 of them (140 questions > 78 articles), so this
fresh set cannot also be topic-disjoint at the article level -- that is
a property of the underlying dataset, not a limitation of this script.
What IS guaranteed disjoint is the actual question id: none of these 140
questions were answered by any model in the original pilot/re-test.

Writes data/arcd_eval_set_replication.json in the same schema as
data/arcd_eval_set.json:
    [
      {"id", "title", "question", "gold_answers": [...], "documents": [...6],
       "gold_doc_index": int},
      ...
    ]
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_EVAL_SET = ROOT / "data" / "arcd_eval_set.json"
OUT_PATH = ROOT / "data" / "arcd_eval_set_replication.json"
N_QUESTIONS = 140
N_DISTRACTORS = 5
SEED = 2026  # deliberately different from the original SEED=42

random.seed(SEED)

original = json.load(open(ORIGINAL_EVAL_SET, encoding="utf-8"))
used_ids = {ex["id"] for ex in original}
print(f"Excluding {len(used_ids)} question ids already used in the original set.")

print("Loading ARCD (hsseinmz/arcd) from Hugging Face...")
ds = load_dataset("hsseinmz/arcd", split="validation")
print(f"Loaded {len(ds)} validation examples.")

by_title = {}
for ex in ds:
    by_title.setdefault(ex["title"], []).append(ex)
titles = list(by_title.keys())

candidates = [ex for ex in ds if ex["id"] not in used_ids and ex["answers"]["text"]]
print(f"{len(candidates)} unused, answerable examples available to sample from.")
if len(candidates) < N_QUESTIONS:
    raise SystemExit(
        f"Only {len(candidates)} unused examples available, need {N_QUESTIONS}. "
        "Lower N_QUESTIONS or verify data/arcd_eval_set.json is the correct original set."
    )

random.shuffle(candidates)
sampled = candidates[:N_QUESTIONS]
sampled_titles = {ex["title"] for ex in sampled}
print(f"Sampled {len(sampled)} fresh questions, spanning {len(sampled_titles)} of the {len(titles)} total articles.")

eval_set = []
for ex in sampled:
    gold_title = ex["title"]
    gold_context = ex["context"]

    other_titles = [t for t in titles if t != gold_title]
    distractor_titles = random.sample(other_titles, min(N_DISTRACTORS, len(other_titles)))
    distractors = [random.choice(by_title[t])["context"] for t in distractor_titles]

    documents = [gold_context] + distractors
    order = list(range(len(documents)))
    random.shuffle(order)
    shuffled_docs = [documents[i] for i in order]
    gold_doc_index = order.index(0)

    eval_set.append({
        "id": ex["id"],
        "title": gold_title,
        "question": ex["question"],
        "gold_answers": list(dict.fromkeys(ex["answers"]["text"])),
        "documents": shuffled_docs,
        "gold_doc_index": gold_doc_index,
    })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_set, f, ensure_ascii=False, indent=2)

overlap = used_ids & {ex["id"] for ex in eval_set}
assert not overlap, f"BUG: {len(overlap)} ids overlap with the original set: {overlap}"

print(f"\nWrote {len(eval_set)} fresh, disjoint eval items to {OUT_PATH}")
print("Verified: zero id overlap with the original 140-question set.")
