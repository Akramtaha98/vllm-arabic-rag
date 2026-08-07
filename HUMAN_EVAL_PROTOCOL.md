# Human Evaluation Protocol

Purpose: automatic EM/F1 metrics can be fooled by near-matches, paraphrase,
or partial credit that isn't semantically correct. This protocol has 1-2
bilingual (Arabic/English) evaluators independently judge a sample of real
answers, blind to which method produced them, to validate that the
paper's automatic-metric result (LSPM ≈ raw, both > naive truncation, at
r = 0.3) holds up under human judgment.

## 1. What's already prepared

- `data/human_eval_sample_blind.csv` — 90 rows (30 ARCD questions × 3
  conditions: raw, LSPM@r=0.3, naive truncation@r=0.3), fully shuffled so
  the three answers to the same question are not adjacent, with random
  opaque `item_id`s (e.g. `HE-482913`) instead of anything that reveals
  method or order.
- `data/human_eval_codebook.json` — the private mapping from `item_id` to
  the real `(question_id, method, ratio)`. **Do not share this with
  evaluators until they have finished rating.**

Each row shown to evaluators has: `question`, `source_context` (the full,
unpruned 6-document context — the same for all three conditions, so it
doesn't leak which method produced the answer), `gold_answers` (from
ARCD), and `candidate_answer` (what the model actually said).

## 2. Recruiting evaluators

You need 1-2 people fluent in Arabic (reading comprehension is what
matters; native or near-native is ideal) who did not build this system and
are not aware of which method is expected to win. Do not tell them the
paper's hypothesis before they rate — tell them only the rubric below.

If you have 2 evaluators, have both rate all 90 rows independently
(without conferring) — this is what lets `analyze_human_eval.py` compute
inter-rater agreement (Cohen's kappa), which reviewers will want to see
alongside the mean scores. If you only have 1 evaluator, that's still
useful evidence, just without the agreement statistic — say so plainly in
the paper rather than fabricating a second rater.

## 3. The rubric (give evaluators this section verbatim)

For each row, read the question, the source context, the gold answer(s),
and the candidate answer. Then score two things:

**Correctness** (0, 1, or 2):
- **2 — Fully correct.** The candidate answer states the same fact as the
  gold answer(s), even if worded differently (e.g. "عام 1968" vs. "1968").
- **1 — Partially correct.** The answer is on-topic and contains some
  correct information but is incomplete, imprecise, or only partially
  matches the gold answer.
- **0 — Incorrect.** The answer states a different fact than the gold
  answer(s), or fails to answer the question.

**Faithfulness** (0 or 1):
- **1 — Grounded.** Every claim in the candidate answer is actually
  supported by the source context provided. The model did not invent
  information not present in the context (even if that invented
  information happens to be true).
- **0 — Not grounded.** The answer includes a claim not supported by the
  source context (a hallucination), or the answer contradicts the source
  context.

Note: correctness and faithfulness are independent. An answer can be
correct but unfaithful (right answer, but not actually derivable from the
given context — e.g. the model used outside/prior knowledge), or faithful
but incorrect (the context itself doesn't contain the gold answer and the
model reasonably said so, or picked the wrong supported fact).

Fill in `correctness_rater1` / `faithfulness_rater1` (and `_rater2` if you
have a second evaluator) for every row. Use the `notes_raterN` column
freely for anything ambiguous — you don't need to resolve disagreements
before analysis; the kappa statistic is the point of having two raters.

## 4. Blinding — why the same context is shown for every row

You'll notice `source_context` is always the full, unpruned context, even
for the LSPM and naive-truncation answers, which were actually generated
from a shorter, pruned version of that context. This is deliberate, for
two reasons:

1. Showing the actual pruned/truncated context each method saw would
   immediately reveal which method produced the answer (raw context is
   long and complete; naive truncation visibly cuts off mid-sentence;
   LSPM's selection is topically coherent but shorter) — breaking the
   blind.
2. Faithfulness should be judged against the full source of truth, not
   against whatever partial slice one method happened to see. An answer
   that pulls in a true fact from a sentence LSPM discarded but naive
   truncation kept is still "faithful" in the sense that matters for the
   paper's claim (is the RAG system hallucinating, or grounding its
   answers in the retrieved corpus).

## 5. After rating: run the analysis

```bash
python scripts/analyze_human_eval.py
```

This reads the filled `data/human_eval_sample_blind.csv` back against the
private codebook and prints/writes (`results/human_eval_summary.csv`):
mean correctness and faithfulness per method with 95% bootstrap CIs, and
Cohen's kappa between raters if both columns are filled.

Send me `results/human_eval_summary.csv` (and the filled
`data/human_eval_sample_blind.csv` if you're comfortable sharing the raw
ratings) and I will fold the human-evaluation numbers into a new Results
subsection and revise the Limitations section to note this replaces
"automatic metrics only" as a stated weakness.
