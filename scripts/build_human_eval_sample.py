"""
Builds a blind human-evaluation sample for the paper's headline claim
(LSPM vs. naive truncation vs. raw at r=0.3, the most aggressive ratio,
where LSPM significantly beats naive on F1 and is equivalent to raw).

For N_QUESTIONS randomly sampled ARCD questions, pulls all three answers
(raw, lspm@0.3, naive@0.3) from data/arcd_results.jsonl, giving
N_QUESTIONS * 3 answer instances (default 30 * 3 = 90, inside the
paper-review-recommended 50-100 range).

Blinding: each answer instance gets a random opaque item_id; rows are
fully shuffled (not grouped by question, so the same question's three
answers don't appear adjacently); the CSV given to evaluators never
contains the method/ratio label. That mapping is written SEPARATELY to
data/human_eval_codebook.json, which must NOT be shown to evaluators
before they finish rating -- only used afterward by analyze_human_eval.py.

For faithfulness judging, each row includes the full RAW (unpruned)
source context, regardless of which method produced the candidate answer.
This is deliberate: showing the pruned/truncated context would leak which
method produced the answer (raw is long and coherent, naive truncation
often cuts off mid-topic, LSPM's selection is topically coherent but
shorter) and break blinding. Judging faithfulness against the same
full-context "ground truth" for every row is also the methodologically
correct choice -- faithfulness means "is this answer supported by the
source material," not "is this answer supported by the specific context
slice this one method happened to see."

Usage:
    python scripts/build_human_eval_sample.py --n-questions 30 --seed 42

Outputs:
    data/human_eval_sample_blind.csv   -- give this to evaluators
    data/human_eval_codebook.json      -- keep private until after rating
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_results(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_eval_set(path):
    items = json.load(open(path, encoding="utf-8"))
    return {item["id"]: item for item in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=30,
                     help="Number of ARCD questions to sample (x3 conditions each)")
    ap.add_argument("--ratio", type=float, default=0.3,
                     help="Compression ratio to evaluate for lspm/naive (paper's headline ratio)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="data/arcd_results.jsonl")
    ap.add_argument("--eval-set", default="data/arcd_eval_set_140.json")
    ap.add_argument("--out-csv", default="data/human_eval_sample_blind.csv")
    ap.add_argument("--out-codebook", default="data/human_eval_codebook.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    results = load_results(ROOT / args.results)
    eval_set = load_eval_set(ROOT / args.eval_set)

    all_qids = sorted(set(r["id"] for r in results))
    if args.n_questions > len(all_qids):
        raise SystemExit(f"Requested {args.n_questions} questions but only {len(all_qids)} available.")
    sampled_qids = rng.sample(all_qids, args.n_questions)

    # Index results by (id, method, ratio) for quick lookup.
    by_key = {}
    for r in results:
        by_key[(r["id"], r["method"], r.get("ratio"))] = r

    conditions = [("raw", 1.0), ("lspm", args.ratio), ("naive", args.ratio)]

    instances = []
    for qid in sampled_qids:
        item = eval_set.get(qid)
        if item is None:
            print(f"WARNING: {qid} not found in {args.eval_set}, skipping")
            continue
        full_context = " ".join(item["documents"])
        gold = item["gold_answers"]
        for method, ratio in conditions:
            r = by_key.get((qid, method, ratio))
            if r is None:
                print(f"WARNING: no result for ({qid}, {method}, {ratio}), skipping this instance")
                continue
            instances.append({
                "question_id": qid,
                "method": method,
                "ratio": ratio,
                "question": r["question"],
                "full_context": full_context,
                "gold_answers": gold,
                "answer": r["answer"],
            })

    rng.shuffle(instances)  # decouple the 3 per-question rows from each other

    # Assign opaque, non-sequential item IDs (won't reveal grouping or order sampled).
    id_pool = list(range(100000, 999999))
    rng.shuffle(id_pool)
    item_ids = [f"HE-{id_pool[i]}" for i in range(len(instances))]

    codebook = {}
    blind_rows = []
    for item_id, inst in zip(item_ids, instances):
        codebook[item_id] = {
            "question_id": inst["question_id"],
            "method": inst["method"],
            "ratio": inst["ratio"],
        }
        blind_rows.append({
            "item_id": item_id,
            "question": inst["question"],
            "source_context": inst["full_context"],
            "gold_answers": " | ".join(inst["gold_answers"]),
            "candidate_answer": inst["answer"],
            "correctness_rater1": "",
            "correctness_rater2": "",
            "faithfulness_rater1": "",
            "faithfulness_rater2": "",
            "notes_rater1": "",
            "notes_rater2": "",
        })

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        w.writeheader()
        w.writerows(blind_rows)

    out_codebook = ROOT / args.out_codebook
    with open(out_codebook, "w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    print(f"Sampled {args.n_questions} questions x {len(conditions)} conditions "
          f"= {len(blind_rows)} blind rating instances.")
    print(f"Wrote {out_csv} (give this to evaluators)")
    print(f"Wrote {out_codebook} (KEEP PRIVATE until rating is complete)")


if __name__ == "__main__":
    main()
