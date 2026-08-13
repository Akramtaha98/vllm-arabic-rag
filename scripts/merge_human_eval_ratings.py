"""
Merges two independently-filled per-annotator rating sheets
(data/human_eval_rater1.csv, data/human_eval_rater2.csv -- each produced by
split_human_eval_for_raters.py and filled in separately by one evaluator)
back into data/human_eval_sample_blind.csv, which is what
scripts/analyze_human_eval.py expects (rater1/rater2 columns side by side).

Usage (after both evaluators send back their filled CSVs):
    python scripts/merge_human_eval_ratings.py

Then:
    python scripts/analyze_human_eval.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_ratings(path):
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:2] == b"PK":
        raise SystemExit(
            f"\n{path} is not a CSV file -- it's a zipped document (likely Apple "
            "Numbers or Excel saved in its native format under a .csv filename).\n"
            "Fix: open it in Numbers/Excel and use File -> Export To -> CSV..., "
            f"then save over {path} again.\n"
        )
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Numbers' "Export To > CSV" sometimes prepends a stray line (e.g. the
    # table/sheet name) before the real header row. Skip down to whichever
    # line actually starts the real header.
    header_idx = None
    for i, line in enumerate(lines[:5]):
        if line.strip().split(",")[0].strip('"') == "item_id":
            header_idx = i
            break
    if header_idx is None:
        raise SystemExit(
            f"\n{path}: couldn't find an 'item_id' header in the first 5 lines.\n"
            f"First line found: {lines[0].strip()!r}\n"
            "Check that this is the exported CSV (File -> Export To -> CSV... "
            "in Numbers/Excel), not a differently-formatted file.\n"
        )
    if header_idx > 0:
        print(f"Note: skipped {header_idx} stray line(s) before the header in {path}")

    rows = list(csv.DictReader(lines[header_idx:]))
    return {r["item_id"]: r for r in rows}


def main():
    master_path = ROOT / "data" / "human_eval_sample_blind.csv"
    r1_path = ROOT / "data" / "human_eval_rater1.csv"
    r2_path = ROOT / "data" / "human_eval_rater2.csv"

    with open(master_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    r1 = load_ratings(r1_path) if r1_path.exists() else {}
    r2 = load_ratings(r2_path) if r2_path.exists() else {}

    filled1 = filled2 = 0
    for row in rows:
        item_id = row["item_id"]
        if item_id in r1:
            row["correctness_rater1"] = r1[item_id].get("correctness_rating", "")
            row["faithfulness_rater1"] = r1[item_id].get("faithfulness_rating", "")
            row["notes_rater1"] = r1[item_id].get("notes", "")
            if r1[item_id].get("correctness_rating", "").strip():
                filled1 += 1
        if item_id in r2:
            row["correctness_rater2"] = r2[item_id].get("correctness_rating", "")
            row["faithfulness_rater2"] = r2[item_id].get("faithfulness_rating", "")
            row["notes_rater2"] = r2[item_id].get("notes", "")
            if r2[item_id].get("correctness_rating", "").strip():
                filled2 += 1

    with open(master_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Merged into {master_path}")
    print(f"  rater1: {filled1}/{len(rows)} rows filled")
    print(f"  rater2: {filled2}/{len(rows)} rows filled")
    print("Now run: python scripts/analyze_human_eval.py")


if __name__ == "__main__":
    main()
