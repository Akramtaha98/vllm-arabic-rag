"""Arabic-normalized EM / token-F1 scorer for data/arcd_results.jsonl.

Reimplements the SQuAD-style scorer used by scripts/analyze_arcd_results.py
(eval/arabic_em_f1.py is not present in this working copy) and is validated
against the released results/arcd_stats_summary.csv before being used for any
new computation.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (this file lives in scripts/)

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(s):
    s = DIACRITICS.sub("", s)
    s = re.sub("[أإآٱ]", "ا", s)  # alef variants -> alef
    s = PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    words = [w[2:] if w.startswith("ال") and len(w) > 2 else w for w in words]
    return " ".join(words)


def token_f1(pred, gold):
    pt, gt = pred.split(), gold.split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def score(pred, golds):
    p = normalize(pred)
    gs = [normalize(g) for g in golds]
    return {"em": max(float(p == g) for g in gs),
            "f1": max(token_f1(p, g) for g in gs)}


def load():
    recs = [json.loads(l) for l in open(ROOT / "data/arcd_results.jsonl", encoding="utf-8") if l.strip()]
    return recs


if __name__ == "__main__":
    recs = load()
    cells = {}
    for r in recs:
        cells.setdefault((r["method"], r["ratio"]), []).append(score(r["answer"], r["gold_answers"]))
    for k in sorted(cells):
        v = cells[k]
        print(k, len(v),
              "EM=%.4f" % (sum(x["em"] for x in v) / len(v)),
              "F1=%.4f" % (sum(x["f1"] for x in v) / len(v)))
