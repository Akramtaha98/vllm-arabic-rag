"""
Precomputes every (method, ratio, query) context the GPU benchmark needs,
once, up front, on CPU, and writes them to a JSON lookup file.

Why this exists: in the original sweep (paper Section 5.8, round 7), LSPM's
cross-encoder scoring ran *inside* the same single-process Locust load
generator that issued requests. Under Locust's cooperative (gevent)
concurrency model, that CPU-bound scoring became a plausible client-side
bottleneck at high simulated concurrency, confounding the between-method
comparison at c >= 10 (see paper Section 5.8, Appendix G). Precomputing
every context ahead of time removes pruning computation from the load
generator entirely for all three methods (raw/lspm/naive) -- locustfile.py
then does a plain dict lookup per request instead of live computation.

Usage (from repo root):
    python benchmark/precompute_contexts.py --out benchmark/precomputed_contexts.json

Fast: 5 sample queries x (1 raw + 3 lspm ratios + 3 naive ratios) = 35
contexts total, dominated by one-time cross-encoder model load.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, split_sentences  # noqa: E402
from middleware.retriever import MOCK_CORPUS  # noqa: E402

SAMPLE_QUERIES = [
    "متى تأسست جامعة الملك سعود؟",
    "ما هي اهتمامات قسم الحاسب في الجامعة؟",
    "كيف يكون الطقس في الرياض خلال الصيف؟",
    "ماذا يوجد في مكتبة الجامعة؟",
    "ما هو مركز أبحاث الذكاء الاصطناعي؟",
]

RATIOS = [0.3, 0.5, 0.7]
PRUNER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def naive_truncate(documents, compression_ratio):
    """Identical logic to locustfile.py's naive_truncate and
    scripts/run_arcd_pilot.py's baseline, kept in sync by hand since this
    is a small, stable function -- see those files for the canonical
    definition this must match."""
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    n = len(all_sentences)
    num_to_keep = max(1, int(round(n * compression_ratio)))
    num_to_keep = min(num_to_keep, n)
    return " ".join(all_sentences[:num_to_keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/precomputed_contexts.json")
    args = ap.parse_args()

    docs = MOCK_CORPUS
    lookup = {}

    # raw: query-independent, one entry reused for every query
    raw_context = " ".join(docs)
    for q in SAMPLE_QUERIES:
        lookup[f"raw|None|{q}"] = raw_context

    # naive: query-independent per ratio (deterministic truncation of the
    # fixed corpus), but keyed by query too for a uniform lookup interface
    for ratio in RATIOS:
        naive_context = naive_truncate(docs, ratio)
        for q in SAMPLE_QUERIES:
            lookup[f"naive|{ratio}|{q}"] = naive_context

    # lspm: genuinely query-dependent (cross-encoder scores relevance to
    # each query), so compute all 3 ratios x 5 queries = 15 combinations
    print("Loading cross-encoder for LSPM precomputation (one-time)...")
    pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME)
    for ratio in RATIOS:
        for q in SAMPLE_QUERIES:
            result = pruner.prune(q, docs, compression_ratio=ratio)
            lookup[f"lspm|{ratio}|{q}"] = result.pruned_text
            print(f"  precomputed lspm r={ratio} q={q[:30]}...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(lookup)} precomputed contexts to {out_path}")


if __name__ == "__main__":
    main()
