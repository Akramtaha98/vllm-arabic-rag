"""
Builds a fixed, seeded, identical request manifest -- an ordered sequence of
query indices -- shared across every (method, network-profile, concurrency,
repeat) cell in the network-aware validation grid (run_validation_grid.py).

Why this exists: the original v1/v2 harness (locustfile.py) picks a query at
random per request (random.choice(SAMPLE_QUERIES)), which is fine for a
same-machine sweep but is not a controlled manifest -- two cells being
compared could, by chance, see different query mixes. For the network-aware
experiment (client -> edge -> cloud, multiple network profiles), the paper's
own Task 1 instruction requires "identical request manifests" so that any
difference between methods/profiles/concurrency levels is attributable to
the condition under test, not to sampling variation in which queries were
asked.

The manifest is a plain ordered list, generated once with a fixed seed and
saved to JSON. Every consumer (edge_gateway.py, the locust client hitting
it) reads the same file and advances through it with a shared, monotonic,
gevent-safe counter (see edge_gateway.py's `next_manifest_query()`), so
request #k is always the same query across every cell, for any run length.
The manifest is intentionally longer than any single cell's expected
request count so it never wraps mid-cell at the concurrency levels used in
this project (1/10/50 users x <=180s at ~1 req/1-2s/user).

Usage:
    python benchmark/request_manifest.py --out benchmark/request_manifest.json \
        --length 2000 --seed 20260813

Design note on source pool: uses the same 5-query SAMPLE_QUERIES pool as the
existing GPU benchmark (locustfile.py), for continuity with the paper's
Section 5.8 protocol description ("the same small set of 5 sample Arabic
questions"). If/when the network-aware experiments move to the larger ARCD
pool for closer alignment with the accuracy results (Section 5.4), pass
--source arcd (not yet implemented here -- flagged as a deliberate scope
limit, not a silent substitution).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.sample_queries import SAMPLE_QUERIES  # noqa: E402


def build_manifest(length: int, seed: int, pool):
    rng = random.Random(seed)
    return [{"position": i, "query_index": rng.randrange(len(pool)), "query": None} for i in range(length)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmark" / "request_manifest.json"))
    ap.add_argument("--length", type=int, default=2000,
                     help="Total manifest entries. Must exceed the max requests any single "
                          "cell in the validation grid could issue (c=50 users x ~180s run "
                          "x up to ~2 req/s/user is a generous upper bound of ~18000; default "
                          "2000 is sized for the c<=50, <=180s validation grid specifically -- "
                          "raise this before using the manifest for a larger/longer sweep, "
                          "the consumer will raise an error rather than silently wrap")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    manifest = build_manifest(args.length, args.seed, SAMPLE_QUERIES)
    for entry in manifest:
        entry["query"] = SAMPLE_QUERIES[entry["query_index"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "pool_size": len(SAMPLE_QUERIES), "length": args.length,
                   "entries": manifest}, f, ensure_ascii=False, indent=1)

    # Sanity check, printed so it's visible in the run log: distribution
    # should be roughly uniform over the pool, not skewed by a seeding bug.
    counts = [0] * len(SAMPLE_QUERIES)
    for entry in manifest:
        counts[entry["query_index"]] += 1
    print(f"Wrote {len(manifest)} entries to {out_path} (seed={args.seed}, pool={len(SAMPLE_QUERIES)} queries)")
    print(f"Per-query counts: {counts} (expect roughly uniform, ~{args.length / len(SAMPLE_QUERIES):.0f} each)")


if __name__ == "__main__":
    main()
