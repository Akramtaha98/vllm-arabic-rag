"""
Controlled warm-up phase (Task 1: "controlled warm-up"), run once before
each timed measurement cell in run_validation_grid.py.

Fires a fixed number of sequential, unmeasured requests directly at the
target (vLLM server or edge gateway) before the timed locust run begins, so
the first-N-requests-after-cold-start effect (model/CUDA graph warm-up,
connection-pool establishment, initial scheduler-queue fill) does not land
inside the measurement window. This is deliberately a separate, simple
script rather than a "don't count the first N locust requests" flag inside
locustfile.py, because locust's concurrent user model does not give a
clean way to guarantee the very first wave of concurrent requests is
excluded post-hoc -- running warm-up strictly before spawning any locust
users is the only way to make "warm-up" and "measurement" non-overlapping
by construction.

Usage:
    # Direct against vLLM (original v1/v2 harness topology):
    python benchmark/warmup.py --host http://localhost:8000 --n 10

    # Against the edge gateway (network-aware experiment topology) --
    # --gateway-mode posts to /generate with a method, instead of
    # /v1/chat/completions with a raw chat payload:
    python benchmark/warmup.py --host http://localhost:9000 --n 10 \
        --gateway-mode --method raw
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.sample_queries import SAMPLE_QUERIES, SYSTEM_PROMPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--n", type=int, default=10, help="Number of sequential warm-up requests")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=64, help="Short completions -- warm-up only, "
                     "not measuring generation quality or full-length latency")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--gateway-mode", action="store_true",
                     help="Warm up the edge gateway's /generate endpoint instead of calling "
                          "vLLM's /v1/chat/completions directly. Use this whenever --host points "
                          "at edge_gateway.py, not at vLLM.")
    ap.add_argument("--method", default="raw",
                     help="method field to send in --gateway-mode (raw/naive/fixed_lspm/kv_aware/network_aware)")
    args = ap.parse_args()

    if args.gateway_mode:
        url = args.host.rstrip("/") + "/generate"
    else:
        url = args.host.rstrip("/") + "/v1/chat/completions"

    ok, failed = 0, 0
    for i in range(args.n):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        if args.gateway_mode:
            payload = {"query": query, "method": args.method, "profile_name": "warmup"}
        else:
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"السؤال: {query}"},
                ],
                "temperature": 0.3,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=args.timeout)
            r.raise_for_status()
            if args.gateway_mode and isinstance(r.json(), dict) and "error" in r.json():
                # The gateway returns HTTP 200 with an {"error": ...} body on
                # internal failures (e.g. cloud unreachable) rather than an
                # HTTP error status, so raise_for_status() alone would not
                # catch this -- check explicitly rather than counting a
                # gateway-side failure as a successful warm-up.
                raise RuntimeError(f"gateway returned error body: {r.json()}")
            elapsed = (time.perf_counter() - t0) * 1000
            ok += 1
            print(f"  warm-up {i + 1}/{args.n}: ok, {elapsed:.0f}ms")
        except Exception as e:
            failed += 1
            print(f"  warm-up {i + 1}/{args.n}: FAILED ({e})")

    print(f"Warm-up complete: {ok} ok, {failed} failed, target={args.host}")
    if failed == args.n:
        print("ERROR: every warm-up request failed -- the server is likely not reachable/ready. "
              "Aborting rather than proceeding to a measurement run against an unverified server.")
        sys.exit(1)


if __name__ == "__main__":
    main()
