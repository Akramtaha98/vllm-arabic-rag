"""
Locust client for the edge-cloud validation grid. Unlike the original
locustfile.py (which calls vLLM directly), this file's users call the edge
gateway's single /generate endpoint (edge_gateway.py), which does
retrieval + pruning + controller decision + the actual cloud call
internally and returns TTFT/latency/token counts already computed at the
edge in one JSON response. This makes the client itself simple: one POST,
no SSE parsing, since the gateway already streamed from vLLM and buffered
the result.

Environment variables:
    METHOD            "raw" | "naive" | "fixed_lspm" | "kv_aware" | "network_aware"
    PROFILE_NAME       label only, passed through into the log for grouping
                        (the actual network shaping is applied by
                        network_profiles.py against the edge<->cloud link,
                        not by this client)
    MANIFEST_PATH      required for the validation grid (see request_manifest.py)
                        -- unlike locustfile.py, this file does NOT fall back
                        to random queries, since "identical request manifest"
                        is a hard requirement for this specific experiment
                        (Task 1), not an optional nicety.

Usage (single cell, normally driven by run_validation_grid.py):
    METHOD=network_aware PROFILE_NAME=constrained_wireless \
    MANIFEST_PATH=benchmark/request_manifest.json \
    locust -f benchmark/locustfile_edge.py --host http://localhost:9000 \
        --users 10 --spawn-rate 5 --run-time 60s --headless --csv=results/net_grid/cell_x
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

from locust import HttpUser, task, between, events

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METHOD = os.environ["METHOD"]
PROFILE_NAME = os.getenv("PROFILE_NAME", "unknown")
MANIFEST_PATH = os.environ["MANIFEST_PATH"]

with open(MANIFEST_PATH, encoding="utf-8") as _f:
    _MANIFEST = json.load(_f)["entries"]
_MANIFEST_INDEX = itertools.count()


def next_query() -> str:
    idx = next(_MANIFEST_INDEX)
    if idx >= len(_MANIFEST):
        raise IndexError(
            f"Manifest at {MANIFEST_PATH} exhausted after {idx} requests. "
            "Regenerate with a larger --length rather than wrapping -- see request_manifest.py."
        )
    return _MANIFEST[idx]["query"]


class EdgeUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def call_edge(self):
        query = next_query()
        payload = {"query": query, "method": METHOD, "profile_name": PROFILE_NAME}
        name = f"generate[{METHOD}|{PROFILE_NAME}]"

        t0 = time.perf_counter()
        try:
            resp = self.client.post("/generate", json=payload, name=name)
        except Exception as e:
            events.request.fire(
                request_type="CUSTOM", name=f"{name}_error", response_time=0,
                response_length=0, exception=e, context={},
            )
            return
        client_latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            print(f"DEBUG FAILURE: status={resp.status_code} body={resp.text[:300]!r}")
            return
        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"DEBUG FAILURE: non-JSON response body={resp.text[:300]!r}")
            return
        if "error" in data:
            print(f"DEBUG FAILURE: gateway reported error: {data}")
            return

        # Report the gateway's own (more precise, edge-observed) TTFT and
        # cloud-call latency as custom metrics, alongside locust's native
        # client-observed request timer (which also includes the client<->
        # edge hop, normally negligible since they're co-located).
        if data.get("ttft_ms") is not None:
            events.request.fire(
                request_type="CUSTOM", name=f"ttft_ms[{METHOD}|{PROFILE_NAME}]",
                response_time=data["ttft_ms"], response_length=0, exception=None, context={},
            )
        events.request.fire(
            request_type="CUSTOM", name=f"client_latency_ms[{METHOD}|{PROFILE_NAME}]",
            response_time=client_latency_ms, response_length=0, exception=None, context={},
        )
        if data.get("ratio_used") is not None:
            events.request.fire(
                request_type="CUSTOM", name=f"ratio_used[{METHOD}|{PROFILE_NAME}]",
                response_time=data["ratio_used"] * 1000,  # locust wants a number; x1000 keeps units visually distinct in reports, undone in analysis
                response_length=0, exception=None, context={},
            )
