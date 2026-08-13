"""
Edge gateway service for the network-aware experiment
(paper/NETWORK_SPECIALIZATION_PLAN.md, Deliverable 3's architecture).

Runs the "edge tier": retrieval, semantic pruning (LSPM/naive/raw), and the
compression-ratio controller (fixed / KV-only / network-aware, Task 5). It
then forwards the resulting prompt to the "cloud tier" (a self-hosted vLLM
server, unchanged from Section 4.6's setup), streams the response back to
the client, and writes one centralized JSONL log line per request
containing every network, server, GPU, and request-level field the
experiment needs (Task 6). Answer-quality scoring (EM/F1/faithfulness,
Task 7) is NOT done inline here -- it is done offline against the logged
query/answer pairs, using the paper's existing ARCD scoring pipeline
(scripts/analyze_arcd_results.py), since the current 5-query mock corpus
used for the *systems* metrics has no ground-truth answers (see module
note at the bottom on the ARCD-pool extension needed before correctness
numbers can be produced from this gateway's traffic).

Deployment: this process is the "edge" half of the topology. Run it on the
machine that will be shaped as the network-constrained side (or, in the
single-rented-GPU first phase, run it on the same box as vLLM but bind it
to a separate network namespace / veth pair so tc netem has a real link to
shape between the two -- see network_profiles.py's docstring). The client
load generator (locustfile_edge.py) talks to THIS service, not directly to
vLLM.

Usage:
    CLOUD_VLLM_URL=http://<cloud-host>:8000 \
    CONTROLLER_MODE=network_aware \
    uvicorn benchmark.edge_gateway:app --host 0.0.0.0 --port 9000

Environment variables:
    CLOUD_VLLM_URL       required. Base URL of the cloud-tier vLLM server.
    CONTROLLER_MODE       "fixed" | "kv_only" | "network_aware" (default "fixed")
    FIXED_RATIO            used when CONTROLLER_MODE=fixed (default 0.5)
    NETWORK_PROBE_TARGET   host to ping for the RTT input (default: CLOUD_VLLM_URL's host)
    NETWORK_PROBE_INTERVAL_S  seconds between background RTT/bandwidth probes (default 2.0)
    REQUEST_LOG_PATH        JSONL output path (default results/edge_gateway_log.jsonl)
    PRECOMPUTED_CONTEXTS_PATH  optional, same lookup-table mechanism as locustfile.py
    NETWORK_EMULATION_MODE  "none" (default) | "application". When "application", every
                             /generate request has delay/jitter/loss/bandwidth-throttling
                             injected in software, keyed by the per-request `profile_name`
                             field, using benchmark/network_profiles.py's PROFILES table.
                             This is a fallback for environments without CAP_NET_ADMIN
                             (confirmed necessary on the RunPod pod this was deployed on --
                             see network_profiles.py's module docstring for why, and for the
                             explicit disclosure requirement this implies for the paper).
                             Do NOT set this if real tc-netem shaping is being applied
                             upstream of this process -- the two would stack.

Fix history (2026-08-14, after analyzing the first 90-cell validation grid):
1. decide_ratio()'s kv_aware and network_aware branches made blocking
   synchronous HTTP calls to /metrics from inside this async handler,
   freezing the single event loop for every concurrent request during that
   call. kv_aware made THREE such calls per request (one wasted, two more
   inside the KV-only controller itself); network_aware made one. This
   showed up in the grid as artificially inflated TTFT/latency for exactly
   those two methods at concurrency=50 (kv_aware ~560-680ms TTFT vs
   ~75-165ms for raw/naive/fixed_lspm) -- an instrumentation artifact, not
   real controller cost. Fixed by making the metrics fetch async
   (httpx.AsyncClient here; asyncio.to_thread(...) around the KV-only
   controller's synchronous `requests` call in network_controller.py) and
   by no longer fetching kv_pct at all for kv_aware, which never used the
   shared fetch's result in the first place.
2. Under NETWORK_EMULATION_MODE=application, the network-aware
   controller's rtt/bandwidth inputs came from NetworkProbe, which
   pings/measures the REAL unshaped underlying link -- it cannot see the
   software-injected delay/throttle applied elsewhere in this file, so the
   controller was effectively blind to which profile was "active." In the
   first grid this was confirmed: network_aware's mean selected ratio was
   statistically identical (0.587) under both edge_lan and
   constrained_wireless. Fixed: decide_ratio() now feeds the controller the
   profile's own configured delay_ms/rate_mbit directly when running under
   application-level emulation (ground truth is known in that mode), and
   falls back to the real NetworkProbe only when NETWORK_EMULATION_MODE is
   "none" (i.e. real tc netem shaping a real link, where a live measurement
   is the only way to know the actual condition).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from middleware.pruning import SemanticPruner, split_sentences  # noqa: E402
from middleware.retriever import MOCK_CORPUS  # noqa: E402
from benchmark.network_controller import (  # noqa: E402
    NetworkAwareController, KVOnlyDecision, NetworkAwareDecision, snap_to_validated_ratio,
)
from benchmark import network_profiles  # noqa: E402

CLOUD_VLLM_URL = os.environ["CLOUD_VLLM_URL"].rstrip("/")
CONTROLLER_MODE = os.getenv("CONTROLLER_MODE", "fixed")  # fixed | kv_only | network_aware
FIXED_RATIO = float(os.getenv("FIXED_RATIO", "0.5"))
NETWORK_PROBE_TARGET = os.getenv("NETWORK_PROBE_TARGET") or urlparse(CLOUD_VLLM_URL).hostname
NETWORK_PROBE_INTERVAL_S = float(os.getenv("NETWORK_PROBE_INTERVAL_S", "2.0"))
REQUEST_LOG_PATH = os.getenv("REQUEST_LOG_PATH", str(ROOT / "results" / "edge_gateway_log.jsonl"))
PRECOMPUTED_CONTEXTS_PATH = os.getenv("PRECOMPUTED_CONTEXTS_PATH")
PRUNER_MODEL_NAME = os.getenv("PRUNER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
NETWORK_EMULATION_MODE = os.getenv("NETWORK_EMULATION_MODE", "none")  # "none" | "application"
if NETWORK_EMULATION_MODE not in ("none", "application"):
    raise ValueError(f"NETWORK_EMULATION_MODE must be 'none' or 'application', got {NETWORK_EMULATION_MODE!r}")

Path(REQUEST_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
_log_lock = threading.Lock()

_PRECOMPUTED = None
if PRECOMPUTED_CONTEXTS_PATH:
    with open(PRECOMPUTED_CONTEXTS_PATH, encoding="utf-8") as _f:
        _PRECOMPUTED = json.load(_f)

_pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME) if _PRECOMPUTED is None else None

app = FastAPI(title="LSPM Edge Gateway")


# --------------------------------------------------------------------------
# In-flight request tracking (feeds "concurrency" and "queue_length" inputs
# to the controller -- both measured locally at the edge, not estimated)
# --------------------------------------------------------------------------
class _RequestTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.in_flight = 0
        self.queued = 0  # requests accepted but not yet forwarded to cloud

    def enter_queue(self):
        with self._lock:
            self.queued += 1

    def leave_queue_enter_inflight(self):
        with self._lock:
            self.queued = max(0, self.queued - 1)
            self.in_flight += 1

    def leave_inflight(self):
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    def snapshot(self):
        with self._lock:
            return self.in_flight, self.queued


tracker = _RequestTracker()


# --------------------------------------------------------------------------
# Background network-condition probe: periodically pings the cloud tier and
# does a small timed transfer to produce passive RTT/bandwidth estimates
# for the controller. Explicitly passive/estimated -- documented as such in
# every log line (see `network_probe_method` field below) rather than
# presented as a precise dedicated-probe measurement.
# --------------------------------------------------------------------------
class NetworkProbe:
    def __init__(self, target: str, interval_s: float, window: int = 10):
        self.target = target
        self.interval_s = interval_s
        self._rtt_window = deque(maxlen=window)
        self._bw_window = deque(maxlen=window)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _ping_once(self):
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", self.target],
                capture_output=True, text=True, timeout=5,
            )
            m = re.search(r"time=([\d.]+)\s*ms", result.stdout)
            return float(m.group(1)) if m else None
        except Exception:
            return None

    def record_transfer(self, n_bytes: int, elapsed_s: float):
        """Called after every real request completes -- a passive bandwidth
        sample from actual application traffic, generally more representative
        than a synthetic probe transfer, and doesn't consume extra bandwidth
        on a constrained link just to measure it."""
        if elapsed_s > 0 and n_bytes > 0:
            mbps = (n_bytes * 8 / 1_000_000) / elapsed_s
            with self._lock:
                self._bw_window.append(mbps)

    def _loop(self):
        while not self._stop.is_set():
            rtt = self._ping_once()
            if rtt is not None:
                with self._lock:
                    self._rtt_window.append(rtt)
            time.sleep(self.interval_s)

    def current(self):
        with self._lock:
            rtt = statistics.mean(self._rtt_window) if self._rtt_window else None
            bw = statistics.mean(self._bw_window) if self._bw_window else None
        return rtt, bw


probe = NetworkProbe(NETWORK_PROBE_TARGET, NETWORK_PROBE_INTERVAL_S) if NETWORK_PROBE_TARGET else None


async def _fetch_kv_cache_usage_pct() -> float | None:
    """Async (httpx.AsyncClient, not the old blocking httpx.get) so this
    does not freeze the event loop for every other concurrent /generate
    request while it waits on the network round trip -- see the module
    docstring's history note on why this changed."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(CLOUD_VLLM_URL + "/metrics")
            r.raise_for_status()
    except Exception:
        return None
    m = re.search(r"^vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
                   r.text, re.MULTILINE)
    return float(m.group(2)) * 100 if m else None  # fraction -> percent, matches controller's pct convention


_network_aware_controller = NetworkAwareController()


def naive_truncate(documents, compression_ratio):
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    n = len(all_sentences)
    num_to_keep = max(1, int(round(n * compression_ratio)))
    return " ".join(all_sentences[:min(num_to_keep, n)])


def build_context(method: str, ratio: float, query: str) -> str:
    if _PRECOMPUTED is not None and method in ("raw", "lspm", "naive"):
        ratio_key = ratio if method != "raw" else None
        lookup_key = f"{method}|{ratio_key}|{query}"
        if lookup_key in _PRECOMPUTED:
            return _PRECOMPUTED[lookup_key]
    docs = MOCK_CORPUS
    if method == "raw":
        return " ".join(docs)
    if method == "naive":
        return naive_truncate(docs, ratio)
    # lspm (fixed, kv_aware, network_aware all use the same pruning
    # mechanism -- they differ only in how `ratio` was chosen upstream)
    result = _pruner.prune(query, docs, compression_ratio=ratio)
    return result.pruned_text


def _emulation_active_for(profile_name: str) -> bool:
    return NETWORK_EMULATION_MODE == "application" and profile_name in network_profiles.PROFILES


async def decide_ratio(method: str, query: str, profile_name: str) -> tuple[float, NetworkAwareDecision | None]:
    """Returns (ratio, decision_record_or_None). decision_record is None
    for "raw" (no ratio applies) and "fixed" (no controller decision to log,
    the ratio is a constant), populated for kv_aware/network_aware so every
    adaptive decision is auditable, per Task 5's explicit logging
    requirement ("Log every input and resulting pruning decision").

    Async: kv_aware and network_aware both may need a real /metrics fetch,
    and that fetch must not block the event loop for other concurrent
    requests -- see this module's "Fix history" note above."""
    if method == "raw":
        return None, None
    if method in ("naive", "fixed_lspm"):
        return FIXED_RATIO, None
    in_flight, queued = tracker.snapshot()
    if method == "kv_aware":
        # No shared kv_pct fetch here -- KVOnlyDecision does its own single
        # fetch internally. The old code fetched kv_pct here too and threw
        # it away for this branch, i.e. a second wasted blocking call.
        d = await KVOnlyDecision(metrics_url=CLOUD_VLLM_URL + "/metrics").decide(fallback_ratio=FIXED_RATIO)
        return d.ratio_selected, d
    if method == "network_aware":
        if _emulation_active_for(profile_name):
            # Real probes can't see software-injected delay/throttle (see
            # Fix history note 2) -- use the profile's own known parameters
            # instead of a measurement that would always report the fast
            # unshaped link.
            p = network_profiles.PROFILES[profile_name]
            rtt, bw = float(p["delay_ms"]), float(p["rate_mbit"])
        else:
            rtt, bw = probe.current() if probe else (None, None)
        kv_pct = await _fetch_kv_cache_usage_pct()
        d = _network_aware_controller.decide(
            bandwidth_mbps=bw, rtt_ms=rtt, queue_length=float(queued),
            concurrency=float(in_flight), kv_cache_usage_pct=kv_pct,
        )
        return d.ratio_selected, d
    raise ValueError(f"Unknown method {method!r}")


def _log(record: dict):
    with _log_lock:
        with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.on_event("startup")
def _startup():
    if probe is not None:
        probe.start()
    print(f"Edge gateway up. cloud={CLOUD_VLLM_URL} controller_mode={CONTROLLER_MODE} "
          f"probe_target={NETWORK_PROBE_TARGET} log={REQUEST_LOG_PATH} "
          f"network_emulation_mode={NETWORK_EMULATION_MODE}")
    if NETWORK_EMULATION_MODE == "application":
        print("  NOTE: application-level network emulation is ACTIVE. Delay/jitter/loss/"
              "bandwidth-throttle are injected in software per request, keyed by profile_name. "
              "This is NOT kernel-level tc netem shaping -- see network_profiles.py docstring.")


@app.get("/health")
def health():
    return {"status": "ok", "cloud_vllm_url": CLOUD_VLLM_URL, "controller_mode": CONTROLLER_MODE,
             "network_emulation_mode": NETWORK_EMULATION_MODE}


SYSTEM_PROMPT = "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك وبشكل مختصر."


@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    request_id = body.get("request_id") or str(uuid.uuid4())
    query = body["query"]
    method = body.get("method", CONTROLLER_MODE if CONTROLLER_MODE != "fixed" else "fixed_lspm")
    profile_name = body.get("profile_name", "unknown")

    tracker.enter_queue()
    t_recv = time.time()

    ratio, decision = await decide_ratio(method, query, profile_name)
    context = build_context(method, ratio if ratio is not None else FIXED_RATIO, query)

    tracker.leave_queue_enter_inflight()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"السياق: {context}\n\nالسؤال: {query}"},
        ],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request_bytes = len(json.dumps(payload).encode("utf-8"))

    # ----------------------------------------------------------------
    # Application-level network emulation (fallback for no CAP_NET_ADMIN --
    # see network_profiles.py module docstring). Loss is checked first: a
    # "lost" request never reaches the cloud tier at all. t_send is stamped
    # BEFORE the uplink sleep, so the sleep is naturally folded into
    # ttft_ms and total_latency_ms below (both measured relative to
    # t_send), the same way real network delay would show up to a client.
    # ----------------------------------------------------------------
    t_send = time.time()
    uplink_delay_s = 0.0
    emulation_active = _emulation_active_for(profile_name)
    if emulation_active and network_profiles.should_drop_request(profile_name):
        tracker.leave_inflight()
        _log({
            "request_id": request_id, "epoch": t_recv, "method": method, "ratio": ratio,
            "profile_name": profile_name, "status": "emulated_loss",
            "network_emulation_mode": NETWORK_EMULATION_MODE,
        })
        return {"error": "emulated_network_loss", "profile_name": profile_name}
    if emulation_active:
        uplink_delay_s = network_profiles.sample_uplink_delay_s(profile_name)
        await asyncio.sleep(uplink_delay_s)

    ttft_ms = None
    usage = None
    response_bytes = 0
    chunks = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", CLOUD_VLLM_URL + "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    tracker.leave_inflight()
                    _log({
                        "request_id": request_id, "epoch": t_recv, "method": method, "ratio": ratio,
                        "profile_name": profile_name, "status": "http_error",
                        "http_status": resp.status_code, "body_snippet": body_text[:300].decode(errors="replace"),
                        "network_emulation_mode": NETWORK_EMULATION_MODE,
                    })
                    return {"error": "cloud_http_error", "status": resp.status_code}
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    response_bytes += len(line.encode("utf-8"))
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    line = line.strip()
                    if line == "[DONE]":
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            if ttft_ms is None:
                                ttft_ms = (time.time() - t_send) * 1000
                            chunks.append(delta)
    except Exception as e:
        tracker.leave_inflight()
        _log({
            "request_id": request_id, "epoch": t_recv, "method": method, "ratio": ratio,
            "profile_name": profile_name, "status": "exception", "error": str(e),
            "network_emulation_mode": NETWORK_EMULATION_MODE,
        })
        return {"error": "exception", "detail": str(e)}

    t_done = time.time()
    tracker.leave_inflight()
    answer_text = "".join(chunks)
    probe.record_transfer(response_bytes, t_done - t_send) if probe else None

    # Downlink delay + bandwidth throttle, applied after the full response
    # is buffered (this gateway returns one JSON response, not a live
    # stream to the client -- see module docstring -- so "downlink" here
    # means the emulated cloud->edge leg, not edge->client). t_done is
    # re-stamped after these sleeps so total_latency_ms reflects them;
    # ttft_ms deliberately is NOT adjusted here since first-token time
    # in a real shaped network would be governed by uplink + propagation,
    # not by the full-response bandwidth cap.
    downlink_delay_s = 0.0
    bandwidth_throttle_s = 0.0
    if emulation_active:
        downlink_delay_s = network_profiles.sample_downlink_delay_s(profile_name)
        bandwidth_throttle_s = network_profiles.bandwidth_throttle_sleep_s(
            profile_name, response_bytes, t_done - t_send)
        await asyncio.sleep(downlink_delay_s + bandwidth_throttle_s)
        t_done = time.time()

    total_latency_ms = (t_done - t_send) * 1000

    in_flight_at_decision, queued_at_decision = tracker.snapshot()  # post-hoc, approximate; see note below

    applied_emulation = None
    if emulation_active:
        applied_emulation = {
            "mode": "application", "profile_name": profile_name,
            "uplink_delay_s": uplink_delay_s, "downlink_delay_s": downlink_delay_s,
            "bandwidth_throttle_s": bandwidth_throttle_s,
        }

    record = {
        "request_id": request_id, "epoch": t_recv, "profile_name": profile_name,
        "method": method, "ratio_requested": ratio, "controller_mode": CONTROLLER_MODE,
        "controller_decision": decision.to_json_dict() if decision else None,
        "query": query, "answer": answer_text,
        "ttft_ms": ttft_ms, "total_latency_ms": total_latency_ms,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "token_count_source": "usage_field" if usage else "unavailable",
        "request_bytes": request_bytes, "response_bytes": response_bytes,
        "context_char_count": len(context), "status": "ok",
        "network_emulation_mode": NETWORK_EMULATION_MODE,
        "applied_emulation": applied_emulation,
    }
    _log(record)
    return {
        "request_id": request_id, "answer": answer_text, "ratio_used": ratio,
        "ttft_ms": ttft_ms, "total_latency_ms": total_latency_ms,
    }


# --------------------------------------------------------------------------
# Scope note on answer-quality measurement (Task 7): this gateway logs the
# raw (query, answer) pair for every request, but the 5-query mock corpus
# used here (shared with the existing v1/v2 systems benchmark, for
# continuity with Section 5.8's protocol) has no ground-truth answers, so
# EM/F1/faithfulness cannot be computed from validation-grid traffic as-is.
# Producing real correctness numbers requires pointing MANIFEST_PATH /
# PRECOMPUTED_CONTEXTS_PATH at the ARCD-grounded pool (Section 4.4's
# existing 140-question set) instead of the 5-query mock set, and running
# scripts/analyze_arcd_results.py-style scoring against this gateway's
# logged answers. This is deliberately not done inside this file (mixing
# quality- and systems-scale queries in one manifest would confound the
# systems measurement's controlled concurrency behavior with the smaller,
# fixed-corpus quality measurement) -- it is the next script to write once
# the validation grid itself is confirmed working end-to-end.
