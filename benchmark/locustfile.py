"""
Locust load test comparing context conditions against a real, self-hosted
vLLM server: raw (unpruned), lspm (semantic cross-encoder pruning at
COMPRESSION_RATIO), and naive (length-matched truncation baseline).

Measures, per request: TTFT (time-to-first-token, via streaming), total
end-to-end latency, and completion tokens/sec. Concurrency is set by
locust's --users flag; run_full_sweep.py drives the full ratio x
concurrency grid and scrapes vLLM's /metrics endpoint for KV-cache
occupancy in parallel.

--- Repair pass (network-aware experiment, Task 1) ---

1. **Real prompt/completion token counts.** The original version only
   estimated completion tokens by counting streamed SSE chunks (a rough
   proxy, since one chunk is not guaranteed to be exactly one token for
   every tokenizer/decoding path). This version requests
   `stream_options: {"include_usage": true}`, which OpenAI-compatible vLLM
   servers (>=0.5.x) send as a final SSE chunk containing an exact `usage`
   object (`prompt_tokens`, `completion_tokens`, `total_tokens`). If the
   server's vLLM version does not support this option, the field is simply
   absent from the final chunk and this script falls back to the old
   chunk-count estimate, explicitly flagging `token_count_source` as
   `"estimated"` rather than silently presenting an estimate as exact.

2. **Manifest-driven, identical request order.** If `MANIFEST_PATH` is
   set (see request_manifest.py), each request pulls the next query from
   a shared, seeded, pre-generated sequence instead of `random.choice`, so
   every method/profile/concurrency/repeat cell that reads the same
   manifest sees the identical query at the identical request position.
   Falls back to the original per-request `random.choice` if unset, so
   existing v1/v2 sweep usage (run_full_sweep.py) is unaffected.

3. **Structured per-request JSONL log.** If `REQUEST_LOG_PATH` is set,
   every request appends one JSON line with every field needed to later
   join this log against the edge gateway's network-condition log and the
   metrics_scraper's KV-cache log by request_id and epoch timestamp.

Usage (single cell, manual, backward-compatible with the original v1/v2
usage -- MANIFEST_PATH/REQUEST_LOG_PATH are both optional):
    RAG_MODE=raw locust -f benchmark/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 5 --run-time 90s --headless --csv=results/raw_c50

Normally you don't invoke this directly -- see run_full_sweep.py (original
single-tier sweep) or run_validation_grid.py (network-aware edge-cloud
validation grid), which drive the full grid automatically.
"""

import itertools
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from locust import HttpUser, task, between, events

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner, split_sentences  # noqa: E402
from middleware.retriever import MOCK_CORPUS  # noqa: E402

RAG_MODE = os.getenv("RAG_MODE", "raw")  # "raw" | "lspm" | "naive"
COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.5"))
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
PRUNER_MODEL_NAME = os.getenv("PRUNER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# If set (run_full_sweep.py sets this by default -- see precompute_contexts.py
# and paper Section 5.8/8), every context is a plain dict lookup instead of
# live computation inside this process. This exists specifically to remove
# LSPM's cross-encoder scoring from the Locust load-generator process, which
# the paper identifies as a likely client-side bottleneck confounding the
# between-method comparison at higher concurrency in the original run.
PRECOMPUTED_CONTEXTS_PATH = os.getenv("PRECOMPUTED_CONTEXTS_PATH")
_PRECOMPUTED = None
if PRECOMPUTED_CONTEXTS_PATH:
    with open(PRECOMPUTED_CONTEXTS_PATH, encoding="utf-8") as _f:
        _PRECOMPUTED = json.load(_f)

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

from benchmark.sample_queries import SAMPLE_QUERIES, SYSTEM_PROMPT  # noqa: E402

# --------------------------------------------------------------------------
# Manifest-driven query order (repair item 2)
# --------------------------------------------------------------------------
MANIFEST_PATH = os.getenv("MANIFEST_PATH")
_MANIFEST = None
_MANIFEST_INDEX = itertools.count()
if MANIFEST_PATH:
    with open(MANIFEST_PATH, encoding="utf-8") as _f:
        _MANIFEST = json.load(_f)["entries"]


def next_query() -> str:
    """Returns the next query from the shared manifest (identical, ordered
    sequence across every cell reading this file) if MANIFEST_PATH is set,
    otherwise falls back to the original per-request random.choice. Uses
    itertools.count(), safe here because locust/gevent is cooperatively
    scheduled (no OS-thread preemption between the increment and the read),
    not because of any lock -- documented so this isn't mistaken for
    thread-safe under a real multi-threaded executor."""
    if _MANIFEST is not None:
        idx = next(_MANIFEST_INDEX)
        if idx >= len(_MANIFEST):
            raise IndexError(
                f"Manifest at {MANIFEST_PATH} exhausted after {idx} requests "
                f"(manifest length {len(_MANIFEST)}). Regenerate with a larger "
                "--length rather than silently wrapping/reusing -- wrapping "
                "would break the 'identical manifest across cells' guarantee "
                "for any cell that runs longer than this one did."
            )
        return _MANIFEST[idx]["query"]
    import random
    return random.choice(SAMPLE_QUERIES)


# --------------------------------------------------------------------------
# Structured per-request JSONL log (repair item 3)
# --------------------------------------------------------------------------
REQUEST_LOG_PATH = os.getenv("REQUEST_LOG_PATH")
_log_lock = threading.Lock()
_log_file = open(REQUEST_LOG_PATH, "a", encoding="utf-8") if REQUEST_LOG_PATH else None


def _log_request(record: dict):
    if _log_file is None:
        return
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        _log_file.write(line + "\n")
        _log_file.flush()


# Only load the cross-encoder in this process if we're NOT using precomputed
# contexts -- this is the whole point of the precomputed-lookup path.
_pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME) if (RAG_MODE == "lspm" and _PRECOMPUTED is None) else None


def naive_truncate(documents, compression_ratio):
    """Same length-matched truncation baseline used in scripts/run_arcd_pilot.py
    (first-N sentences by character order, N chosen to match LSPM's kept
    sentence count at the same ratio) -- kept identical here so the
    systems-level benchmark uses the same baseline definition as the
    paper's accuracy results."""
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    n = len(all_sentences)
    num_to_keep = max(1, int(round(n * compression_ratio)))
    num_to_keep = min(num_to_keep, n)
    return " ".join(all_sentences[:num_to_keep])


def build_context(query: str) -> str:
    if _PRECOMPUTED is not None:
        ratio_key = COMPRESSION_RATIO if RAG_MODE != "raw" else None
        lookup_key = f"{RAG_MODE}|{ratio_key}|{query}"
        try:
            return _PRECOMPUTED[lookup_key]
        except KeyError:
            raise KeyError(
                f"No precomputed context for {lookup_key!r} in {PRECOMPUTED_CONTEXTS_PATH}. "
                "Re-run benchmark/precompute_contexts.py, or unset PRECOMPUTED_CONTEXTS_PATH "
                "to fall back to live computation."
            )
    docs = MOCK_CORPUS
    if RAG_MODE == "lspm":
        result = _pruner.prune(query, docs, compression_ratio=COMPRESSION_RATIO)
        return result.pruned_text
    if RAG_MODE == "naive":
        return naive_truncate(docs, COMPRESSION_RATIO)
    return " ".join(docs)  # raw


class VLLMUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def chat_completion(self):
        request_id = str(uuid.uuid4())
        query = next_query()
        context = build_context(query)
        prompt_text = f"السياق: {context}\n\nالسؤال: {query}"
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request_bytes = len(json.dumps(payload).encode("utf-8"))

        t0_epoch = time.time()
        t0 = time.perf_counter()
        ttft_ms = None
        completion_tokens_est = 0
        usage = None  # filled from the server's final SSE chunk if it sends one
        response_bytes = 0
        name = f"chat[{RAG_MODE}_r{COMPRESSION_RATIO}]" if RAG_MODE != "raw" else "chat[raw]"

        try:
            with self.client.post(
                "/v1/chat/completions", json=payload, catch_response=True, stream=True, name=name
            ) as resp:
                if resp.status_code != 200:
                    print(f"DEBUG FAILURE: url={resp.url!r} status={resp.status_code} body={resp.text[:300]!r}")
                    resp.failure(f"status={resp.status_code}")
                    _log_request({
                        "request_id": request_id, "epoch": t0_epoch, "rag_mode": RAG_MODE,
                        "ratio": COMPRESSION_RATIO, "query": query, "status": "http_error",
                        "http_status": resp.status_code,
                    })
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    response_bytes += len(line)
                    line = line.decode("utf-8")
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
                        usage = obj["usage"]  # final chunk under stream_options.include_usage
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        completion_tokens_est += 1  # fallback proxy if usage is absent

                total_latency_ms = (time.perf_counter() - t0) * 1000
                resp.success()
        except Exception as e:
            events.request.fire(
                request_type="CUSTOM", name=f"{name}_error", response_time=0,
                response_length=0, exception=e, context={},
            )
            _log_request({
                "request_id": request_id, "epoch": t0_epoch, "rag_mode": RAG_MODE,
                "ratio": COMPRESSION_RATIO, "query": query, "status": "exception",
                "error": str(e),
            })
            return

        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            token_count_source = "usage_field"
        else:
            prompt_tokens = None
            completion_tokens = completion_tokens_est
            token_count_source = "estimated_chunk_count"

        if ttft_ms is not None:
            events.request.fire(
                request_type="CUSTOM", name=f"ttft_ms[{RAG_MODE}_r{COMPRESSION_RATIO}]" if RAG_MODE != "raw" else "ttft_ms[raw]",
                response_time=ttft_ms, response_length=0, exception=None, context={},
            )
        tail_s = max((total_latency_ms - (ttft_ms or 0)), 1e-6) / 1000
        tok_count_for_rate = completion_tokens if completion_tokens else completion_tokens_est
        tokens_per_sec = tok_count_for_rate / tail_s if tok_count_for_rate else 0
        events.request.fire(
            request_type="CUSTOM", name=f"tokens_per_sec[{RAG_MODE}_r{COMPRESSION_RATIO}]" if RAG_MODE != "raw" else "tokens_per_sec[raw]",
            response_time=tokens_per_sec, response_length=tok_count_for_rate, exception=None, context={},
        )

        _log_request({
            "request_id": request_id, "epoch": t0_epoch, "rag_mode": RAG_MODE,
            "ratio": COMPRESSION_RATIO, "query": query, "status": "ok",
            "ttft_ms": ttft_ms, "total_latency_ms": total_latency_ms,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "token_count_source": token_count_source,
            "request_bytes": request_bytes, "response_bytes": response_bytes,
        })


@events.quitting.add_listener
def _close_log(environment, **kwargs):
    if _log_file is not None:
        _log_file.close()
