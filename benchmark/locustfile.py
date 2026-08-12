"""
Locust load test comparing three context conditions against a real,
self-hosted vLLM server:
  A) raw    - unpruned context (all documents, full text)
  B) lspm   - LSPM semantic cross-encoder pruning at COMPRESSION_RATIO
  C) naive  - naive length-matched (first-N-sentences) truncation at
              COMPRESSION_RATIO, the paper's baseline comparator

Measures, per request: TTFT (time-to-first-token, via streaming), total
end-to-end latency, and completion tokens/sec. Concurrency is set by
locust's --users flag; run_full_sweep.py drives the full ratio x
concurrency grid and scrapes vLLM's /metrics endpoint for KV-cache
occupancy in parallel.

Usage (single cell, manual):
    # Raw baseline
    RAG_MODE=raw locust -f benchmark/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 5 --run-time 90s --headless --csv=results/raw_c50

    # LSPM at r=0.3
    RAG_MODE=lspm COMPRESSION_RATIO=0.3 locust -f benchmark/locustfile.py \
        --host http://localhost:8000 --users 50 --spawn-rate 5 --run-time 90s \
        --headless --csv=results/lspm_r03_c50

    # Naive truncation at r=0.3
    RAG_MODE=naive COMPRESSION_RATIO=0.3 locust -f benchmark/locustfile.py \
        --host http://localhost:8000 --users 50 --spawn-rate 5 --run-time 90s \
        --headless --csv=results/naive_r03_c50

Normally you don't invoke this directly -- see run_full_sweep.py, which
drives all (method, ratio, concurrency) cells automatically.
"""

import json
import os
import random
import sys
import time
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

SAMPLE_QUERIES = [
    "متى تأسست جامعة الملك سعود؟",
    "ما هي اهتمامات قسم الحاسب في الجامعة؟",
    "كيف يكون الطقس في الرياض خلال الصيف؟",
    "ماذا يوجد في مكتبة الجامعة؟",
    "ما هو مركز أبحاث الذكاء الاصطناعي؟",
]

SYSTEM_PROMPT = "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك وبشكل مختصر."

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
        query = random.choice(SAMPLE_QUERIES)
        context = build_context(query)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"السياق: {context}\n\nالسؤال: {query}"},
            ],
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }

        t0 = time.perf_counter()
        ttft_ms = None
        completion_tokens_est = 0
        name = f"chat[{RAG_MODE}_r{COMPRESSION_RATIO}]" if RAG_MODE != "raw" else "chat[raw]"

        try:
            with self.client.post(
                "/v1/chat/completions", json=payload, catch_response=True, stream=True, name=name
            ) as resp:
                if resp.status_code != 200:
                    # DEBUG: print the real URL and body once so we can see
                    # why the server rejected this specific request (curl
                    # against the same path succeeds, so this is here to
                    # catch a locust-side URL/payload mismatch).
                    print(f"DEBUG FAILURE: url={resp.url!r} status={resp.status_code} body={resp.text[:300]!r}")
                    resp.failure(f"status={resp.status_code}")
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
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
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        completion_tokens_est += 1  # rough proxy; exact usage isn't sent per-chunk

                total_latency_ms = (time.perf_counter() - t0) * 1000
                resp.success()
        except Exception as e:
            events.request.fire(
                request_type="CUSTOM", name=f"{name}_error", response_time=0,
                response_length=0, exception=e, context={},
            )
            return

        # Custom metrics: TTFT and a rough tokens/sec, reported alongside
        # locust's native per-request latency (which equals total_latency_ms
        # here since we already consumed the full stream above).
        if ttft_ms is not None:
            events.request.fire(
                request_type="CUSTOM", name=f"ttft_ms[{RAG_MODE}_r{COMPRESSION_RATIO}]" if RAG_MODE != "raw" else "ttft_ms[raw]",
                response_time=ttft_ms, response_length=0, exception=None, context={},
            )
        tail_s = max((total_latency_ms - (ttft_ms or 0)), 1e-6) / 1000
        tokens_per_sec = completion_tokens_est / tail_s if completion_tokens_est else 0
        events.request.fire(
            request_type="CUSTOM", name=f"tokens_per_sec[{RAG_MODE}_r{COMPRESSION_RATIO}]" if RAG_MODE != "raw" else "tokens_per_sec[raw]",
            response_time=tokens_per_sec, response_length=completion_tokens_est, exception=None, context={},
        )
