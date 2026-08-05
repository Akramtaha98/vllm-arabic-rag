"""
Locust load test comparing:
  A) Baseline: raw (unpruned) context sent to vLLM
  B) LSPM: semantically pruned context sent to vLLM

Measures Throughput (req/s, tokens/s) and TTFT/latency percentiles under
increasing concurrency, per Part 4.1 of the research plan.

Usage:
    # Baseline run
    RAG_MODE=raw locust -f benchmark/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 5 --run-time 3m --headless --csv=results/baseline

    # LSPM run
    RAG_MODE=pruned COMPRESSION_RATIO=0.5 locust -f benchmark/locustfile.py \
        --host http://localhost:8000 --users 50 --spawn-rate 5 --run-time 3m \
        --headless --csv=results/lspm_r05
"""

import os
import random
import sys
import time
from pathlib import Path

from locust import HttpUser, task, between, events

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from middleware.pruning import SemanticPruner  # noqa: E402
from middleware.retriever import MOCK_CORPUS  # noqa: E402

RAG_MODE = os.getenv("RAG_MODE", "raw")  # "raw" | "pruned"
COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.5"))
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
PRUNER_MODEL_NAME = os.getenv("PRUNER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

SAMPLE_QUERIES = [
    "متى تأسست جامعة الملك سعود؟",
    "ما هي اهتمامات قسم الحاسب في الجامعة؟",
    "كيف يكون الطقس في الرياض خلال الصيف؟",
    "ماذا يوجد في مكتبة الجامعة؟",
    "ما هو مركز أبحاث الذكاء الاصطناعي؟",
]

_pruner = SemanticPruner(model_name=PRUNER_MODEL_NAME) if RAG_MODE == "pruned" else None


def build_context(query: str) -> str:
    docs = MOCK_CORPUS
    if RAG_MODE == "pruned":
        result = _pruner.prune(query, docs, compression_ratio=COMPRESSION_RATIO)
        return result.pruned_text
    return " ".join(docs)


class VLLMUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def chat_completion(self):
        query = random.choice(SAMPLE_QUERIES)
        context = build_context(query)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك وبشكل مختصر.",
                },
                {"role": "user", "content": f"السياق: {context}\n\nالسؤال: {query}"},
            ],
            "temperature": 0.3,
            "max_tokens": 256,
        }

        t0 = time.perf_counter()
        with self.client.post(
            "/v1/chat/completions", json=payload, catch_response=True, name=f"chat[{RAG_MODE}]"
        ) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
                return
            data = resp.json()
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            tokens_per_sec = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0

            events.request.fire(
                request_type="CUSTOM",
                name=f"tokens_per_sec[{RAG_MODE}]",
                response_time=tokens_per_sec,
                response_length=completion_tokens,
                exception=None,
                context={},
            )
            resp.success()
