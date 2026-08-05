"""
Lightweight Semantic Pruning Middleware (LSPM)
------------------------------------------------
Sentence-level relevance scoring and pruning of retrieved RAG context
using a cross-encoder re-ranker, to reduce KV-cache footprint on vLLM.

Supports:
  - Fixed compression ratio (baseline mode)
  - Dynamic compression ratio driven by vLLM /metrics (GPU/queue load)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from sentence_transformers import CrossEncoder

# --------------------------------------------------------------------------
# Arabic-aware sentence splitting
# --------------------------------------------------------------------------
# Arabic sentence terminators: '.', '؟', '!', '،' (comma is NOT a terminator,
# excluded on purpose) plus Arabic question mark U+061F and full stop variants.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\؟\?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split Arabic (or mixed) text into sentences without external NLP deps."""
    text = text.strip()
    if not text:
        return []
    # Normalize newlines to spaces, then split on terminators.
    text = re.sub(r"\s+", " ", text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    # Fallback: also split on bare periods if no terminators were found.
    if len(parts) == 1:
        parts = [p.strip() for p in text.split(".") if p.strip()]
    return [p.strip() for p in parts if p.strip()]


@dataclass
class PruningResult:
    pruned_text: str
    kept_sentences: List[str]
    dropped_sentences: List[str]
    scores: List[float]
    compression_ratio_used: float
    original_sentence_count: int
    kept_sentence_count: int
    original_char_count: int
    pruned_char_count: int
    latency_ms: float


class SemanticPruner:
    """Cross-encoder based sentence pruning for Arabic RAG context."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        self.ranker = CrossEncoder(model_name, device=device)

    def score(self, query: str, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        pairs = [[query, s] for s in sentences]
        scores = self.ranker.predict(pairs)
        return [float(s) for s in scores]

    def prune(
        self,
        query: str,
        documents: List[str],
        compression_ratio: float = 0.5,
        min_sentences: int = 1,
    ) -> PruningResult:
        """
        Rank all sentences across `documents` against `query` and keep the
        top `compression_ratio` fraction, preserving original relative order
        (not score order) for better narrative coherence in the LLM prompt.
        """
        t0 = time.perf_counter()

        all_sentences: List[str] = []
        for doc in documents:
            all_sentences.extend(split_sentences(doc))

        original_char_count = sum(len(s) for s in all_sentences)

        if not all_sentences:
            return PruningResult(
                pruned_text="",
                kept_sentences=[],
                dropped_sentences=[],
                scores=[],
                compression_ratio_used=compression_ratio,
                original_sentence_count=0,
                kept_sentence_count=0,
                original_char_count=0,
                pruned_char_count=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        scores = self.score(query, all_sentences)

        num_to_keep = max(min_sentences, int(round(len(all_sentences) * compression_ratio)))
        num_to_keep = min(num_to_keep, len(all_sentences))

        # Rank indices by score desc, keep top-k, then restore original order.
        ranked_idx = sorted(range(len(all_sentences)), key=lambda i: scores[i], reverse=True)
        keep_idx = set(ranked_idx[:num_to_keep])

        kept_sentences = [all_sentences[i] for i in range(len(all_sentences)) if i in keep_idx]
        dropped_sentences = [all_sentences[i] for i in range(len(all_sentences)) if i not in keep_idx]

        pruned_text = " ".join(kept_sentences)

        return PruningResult(
            pruned_text=pruned_text,
            kept_sentences=kept_sentences,
            dropped_sentences=dropped_sentences,
            scores=scores,
            compression_ratio_used=compression_ratio,
            original_sentence_count=len(all_sentences),
            kept_sentence_count=len(kept_sentences),
            original_char_count=original_char_count,
            pruned_char_count=len(pruned_text),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


# --------------------------------------------------------------------------
# Dynamic compression ratio controller (reads vLLM /metrics)
# --------------------------------------------------------------------------

@dataclass
class DynamicRatioConfig:
    metrics_url: str = "http://localhost:8000/metrics"
    min_ratio: float = 0.2
    max_ratio: float = 0.8
    # gpu KV-cache usage % thresholds (from vllm:gpu_cache_usage_perc)
    high_load_threshold: float = 0.75
    low_load_threshold: float = 0.25
    timeout_s: float = 2.0


class DynamicRatioController:
    """
    Polls vLLM's Prometheus /metrics endpoint and maps current GPU KV-cache
    usage to a compression ratio: high load -> aggressive pruning (low ratio),
    low load -> richer context (high ratio).
    """

    METRIC_NAME = "vllm:gpu_cache_usage_perc"

    def __init__(self, config: Optional[DynamicRatioConfig] = None):
        self.config = config or DynamicRatioConfig()

    def _fetch_gpu_cache_usage(self) -> Optional[float]:
        try:
            resp = requests.get(self.config.metrics_url, timeout=self.config.timeout_s)
            resp.raise_for_status()
        except Exception:
            return None

        for line in resp.text.splitlines():
            if line.startswith(self.METRIC_NAME) and not line.startswith("#"):
                try:
                    value = float(line.strip().split()[-1])
                    return value
                except (ValueError, IndexError):
                    continue
        return None

    def get_ratio(self, fallback_ratio: float = 0.5) -> float:
        usage = self._fetch_gpu_cache_usage()
        if usage is None:
            return fallback_ratio

        cfg = self.config
        if usage >= cfg.high_load_threshold:
            return cfg.min_ratio
        if usage <= cfg.low_load_threshold:
            return cfg.max_ratio

        # Linear interpolation between thresholds.
        span = cfg.high_load_threshold - cfg.low_load_threshold
        if span <= 0:
            return fallback_ratio
        frac = (usage - cfg.low_load_threshold) / span
        ratio = cfg.max_ratio - frac * (cfg.max_ratio - cfg.min_ratio)
        return max(cfg.min_ratio, min(cfg.max_ratio, ratio))
