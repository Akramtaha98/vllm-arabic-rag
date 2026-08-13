"""
Rule-based compression-ratio controllers for the network-aware LSPM
experiment (Task 5/6 of the network-specialization plan,
paper/NETWORK_SPECIALIZATION_PLAN.md, Deliverable 4).

Two controllers are implemented here, both transparent and rule-based --
explicitly NOT reinforcement learning, per instruction:

1. KVOnlyDecision -- wraps the existing, never-before-run
   middleware.pruning.DynamicRatioController (paper Section 3.4's original
   design: linear interpolation between two GPU KV-cache thresholds), with
   a logging wrapper so its decisions are recorded in exactly the same
   format as the new network-aware controller for a fair, symmetric
   comparison. This file does not reimplement that logic -- reusing the
   existing module is the correct way to finally *run* a design the paper
   has so far only described.

2. NetworkAwareController -- the new controller. Uses five inputs, exactly
   the five specified: available bandwidth, RTT, request queue length,
   concurrency (in-flight/concurrent users), and KV-cache utilization.
   No jitter or packet-loss term is included in the ratio decision itself
   (both are still measured and logged for analysis -- see
   network_profiles.py -- but are not decision inputs in this first,
   deliberately simple version, consistent with "do not begin with
   reinforcement learning" -- i.e., start with the simplest defensible
   rule, not the most elaborate one).

Every call to either controller's `decide()` method returns a
NetworkAwareDecision record containing every raw input, every normalized
input, the computed pressure score, and the resulting ratio -- intended to
be logged verbatim (one JSON line per decision) by edge_gateway.py, so a
reviewer can audit exactly why a given ratio was chosen for a given
request.

IMPORTANT -- calibration is not yet empirically fitted. The (min, max)
normalization ranges below are *placeholder operating ranges* stated
explicitly in paper/NETWORK_SPECIALIZATION_PLAN.md (Deliverable 4) as
requiring calibration from real measured data before the controller can be
trusted to make well-scaled decisions. They are written here as named,
overridable constants specifically so that after the validation grid
(run_validation_grid.py) produces real bandwidth/RTT/queue-length
distributions, they can be refit without touching the decision logic
itself. Do not present decisions made under the placeholder calibration as
validated in the paper -- they are a necessary first pass to make the
validation grid runnable at all.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

# --------------------------------------------------------------------------
# Placeholder calibration ranges (see module docstring). (lo, hi): lo maps to
# normalized 0 (no pressure from this input), hi maps to normalized 1 (max
# pressure). Sourced from the four network profiles in network_profiles.py
# and typical single-GPU serving ranges, NOT from fitted data yet.
# --------------------------------------------------------------------------
DEFAULT_CALIBRATION = {
    "bandwidth_mbps": (1.0, 100.0),   # low bandwidth = high pressure -> inverted in normalize()
    "rtt_ms": (1.0, 150.0),           # edge/LAN ~1ms floor .. constrained-wireless ~150ms ceiling
    "queue_length": (0.0, 50.0),
    "concurrency": (1.0, 50.0),
    "kv_cache_usage_pct": (0.0, 100.0),
}

DEFAULT_WEIGHTS_NETWORK_AWARE = {
    "bandwidth_mbps": 0.20,
    "rtt_ms": 0.20,
    "queue_length": 0.20,
    "concurrency": 0.15,
    "kv_cache_usage_pct": 0.25,
}

RATIO_MIN = 0.3
RATIO_MAX = 0.7
VALIDATED_RATIOS = (0.3, 0.5, 0.7)  # paper Section 5.4 -- fidelity-tested ratios only


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def snap_to_validated_ratio(r: float) -> float:
    """Snap a continuous ratio to the nearest fidelity-validated ratio
    (0.3/0.5/0.7, paper Section 5.4), so the controller never selects a
    ratio whose answer-quality effect this paper has not measured."""
    return min(VALIDATED_RATIOS, key=lambda v: abs(v - r))


@dataclass
class NetworkAwareDecision:
    timestamp: float
    controller: str  # "kv_only" | "network_aware" | "fixed"
    raw_inputs: dict
    normalized_inputs: dict
    weights: dict
    pressure_score: float
    ratio_continuous: float
    ratio_selected: float  # after snapping to a validated ratio
    fallback_used: bool
    notes: str = ""

    def to_json_dict(self):
        return asdict(self)


class NetworkAwareController:
    """Transparent, rule-based controller. ratio = f(bandwidth, RTT, queue
    length, concurrency, KV-cache utilization). No learned component."""

    def __init__(self, calibration: Optional[dict] = None, weights: Optional[dict] = None,
                 ratio_min: float = RATIO_MIN, ratio_max: float = RATIO_MAX,
                 snap_to_validated: bool = True):
        self.calibration = calibration or dict(DEFAULT_CALIBRATION)
        self.weights = weights or dict(DEFAULT_WEIGHTS_NETWORK_AWARE)
        w_sum = sum(self.weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"Controller weights must sum to 1.0, got {w_sum} ({self.weights})")
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.snap_to_validated = snap_to_validated

    def _normalize(self, key: str, value: Optional[float]) -> float:
        if value is None:
            return 0.5  # missing input -> neutral pressure, not zero (fail toward caution, not confidence)
        lo, hi = self.calibration[key]
        frac = (value - lo) / (hi - lo) if hi > lo else 0.5
        frac = _clip01(frac)
        if key == "bandwidth_mbps":
            frac = 1.0 - frac  # low bandwidth => high pressure
        return frac

    def decide(self, bandwidth_mbps: Optional[float], rtt_ms: Optional[float],
               queue_length: Optional[float], concurrency: Optional[float],
               kv_cache_usage_pct: Optional[float]) -> NetworkAwareDecision:
        raw = {
            "bandwidth_mbps": bandwidth_mbps, "rtt_ms": rtt_ms,
            "queue_length": queue_length, "concurrency": concurrency,
            "kv_cache_usage_pct": kv_cache_usage_pct,
        }
        fallback_used = any(v is None for v in raw.values())
        normalized = {k: self._normalize(k, v) for k, v in raw.items()}
        pressure = sum(self.weights[k] * normalized[k] for k in normalized)
        ratio_cont = self.ratio_max - (self.ratio_max - self.ratio_min) * pressure
        ratio_cont = max(self.ratio_min, min(self.ratio_max, ratio_cont))
        ratio_sel = snap_to_validated_ratio(ratio_cont) if self.snap_to_validated else ratio_cont
        return NetworkAwareDecision(
            timestamp=time.time(), controller="network_aware",
            raw_inputs=raw, normalized_inputs=normalized, weights=dict(self.weights),
            pressure_score=pressure, ratio_continuous=ratio_cont, ratio_selected=ratio_sel,
            fallback_used=fallback_used,
            notes="one or more inputs missing; neutral (0.5) substituted" if fallback_used else "",
        )


class KVOnlyDecision:
    """Logging wrapper around the existing, never-before-run
    middleware.pruning.DynamicRatioController, so its decisions are recorded
    in the same NetworkAwareDecision schema as the new controller for a
    like-for-like comparison. Does not change that controller's logic."""

    def __init__(self, metrics_url: str, min_ratio: float = 0.2, max_ratio: float = 0.8,
                 high_load_threshold: float = 0.75, low_load_threshold: float = 0.25):
        # Local import to avoid a hard dependency on sentence_transformers
        # (imported by middleware.pruning) for callers that only need the
        # controller math, not the pruner itself.
        from middleware.pruning import DynamicRatioController, DynamicRatioConfig
        self._impl = DynamicRatioController(DynamicRatioConfig(
            metrics_url=metrics_url, min_ratio=min_ratio, max_ratio=max_ratio,
            high_load_threshold=high_load_threshold, low_load_threshold=low_load_threshold,
        ))
        self.min_ratio, self.max_ratio = min_ratio, max_ratio

    async def decide(self, fallback_ratio: float = 0.5) -> NetworkAwareDecision:
        """Async so the underlying synchronous requests.get() call (inside
        _fetch_gpu_cache_usage) runs in a worker thread via asyncio.to_thread
        instead of blocking edge_gateway.py's single event loop -- see
        network_controller.py's module docstring history / the bug this
        fixes. Fetches usage exactly once and reuses it for get_ratio(),
        instead of the old code path that fetched twice (once here, once
        again inside get_ratio() with no usage argument)."""
        usage = await asyncio.to_thread(self._impl._fetch_gpu_cache_usage)
        ratio_cont = self._impl.get_ratio(fallback_ratio=fallback_ratio, usage=usage)
        ratio_sel = snap_to_validated_ratio(ratio_cont)
        return NetworkAwareDecision(
            timestamp=time.time(), controller="kv_only",
            raw_inputs={"kv_cache_usage_pct": usage}, normalized_inputs={},
            weights={}, pressure_score=float("nan") if usage is None else usage,
            ratio_continuous=ratio_cont, ratio_selected=ratio_sel,
            fallback_used=usage is None,
            notes="metrics fetch failed; fallback_ratio used" if usage is None else "",
        )
