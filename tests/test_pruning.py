"""
Unit tests for the LSPM pruning middleware. Run with:
    pytest tests/test_pruning.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from middleware.pruning import split_sentences, DynamicRatioController, DynamicRatioConfig


def test_split_sentences_basic():
    text = "هذه جملة أولى. هذه جملة ثانية؟ وهذه جملة ثالثة!"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_dynamic_ratio_high_load():
    cfg = DynamicRatioConfig(min_ratio=0.2, max_ratio=0.8, high_load_threshold=0.75, low_load_threshold=0.25)
    controller = DynamicRatioController(cfg)
    controller._fetch_gpu_cache_usage = lambda: 0.9  # simulate high load
    assert controller.get_ratio() == 0.2


def test_dynamic_ratio_low_load():
    cfg = DynamicRatioConfig(min_ratio=0.2, max_ratio=0.8, high_load_threshold=0.75, low_load_threshold=0.25)
    controller = DynamicRatioController(cfg)
    controller._fetch_gpu_cache_usage = lambda: 0.1  # simulate low load
    assert controller.get_ratio() == 0.8


def test_dynamic_ratio_interpolation():
    cfg = DynamicRatioConfig(min_ratio=0.2, max_ratio=0.8, high_load_threshold=0.75, low_load_threshold=0.25)
    controller = DynamicRatioController(cfg)
    controller._fetch_gpu_cache_usage = lambda: 0.5  # midpoint
    ratio = controller.get_ratio()
    assert 0.2 < ratio < 0.8


def test_dynamic_ratio_fallback_on_none():
    controller = DynamicRatioController(DynamicRatioConfig(metrics_url="http://invalid:9999/metrics"))
    ratio = controller.get_ratio(fallback_ratio=0.42)
    assert ratio == 0.42
