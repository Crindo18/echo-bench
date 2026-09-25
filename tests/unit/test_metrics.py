"""Latency statistics (section 6.1)."""

import numpy as np
import pytest

from echo_bench.metrics.latency import summarize


def test_percentiles_and_confidence_intervals() -> None:
    values = np.arange(1, 101, dtype=float)  # 1 ... 100 ms
    stats = summarize(values, resamples=500, seed=1)
    assert stats["latency_p50_ms"][0] == pytest.approx(50.5)
    assert stats["latency_p95_ms"][0] == pytest.approx(95.05)
    assert stats["latency_max_ms"][0] == 100
    assert stats["latency_sd_ms"][0] == pytest.approx(values.std(ddof=1))
    for key in ("latency_p50_ms", "latency_p95_ms"):
        value, low, high = stats[key]
        assert low is not None and high is not None and low <= value <= high


def test_same_seed_same_intervals() -> None:
    values = np.random.default_rng(0).gamma(2.0, 5.0, 300)
    assert summarize(values, seed=7) == summarize(values, seed=7)


def test_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError):
        summarize(np.array([]))
