"""Latency statistics (blueprint section 6.1): mean, SD, p50/p90/p95/p99, max,
and 95% bootstrap confidence intervals for p50 and p95."""

import numpy as np

Stat = tuple[float, float | None, float | None]  # (value, ci_low, ci_high)


def bootstrap_ci(
    values: np.ndarray, q: float, resamples: int, rng: np.random.Generator, alpha: float = 0.05
) -> tuple[float, float]:
    picks = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = np.percentile(values[picks], q, axis=1)
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def summarize(values_ms: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> dict[str, Stat]:
    x = np.asarray(values_ms, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no samples to summarize")
    rng = np.random.default_rng(seed)
    p50, p90, p95, p99 = (float(v) for v in np.percentile(x, [50, 90, 95, 99]))
    return {
        "latency_mean_ms": (float(x.mean()), None, None),
        "latency_sd_ms": (float(x.std(ddof=1)) if x.size > 1 else 0.0, None, None),
        "latency_p50_ms": (p50, *bootstrap_ci(x, 50, resamples, rng)),
        "latency_p90_ms": (p90, None, None),
        "latency_p95_ms": (p95, *bootstrap_ci(x, 95, resamples, rng)),
        "latency_p99_ms": (p99, None, None),
        "latency_max_ms": (float(x.max()), None, None),
    }
