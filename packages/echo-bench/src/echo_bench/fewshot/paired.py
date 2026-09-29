"""Paired comparison of two encoders across speakers (thesis statistics section).

Every encoder is scored on the same speakers with the same seeded episodes, so each
speaker gives one PAIRED difference (encoder B minus encoder A). The test is the Wilcoxon
signed-rank test on those differences, exact (every sign pattern enumerated) for up to 20
speakers with a non-zero difference, and a seeded Monte Carlo over sign patterns above that.
Zero differences are dropped (Wilcoxon's method); tied differences get average ranks, and
the exact distribution is built from those same ranks, so ties are handled exactly too.

No SciPy: echo-bench stays light enough for the Raspberry Pi (SciPy arrives with M5 in
echo-analysis; its `scipy.stats.wilcoxon(method="exact")` gives the same p-values when
there are no ties or zeros).
"""

from dataclasses import dataclass

import numpy as np

EXACT_MAX_N = 20
MONTE_CARLO_DRAWS = 200_000


@dataclass(frozen=True)
class PairedResult:
    n_speakers: int  # speakers with a score for both encoders
    mean_diff: float  # B - A, as a fraction (0.037 = 3.7 points)
    median_diff: float
    wins: int  # speakers where B > A
    losses: int  # speakers where B < A
    ties: int
    statistic: float  # W+ : sum of ranks of the positive differences
    p_value: float | None  # two-sided; None if every difference is zero
    method: str  # "exact", "monte carlo" or "none"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def wilcoxon_signed_rank(diffs: np.ndarray, *, seed: int = 0) -> tuple[float, float | None, str]:
    """(W+, two-sided p, method) for paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 0.0, None, "none"
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    centre = ranks.sum() / 2.0
    observed = abs(w_plus - centre)
    if n <= EXACT_MAX_N:
        # All 2^n sign patterns: bit j of pattern i says whether rank j counts as positive.
        patterns = np.arange(2**n, dtype=np.int64)[:, None]
        signs = (patterns >> np.arange(n, dtype=np.int64)) & 1
        null = signs @ ranks
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        null = rng.integers(0, 2, size=(MONTE_CARLO_DRAWS, n)) @ ranks
        method = "monte carlo"
    p = float(np.mean(np.abs(null - centre) >= observed - 1e-9))
    return w_plus, min(1.0, p), method


def compare(a: dict[str, float | None], b: dict[str, float | None]) -> PairedResult:
    """a, b: per-speaker accuracy for one condition, keyed by speaker code."""
    speakers = sorted(s for s in a if a[s] is not None and b.get(s) is not None)
    diffs = np.array([b[s] - a[s] for s in speakers], dtype=np.float64)  # type: ignore[operator]
    if not len(diffs):
        return PairedResult(0, float("nan"), float("nan"), 0, 0, 0, 0.0, None, "none")
    w_plus, p, method = wilcoxon_signed_rank(diffs)
    return PairedResult(
        n_speakers=len(diffs),
        mean_diff=float(diffs.mean()),
        median_diff=float(np.median(diffs)),
        wins=int((diffs > 1e-12).sum()),
        losses=int((diffs < -1e-12).sum()),
        ties=int((np.abs(diffs) <= 1e-12).sum()),
        statistic=w_plus,
        p_value=p,
        method=method,
    )


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (same order), for several comparisons at once."""
    present = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    m = len(present)
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted
