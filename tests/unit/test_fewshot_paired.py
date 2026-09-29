"""The paired speaker-by-speaker test. Reference values from scipy.stats.wilcoxon(method="exact")."""

import numpy as np
import pytest

from echo_bench.fewshot.paired import compare, holm, wilcoxon_signed_rank


def test_all_speakers_better_gives_the_smallest_exact_p() -> None:
    w, p, method = wilcoxon_signed_rank(np.arange(1, 11) / 100)
    assert method == "exact" and w == 55 and p == pytest.approx(2 / 2**10)


def test_matches_scipy_exact_without_ties() -> None:
    d = np.array([0.031, -0.012, 0.054, 0.007, 0.022, -0.003, 0.041, 0.018, 0.009, -0.027])
    _, p, _ = wilcoxon_signed_rank(d)
    assert p == pytest.approx(0.130859375)  # scipy.stats.wilcoxon(d, method="exact").pvalue


def test_zero_differences_are_dropped_and_all_zero_has_no_p() -> None:
    assert wilcoxon_signed_rank(np.zeros(5))[1:] == (None, "none")
    result = compare({"a": 0.5, "b": 0.6, "c": 0.7}, {"a": 0.5, "b": 0.7, "c": None})
    assert (result.n_speakers, result.wins, result.ties) == (2, 1, 1)


def test_holm_adjustment() -> None:
    assert holm([0.01, 0.04, None, 0.03]) == pytest.approx([0.03, 0.06, None, 0.06])
