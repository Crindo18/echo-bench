"""The quick few-shot check: how well do an encoder's embeddings separate words with
Prototypical Networks, before any training of our own? (Not the thesis protocol; that is M5.)

Conditions per speaker, each averaged over seeded random episodes of up to `ways` classes:

  SI   speaker-independent: prototypes from every OTHER speaker (rule L6), queries are
       this speaker's recordings. The K = 0 baseline.
  K=k  personal: prototypes from k of this speaker's own recordings per class, queries are
       their remaining recordings.

Word pool (`pool`):

  matched (default)  Every condition draws its classes from the SAME pool: words this
                     speaker recorded at least max(k) + 1 times that other speakers also
                     recorded. In each episode, SI and every K use the same chosen words, so
                     "K minus SI" measures personalization and nothing else. On UASpeech the
                     300 uncommon words (recorded once) are left out of every condition.
  all                The original behaviour: SI draws from every shared word (on UASpeech,
                     455, two thirds of them uncommon words) while K=k can only use words with
                     k + 1 recordings. Kept to reproduce earlier numbers; don't compare SI
                     with K under this setting.

One vector per recording, so support and query never share a recording (rule L2).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from echo_bench.fewshot.embedding_cache import EmbeddingSet
from echo_core.protonet import class_prototypes, classify

Pool = Literal["matched", "all"]

GROUP_ORDER = [
    "very_low",
    "low",
    "mid",
    "high",
    "severe",
    "moderate_severe",
    "moderate",
    "mild",
    "control",
    "?",
]


@dataclass
class SpeakerResult:
    speaker: str
    group: str
    accuracy: dict[str, float | None] = field(default_factory=dict)
    pool_size: dict[str, int] = field(default_factory=dict)


def _episode_accuracy(
    x: np.ndarray, labels: np.ndarray, support: np.ndarray, query: np.ndarray
) -> float:
    classes, protos = class_prototypes(x[support], list(labels[support]))
    predicted, _ = classify(x[query], classes, protos)
    return float(np.mean(np.asarray(predicted) == labels[query]))


def _si(x, labels, own, chosen) -> float:
    in_chosen = np.isin(labels, chosen)
    return _episode_accuracy(x, labels, ~own & in_chosen, own & in_chosen)


def _personal(x, labels, indices_by_class, chosen, k, rng) -> float:
    support, query = [], []
    for c in chosen:
        shuffled = rng.permutation(indices_by_class[c])
        support.extend(shuffled[:k])
        query.extend(shuffled[k:])
    return _episode_accuracy(x, labels, np.array(support), np.array(query))


def evaluate(
    es: EmbeddingSet,
    groups: dict[str, str],
    *,
    ks: list[int],
    ways: int = 10,
    episodes: int = 20,
    seed: int = 42,
    normalize: bool = False,
    pool: Pool = "matched",
) -> list[SpeakerResult]:
    if pool not in ("matched", "all"):
        raise ValueError(f"pool must be 'matched' or 'all', not {pool!r}")
    x = es.embeddings.astype(np.float32)
    if normalize:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels, speakers = es.label, es.speaker
    results = []
    for s_index, speaker in enumerate(sorted(set(speakers))):
        own = speakers == speaker
        own_classes = set(labels[own])
        result = SpeakerResult(speaker, groups.get(speaker, "?"))
        shared = own_classes & set(labels[~own])
        indices_by_class = {c: np.flatnonzero(own & (labels == c)) for c in own_classes}

        def eligible(k: int, indices_by_class=indices_by_class) -> set[str]:
            return {c for c, idx in indices_by_class.items() if len(idx) >= k + 1}

        if pool == "matched":
            common = sorted(shared & eligible(max(ks, default=0)))
            result.pool_size = {"SI": len(common), **{f"K={k}": len(common) for k in ks}}
            scores: dict[str, list[float]] = {c: [] for c in result.pool_size}
            if len(common) >= 2:
                for episode in range(episodes):
                    # One choice of words per episode, shared by SI and every K (paired).
                    pick = np.random.default_rng([seed, s_index, 0, episode])
                    chosen = pick.choice(common, size=min(ways, len(common)), replace=False)
                    scores["SI"].append(_si(x, labels, own, chosen))
                    for k in ks:
                        rng = np.random.default_rng([seed, s_index, k, episode, 1])
                        scores[f"K={k}"].append(
                            _personal(x, labels, indices_by_class, chosen, k, rng)
                        )
            for condition, values in scores.items():
                result.accuracy[condition] = float(np.mean(values)) if values else None
        else:
            si_pool = sorted(shared)
            result.pool_size["SI"] = len(si_pool)
            accuracies = []
            if len(si_pool) >= 2:
                for episode in range(episodes):
                    rng = np.random.default_rng([seed, s_index, 0, episode])
                    chosen = rng.choice(si_pool, size=min(ways, len(si_pool)), replace=False)
                    accuracies.append(_si(x, labels, own, chosen))
            result.accuracy["SI"] = float(np.mean(accuracies)) if accuracies else None
            for k in ks:
                k_pool = sorted(eligible(k))
                result.pool_size[f"K={k}"] = len(k_pool)
                accuracies = []
                if len(k_pool) >= 2:
                    for episode in range(episodes):
                        rng = np.random.default_rng([seed, s_index, k, episode])
                        chosen = rng.choice(k_pool, size=min(ways, len(k_pool)), replace=False)
                        accuracies.append(_personal(x, labels, indices_by_class, chosen, k, rng))
                result.accuracy[f"K={k}"] = float(np.mean(accuracies)) if accuracies else None
        results.append(result)
    return results


def summarize(
    results: list[SpeakerResult], conditions: list[str]
) -> list[tuple[str, int, dict[str, float | None]]]:
    """Macro averages (each speaker counts once) per group, then over all dysarthric speakers."""
    by_group: dict[str, list[SpeakerResult]] = defaultdict(list)
    for result in results:
        by_group[result.group].append(result)

    def mean(rows: list[SpeakerResult], condition: str) -> float | None:
        values = [r.accuracy[condition] for r in rows if r.accuracy.get(condition) is not None]
        return float(np.mean(values)) if values else None

    ordered = sorted(
        by_group, key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
    )
    rows = [(g, len(by_group[g]), {c: mean(by_group[g], c) for c in conditions}) for g in ordered]
    dysarthric = [r for r in results if r.group not in ("control", "?")]
    if dysarthric:
        rows.append(
            ("all dysarthric", len(dysarthric), {c: mean(dysarthric, c) for c in conditions})
        )
    return rows
