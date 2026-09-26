"""The quick few-shot check: how well do an encoder's embeddings separate words with
Prototypical Networks, before any training of our own? (Not the thesis protocol; that is M5.)

Two conditions per speaker, each averaged over seeded random episodes of up to `ways` classes:

  SI   speaker-independent: prototypes from every OTHER speaker (rule L6), queries are
       this speaker's recordings. The K = 0 baseline.
  K=k  personal: prototypes from k of this speaker's own recordings per class, queries are
       their remaining recordings (only classes with at least k + 1 recordings).

One vector per recording, so support and query never share a recording (rule L2).
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from echo_bench.fewshot.embedding_cache import EmbeddingSet
from echo_core.protonet import class_prototypes, classify

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


def _episode_accuracy(
    x: np.ndarray, labels: np.ndarray, support: np.ndarray, query: np.ndarray
) -> float:
    classes, protos = class_prototypes(x[support], list(labels[support]))
    predicted, _ = classify(x[query], classes, protos)
    return float(np.mean(np.asarray(predicted) == labels[query]))


def evaluate(
    es: EmbeddingSet,
    groups: dict[str, str],
    *,
    ks: list[int],
    ways: int = 10,
    episodes: int = 20,
    seed: int = 42,
    normalize: bool = False,
) -> list[SpeakerResult]:
    x = es.embeddings.astype(np.float32)
    if normalize:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels, speakers = es.label, es.speaker
    results = []
    for s_index, speaker in enumerate(sorted(set(speakers))):
        own = speakers == speaker
        own_classes = set(labels[own])
        result = SpeakerResult(speaker, groups.get(speaker, "?"))

        shared = sorted(own_classes & set(labels[~own]))
        accuracies = []
        if len(shared) >= 2:
            for episode in range(episodes):
                rng = np.random.default_rng([seed, s_index, 0, episode])
                chosen = rng.choice(shared, size=min(ways, len(shared)), replace=False)
                in_chosen = np.isin(labels, chosen)
                accuracies.append(_episode_accuracy(x, labels, ~own & in_chosen, own & in_chosen))
        result.accuracy["SI"] = float(np.mean(accuracies)) if accuracies else None

        indices_by_class = {c: np.flatnonzero(own & (labels == c)) for c in own_classes}
        for k in ks:
            eligible = sorted(c for c, idx in indices_by_class.items() if len(idx) >= k + 1)
            accuracies = []
            if len(eligible) >= 2:
                for episode in range(episodes):
                    rng = np.random.default_rng([seed, s_index, k, episode])
                    chosen = rng.choice(eligible, size=min(ways, len(eligible)), replace=False)
                    support, query = [], []
                    for c in chosen:
                        shuffled = rng.permutation(indices_by_class[c])
                        support.extend(shuffled[:k])
                        query.extend(shuffled[k:])
                    accuracies.append(
                        _episode_accuracy(x, labels, np.array(support), np.array(query))
                    )
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
