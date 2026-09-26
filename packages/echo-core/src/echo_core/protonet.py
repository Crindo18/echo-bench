"""Prototypical Network classification (Snell et al., 2017): the classifier ECHO runs on the device.

A class prototype is the mean of the enrollment embeddings for that class (thesis Eq. 1).
An utterance scores every class by the negative squared Euclidean distance to its
prototype, and a softmax over the scores gives the confidence. Prototypes live outside
the ONNX graph because they are per user and change with enrollment (assumption A2).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


def class_prototypes(embeddings: Vectors, labels: Sequence[str]) -> tuple[list[str], Vectors]:
    """Sorted class names and one mean vector per class."""
    if len(labels) != embeddings.shape[0]:
        raise ValueError(f"{len(labels)} labels for {embeddings.shape[0]} embeddings")
    if not len(labels):
        raise ValueError("no support embeddings")
    names = np.asarray(labels)
    classes = sorted(set(labels))
    means = [embeddings[names == c].astype(np.float64).mean(axis=0) for c in classes]
    return classes, np.stack(means).astype(np.float32)


def scores(queries: Vectors, prototypes: Vectors) -> Vectors:
    """-||query - prototype||^2 for every pair, shape [queries, classes]; never above 0."""
    q = queries.astype(np.float64)
    p = prototypes.astype(np.float64)
    squared = (q * q).sum(axis=1)[:, None] - 2.0 * (q @ p.T) + (p * p).sum(axis=1)[None, :]
    return (-np.maximum(squared, 0.0)).astype(np.float32)


def confidence(score_matrix: Vectors) -> Vectors:
    """Softmax over classes; each row sums to 1."""
    shifted = score_matrix.astype(np.float64) - score_matrix.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def classify(
    queries: Vectors, classes: Sequence[str], prototypes: Vectors
) -> tuple[list[str], Vectors]:
    """Predicted class per query and the confidence matrix."""
    score_matrix = scores(queries, prototypes)
    best = score_matrix.argmax(axis=1)
    return [classes[int(i)] for i in best], confidence(score_matrix)
