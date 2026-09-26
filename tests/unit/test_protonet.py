"""The on-device Prototypical Network math (echo_core.protonet), with property checks from blueprint §8."""

import numpy as np
import pytest

from echo_core.protonet import class_prototypes, classify, confidence, scores

rng = np.random.default_rng(0)


def test_prototype_is_the_class_mean() -> None:
    x = np.array([[0, 0], [2, 2], [10, 10]], dtype=np.float32)
    classes, protos = class_prototypes(x, ["a", "a", "b"])
    assert classes == ["a", "b"]
    np.testing.assert_allclose(protos, [[1, 1], [10, 10]])


def test_support_order_does_not_change_prototypes() -> None:
    x = rng.standard_normal((20, 8)).astype(np.float32)
    labels = [f"c{i % 4}" for i in range(20)]
    order = rng.permutation(20)
    _, first = class_prototypes(x, labels)
    _, second = class_prototypes(x[order], [labels[i] for i in order])
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_scores_are_negative_squared_distances_and_confidences_sum_to_one() -> None:
    q = rng.standard_normal((5, 16)).astype(np.float32)
    p = rng.standard_normal((3, 16)).astype(np.float32)
    s = scores(q, p)
    expected = -((q[:, None, :] - p[None, :, :]) ** 2).sum(axis=-1)
    np.testing.assert_allclose(s, expected, rtol=1e-4, atol=1e-4)
    assert (s <= 0).all()
    np.testing.assert_allclose(confidence(s).sum(axis=1), 1.0, atol=1e-6)


def test_permuting_class_names_permutes_predictions() -> None:
    protos = np.eye(3, dtype=np.float32) * 5
    queries = protos + 0.1
    predicted, _ = classify(queries, ["a", "b", "c"], protos)
    renamed, _ = classify(queries, ["c", "a", "b"], protos)
    assert predicted == ["a", "b", "c"] and renamed == ["c", "a", "b"]


def test_mismatched_labels_are_rejected() -> None:
    with pytest.raises(ValueError):
        class_prototypes(np.zeros((3, 2), dtype=np.float32), ["a", "b"])
