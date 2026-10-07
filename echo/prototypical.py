"""Prototypical Network head — the shared few-shot classifier (spec Section 4).

Identical across all four encoders: same embedding dim, same distance metric.
Per-user enrollment builds intent prototypes from K = {3,5,10} samples; queries
are classified by distance to the user's own prototypes (Snell et al., 2017).

This is the stage where SO4 (few-shot adaptation) is measured. The encoder is
frozen; only prototypes (means of enrolment embeddings) define the classifier,
so no per-user training is needed — enrolment is a forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_prototypes(embeddings: torch.Tensor, labels: torch.Tensor,
                     n_classes: int) -> torch.Tensor:
    """Mean embedding per class -> [n_classes, D]. Classes with no support get 0."""
    d = embeddings.size(1)
    protos = embeddings.new_zeros(n_classes, d)
    counts = embeddings.new_zeros(n_classes)
    protos.index_add_(0, labels, embeddings)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))
    return protos / counts.clamp_min(1.0).unsqueeze(1)


class PrototypicalNetwork(nn.Module):
    def __init__(self, distance: str = "euclidean"):
        super().__init__()
        assert distance in ("euclidean", "cosine")
        self.distance = distance

    def logits(self, query: torch.Tensor, prototypes: torch.Tensor,
               active_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.distance == "euclidean":
            out = -torch.cdist(query, prototypes) ** 2       # [Q, C]
        else:
            q = F.normalize(query, dim=-1)
            p = F.normalize(prototypes, dim=-1)
            out = q @ p.t()
        if active_mask is not None:
            # never predict a class that has no prototype (e.g. an intent with
            # no enrollment samples) — its row would otherwise be the origin.
            out = out.masked_fill(~active_mask.to(out.device)[None, :],
                                  float("-inf"))
        return out

    def forward(self, support_emb, support_labels, query_emb, n_classes):
        protos = build_prototypes(support_emb, support_labels, n_classes)
        return self.logits(query_emb, protos)

    # ---- deployment-time enrollment (per user) ----------------------------- #
    @torch.no_grad()
    def enroll(self, support_emb, support_labels, n_classes) -> torch.Tensor:
        return build_prototypes(support_emb, support_labels, n_classes)

    @torch.no_grad()
    def classify(self, query_emb, prototypes,
                 active_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.logits(query_emb, prototypes, active_mask).argmax(dim=-1)


def episodic_loss(net: PrototypicalNetwork, support_emb, support_labels,
                  query_emb, query_labels, n_classes):
    logits = net(support_emb, support_labels, query_emb, n_classes)
    loss = F.cross_entropy(logits, query_labels)
    acc = (logits.argmax(-1) == query_labels).float().mean()
    return loss, acc
