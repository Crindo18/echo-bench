"""Episodic meta-training of the Prototypical embedding head (spec Section 6).

Why this stage exists
---------------------
CTC pretraining/fine-tuning trains the *encoder* (via ``CTCHead``) to produce a
good acoustic/phonetic representation, but it never touches the
``EmbeddingProjector`` — that linear head is not on the CTC computation graph.
Left untrained, the projector would be a random linear map and the Prototypical
Network would be classifying on a random projection of pooled features.

This module trains the projector the standard Prototypical-Network way (Snell et
al., 2017): sample N-way/K-shot **episodes** from the intent-labelled TORGO
*training* speakers (speaker-disjoint from the enrollment-eval speakers), build
prototypes from the support set, and minimise the cross-entropy of query
distances-to-prototypes. The encoder stays **frozen** throughout (the deployed
feature extractor is fixed), so this is cheap and — crucially for the benchmark —
*identical* for all four encoders: the only thing that differs is the frozen
representation feeding it.

Efficiency
----------
Because the encoder is frozen, each utterance's pooled feature is constant. We
therefore run the encoder **once** over the training utterances, cache the pooled
vectors, and run the episodic loop purely on the cache (thousands of cheap linear
forwards), instead of re-encoding audio every episode.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from ..heads import mean_pool
from ..prototypical import PrototypicalNetwork, episodic_loss
from .common import AcousticModel


@torch.no_grad()
def _precompute_pooled(model: AcousticModel, dataset, usable: Dict[int, List[int]],
                       device, batch_size: int = 8) -> Dict[int, torch.Tensor]:
    """Encode every usable utterance once and cache the frozen pooled feature,
    grouped by intent. Returns {intent: [n_i, encoder_out_dim]}."""
    model.eval()
    flat = [(idx, c) for c, idxs in usable.items() for idx in idxs]
    pooled_by_intent: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(flat), batch_size):
        chunk = flat[start:start + batch_size]
        items = [dataset[idx] for idx, _ in chunk]
        wavs = [it["wav"] for it in items]
        lengths = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
        maxl = int(lengths.max())
        padded = torch.zeros(len(wavs), maxl)
        for i, w in enumerate(wavs):
            padded[i, : w.numel()] = w
        hidden, out_len = model.encode(padded.to(device), lengths.to(device))
        pooled = mean_pool(hidden, out_len).cpu()
        for i, (_, c) in enumerate(chunk):
            pooled_by_intent[c].append(pooled[i])
    return {c: torch.stack(v) for c, v in pooled_by_intent.items()}


def _sample_episode(pooled: Dict[int, torch.Tensor], classes: List[int],
                    n_way: int, k_shot: int, n_query: int, rng: random.Random):
    """Sample one N-way/K-shot episode from cached pooled features.

    Query labels are remapped to 0..way-1 to match the compact prototype tensor.
    """
    ways = rng.sample(classes, k=min(n_way, len(classes)))
    s_vec, s_lab, q_vec, q_lab = [], [], [], []
    for new_c, c in enumerate(ways):
        n = pooled[c].size(0)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(
            rng.randrange(1 << 30)))
        sup = perm[:k_shot]
        qry = perm[k_shot:k_shot + n_query]
        if qry.numel() == 0:                       # ensure >=1 query
            qry = perm[k_shot - 1: k_shot]
        s_vec.append(pooled[c][sup]); s_lab += [new_c] * sup.numel()
        q_vec.append(pooled[c][qry]); q_lab += [new_c] * qry.numel()
    return (torch.cat(s_vec), torch.tensor(s_lab),
            torch.cat(q_vec), torch.tensor(q_lab), len(ways))


def episodic_metatrain(model: AcousticModel, dataset, train_idx: List[int],
                       device, *, epochs: int = 10, episodes_per_epoch: int = 200,
                       n_way: int = 10, k_shot: int = 5, n_query: int = 5,
                       lr: float = 1e-3, distance: str = "euclidean",
                       seed: int = 0, verbose: bool = True) -> Optional[float]:
    """Train ``model.projector`` on intent episodes; freeze it when done.

    The encoder (and front-end) are frozen first, so only the projector learns.
    Returns the final mean episodic query accuracy (or ``None`` if there were too
    few intents to form episodes — in which case the projector is left frozen at
    init and the caller should note the eval uses an untrained embedding head).
    """
    model.freeze_encoder()
    for p in model.projector.parameters():
        p.requires_grad_(True)

    # index intents cheaply from item metadata (no audio load)
    intent_to_idx: Dict[int, List[int]] = defaultdict(list)
    for idx in train_idx:
        it = dataset.items[idx] if hasattr(dataset, "items") else dataset[idx]
        intent = it["intent"] if isinstance(it, dict) else it.intent
        if intent is not None:
            intent_to_idx[int(intent)].append(idx)
    usable = {c: ix for c, ix in intent_to_idx.items() if len(ix) >= k_shot + 1}
    if len(usable) < 2:
        if verbose:
            print("[metatrain] <2 intents with enough samples; skipping "
                  "projector meta-training (embedding head stays at init).")
        for p in model.projector.parameters():
            p.requires_grad_(False)
        return None

    if verbose:
        print(f"[metatrain] caching pooled features for {sum(len(v) for v in usable.values())} "
              f"utterances across {len(usable)} intents ...")
    pooled = _precompute_pooled(model, dataset, usable, device)
    pooled = {c: v.to(device) for c, v in pooled.items()}
    classes = list(pooled.keys())

    proto = PrototypicalNetwork(distance=distance)
    opt = torch.optim.Adam(model.projector.parameters(), lr=lr)
    rng = random.Random(seed)

    last_acc = float("nan")
    for epoch in range(1, epochs + 1):
        model.projector.train()
        losses, accs = [], []
        for _ in range(episodes_per_epoch):
            s_vec, s_lab, q_vec, q_lab, way = _sample_episode(
                pooled, classes, n_way, k_shot, n_query, rng)
            s_emb = model.projector.from_pooled(s_vec)
            q_emb = model.projector.from_pooled(q_vec)
            loss, acc = episodic_loss(proto, s_emb, s_lab.to(device),
                                      q_emb, q_lab.to(device), way)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item()); accs.append(acc.item())
        last_acc = sum(accs) / len(accs)
        if verbose:
            print(f"[metatrain] epoch {epoch:3d} | loss {sum(losses)/len(losses):6.3f} "
                  f"| episodic acc {last_acc*100:5.1f}%")

    for p in model.projector.parameters():
        p.requires_grad_(False)
    model.projector.eval()
    return last_acc
