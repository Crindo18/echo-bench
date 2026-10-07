"""Meta-train-K x eval-K sweep — a comprehensive K comparison.

The eval K (how many samples a user enrolls) is already swept by
``cross_validate`` via ``--ks``. This adds the *other* axis: the K the embedding
head was **meta-trained** at. Sweeping both gives a matrix that answers "does the
head need to be trained at the same K it is deployed at, and which combination is
best?".

It is efficient and rigorous:
  * the frozen encoder is run **once** and its pooled features cached;
  * for each meta-train K and each CV fold, a *fresh* projector head is
    meta-trained on that fold's **training speakers only** (so it is
    leakage-free — the projector never sees the test speakers), then evaluated on
    the test speakers at every eval K;
  * results are averaged over (seed x fold) with a CI, exactly like the main
    protocol.

Because only the linear head is retrained (on cached features), the whole sweep
is cheap even across several K values.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from ..data.datasets import Collate
from ..data.splits import CVPlan
from ..heads import EmbeddingProjector, mean_pool
from ..prototypical import PrototypicalNetwork, build_prototypes, episodic_loss
from .benchmark import accuracy
from .protocol import _agg, _enroll_eval


@torch.no_grad()
def _pool_all(model, dataset, device, batch_size: int = 16):
    """Cache the frozen encoder's pooled feature per labelled utterance."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=Collate(None))
    pooled, intents, speakers = [], [], []
    for batch in loader:
        wav = batch["wav"].to(device)
        wl = batch["wav_lengths"].to(device)
        hidden, out_len = model.encode(wav, wl)
        p = mean_pool(hidden, out_len).cpu()
        for i in range(p.size(0)):
            it = batch["intents"][i]
            if it is None:
                continue
            pooled.append(p[i]); intents.append(it); speakers.append(batch["speakers"][i])
    if not pooled:
        return torch.zeros(0, model.encoder.out_dim), torch.zeros(0, dtype=torch.long), []
    return torch.stack(pooled), torch.tensor(intents, dtype=torch.long), speakers


def _train_head(pooled_tr, int_tr, in_dim, embed_dim, k_shot, n_way,
                epochs, episodes, lr, distance, seed):
    """Meta-train a fresh projector on cached TRAIN pooled features at k_shot."""
    proj = EmbeddingProjector(in_dim, embed_dim)
    proto = PrototypicalNetwork(distance=distance)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    rng = random.Random(seed)
    by_c: Dict[int, List[int]] = {}
    for i, c in enumerate(int_tr.tolist()):
        by_c.setdefault(c, []).append(i)
    classes = [c for c, v in by_c.items() if len(v) >= k_shot + 1]
    if len(classes) < 2:                      # not enough to form episodes
        proj.eval()
        return proj
    proj.train()
    for _ in range(epochs):
        for _ in range(episodes):
            ways = rng.sample(classes, k=min(n_way, len(classes)))
            s_idx, s_lab, q_idx, q_lab = [], [], [], []
            for new_c, c in enumerate(ways):
                idxs = by_c[c][:]
                rng.shuffle(idxs)
                sup = idxs[:k_shot]
                qry = idxs[k_shot:k_shot + k_shot] or idxs[k_shot - 1:k_shot]
                s_idx += sup; s_lab += [new_c] * len(sup)
                q_idx += qry; q_lab += [new_c] * len(qry)
            s_emb = proj.from_pooled(pooled_tr[torch.tensor(s_idx)])
            q_emb = proj.from_pooled(pooled_tr[torch.tensor(q_idx)])
            loss, _ = episodic_loss(proto, s_emb, torch.tensor(s_lab),
                                    q_emb, torch.tensor(q_lab), len(ways))
            opt.zero_grad(); loss.backward(); opt.step()
    for p in proj.parameters():
        p.requires_grad_(False)
    proj.eval()
    return proj


def metatrain_eval_sweep(model, dataset, plan: CVPlan, device,
                         meta_ks: Sequence[int] = (3, 5, 10),
                         eval_ks: Sequence[int] = (3, 5, 10),
                         epochs: int = 8, episodes: int = 150, n_way: int = 10,
                         lr: float = 1e-3, distance: str = "euclidean") -> Dict:
    """Return {matrix[meta_k][eval_k] = agg} over the CV plan (leakage-free)."""
    pooled, intents, speakers = _pool_all(model, dataset, device)
    n_classes = int(intents.max().item()) + 1 if intents.numel() else 1
    in_dim = pooled.size(1) if pooled.numel() else model.encoder.out_dim
    embed_dim = model.projector.proj.out_features
    proto = PrototypicalNetwork(distance=distance)

    matrix: Dict[int, Dict[int, Dict]] = {}
    for mk in meta_ks:
        series: Dict[int, List[float]] = {ek: [] for ek in eval_ks}
        for f in plan.folds:
            tr = set(f.train_speakers) | set(f.val_speakers)
            te = set(f.test_speakers)
            idx_tr = [i for i, s in enumerate(speakers) if s in tr]
            idx_te = [i for i, s in enumerate(speakers) if s in te]
            if not idx_tr or not idx_te:
                continue
            proj = _train_head(pooled[torch.tensor(idx_tr)], intents[torch.tensor(idx_tr)],
                               in_dim, embed_dim, mk, n_way, epochs, episodes, lr,
                               distance, seed=1000 * f.seed + f.fold)
            with torch.no_grad():
                emb = proj.from_pooled(pooled)
            si = build_prototypes(emb[torch.tensor(idx_tr)],
                                  intents[torch.tensor(idx_tr)], n_classes)
            si_active = si.norm(dim=1) > 0
            te_emb = emb[torch.tensor(idx_te)]
            te_int = intents[torch.tensor(idx_te)]
            te_spk = [speakers[i] for i in idx_te]
            gen = torch.Generator().manual_seed(1000 * f.seed + f.fold)
            for ek in eval_ks:
                posts, labs = [], []
                for spk in sorted(set(te_spk)):
                    sel = torch.tensor([j for j, sp in enumerate(te_spk) if sp == spk])
                    r = _enroll_eval(proto, te_emb[sel], te_int[sel],
                                     si, si_active, ek, gen, n_classes)
                    if r is None:
                        continue
                    post, _, lab = r
                    posts.append(post); labs.append(lab)
                if labs:
                    series[ek].append(accuracy(torch.cat(posts), torch.cat(labs)))
        matrix[mk] = {ek: _agg(series[ek]) for ek in eval_ks}
    return dict(meta_ks=list(meta_ks), eval_ks=list(eval_ks), matrix=matrix)


def print_sweep(sw: Dict, name: str = "") -> None:
    mks, eks = sw["meta_ks"], sw["eval_ks"]
    print(f"\n===== meta-train K x eval K post-enrollment accuracy: {name} =====")
    print("  (rows = K the head was meta-trained at; cols = K enrolled at test)")
    print("  meta\\eval " + "".join(f"  K={ek:<9}" for ek in eks))
    for mk in mks:
        cells = ""
        for ek in eks:
            a = sw["matrix"][mk][ek]
            cells += (f"  {a['mean']*100:5.1f}%±{a['ci95']*100:<4.1f}"
                      if a["n"] else "     -     ")
        print(f"  K={mk:<6}{cells}")
