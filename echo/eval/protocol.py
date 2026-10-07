"""Few-shot enrollment accuracy measured across the CV plan.

This is the accuracy engine behind the model scorecard. For every (seed, fold) in
the plan it runs the deployed protocol — speaker-independent prototypes from the
training speakers, then per-test-speaker K-shot enrollment — and reports, across
all folds:

  * **SI** (K=0): accuracy with prototypes from *other* speakers only.
  * **post@K**: accuracy after enrolling K of the test speaker's own recordings
    per intent (K in {3,5,10}), with the classifier restricted to enrolled
    intents (no empty-prototype predictions).
  * **gain@K = post@K − SI** measured on the *same* query utterances, so the
    adaptation effect is apples-to-apples.
  * **macro-F1@K**, and — the decision-critical view for atypical speech — a
    **per-severity-tier** accuracy breakdown.

Aggregation is mean ± ~95% CI over the (seed × fold) estimates, so a model's
number carries its uncertainty and its seed/fold **stability** is visible rather
than hidden in a single split.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import torch

from ..data.splits import CVPlan
from ..prototypical import PrototypicalNetwork, build_prototypes
from .benchmark import accuracy, extract_embeddings, macro_f1


def _agg(values: List[float]) -> Dict:
    xs = [x for x in values if x == x]
    if not xs:
        return dict(mean=float("nan"), sd=float("nan"), ci95=float("nan"), n=0)
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return dict(mean=m, sd=0.0, ci95=0.0, n=1)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    ci = 1.96 * sd / (len(xs) ** 0.5)          # normal approx, honest for n≥~10
    return dict(mean=m, sd=sd, ci95=ci, n=len(xs))


def _enroll_eval(proto, s_emb, s_int, si_protos, si_active, k, gen, n_classes):
    """One test speaker: enroll K/class, return (post_pred, si_pred_on_queries,
    labels) or None if no class has > K samples."""
    enroll_mask = torch.zeros(s_int.numel(), dtype=torch.bool)
    enrolled = []
    for c in s_int.unique():
        pos = (s_int == c).nonzero(as_tuple=True)[0]
        if pos.numel() <= k:
            continue
        perm = pos[torch.randperm(pos.numel(), generator=gen)]
        enroll_mask[perm[:k]] = True
        enrolled.append(int(c))
    if not enrolled:
        return None
    enrolled_t = torch.tensor(enrolled)
    active = torch.zeros(n_classes, dtype=torch.bool)
    active[enrolled_t] = True
    qmask = (~enroll_mask) & torch.isin(s_int, enrolled_t)
    if qmask.sum() == 0:
        return None
    protos = build_prototypes(s_emb[enroll_mask], s_int[enroll_mask], n_classes)
    q_emb, q_lab = s_emb[qmask], s_int[qmask]
    post = proto.classify(q_emb, protos, active)
    si = proto.classify(q_emb, si_protos, si_active)      # SI on identical queries
    return post, si, q_lab


def cross_validate(model, dataset, plan: CVPlan, device,
                   ks: Sequence[int] = (3, 5, 10),
                   distance: str = "euclidean") -> Dict:
    """Evaluate ``model`` across ``plan`` and return an aggregated result dict."""
    emb, intents, speakers = extract_embeddings(
        model, dataset, list(range(len(dataset))), device)
    speakers = list(speakers)
    # class count is derived from the labels present, so any label scheme works
    # (36-intent taxonomy, or one class per distinct prompt) with no hard-wiring.
    n_classes = int(intents.max().item()) + 1 if intents.numel() else 1
    # per-tier reporting always uses the recorded severity (speaker_tier), which
    # is populated whether or not the folds were stratified by it.
    tier_of = dict(plan.speaker_tier)
    proto = PrototypicalNetwork(distance=distance)

    # per-fold scalar series
    si_series: List[float] = []
    post_series: Dict[int, List[float]] = {k: [] for k in ks}
    gain_series: Dict[int, List[float]] = {k: [] for k in ks}
    f1_series: Dict[int, List[float]] = {k: [] for k in ks}
    # per-tier post@K series: tier -> k -> [fold accuracies]
    tier_series: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: {k: [] for k in ks})

    for f in plan.folds:
        tr = set(f.train_speakers) | set(f.val_speakers)
        te = set(f.test_speakers)
        idx_tr = [i for i, s in enumerate(speakers) if s in tr]
        idx_te = [i for i, s in enumerate(speakers) if s in te]
        if not idx_tr or not idx_te:
            continue
        idx_tr_t = torch.tensor(idx_tr)
        si_protos = build_prototypes(emb[idx_tr_t], intents[idx_tr_t], n_classes)
        si_active = si_protos.norm(dim=1) > 0

        te_emb = emb[torch.tensor(idx_te)]
        te_int = intents[torch.tensor(idx_te)]
        te_spk = [speakers[i] for i in idx_te]
        # SI accuracy over all test utterances of this fold
        si_pred = proto.classify(te_emb, si_protos, si_active)
        si_series.append(accuracy(si_pred, te_int))

        gen = torch.Generator().manual_seed(1000 * f.seed + f.fold)
        spk_set = sorted(set(te_spk))
        for k in ks:
            posts, sis, labs, tiers = [], [], [], []
            for spk in spk_set:
                sel = [j for j, sp in enumerate(te_spk) if sp == spk]
                sel_t = torch.tensor(sel)
                r = _enroll_eval(proto, te_emb[sel_t], te_int[sel_t],
                                 si_protos, si_active, k, gen, n_classes)
                if r is None:
                    continue
                post, si, lab = r
                posts.append(post); sis.append(si); labs.append(lab)
                tiers += [tier_of.get(spk, "unknown")] * lab.numel()
            if not labs:
                continue
            post_all = torch.cat(posts); si_all = torch.cat(sis); lab_all = torch.cat(labs)
            post_acc = accuracy(post_all, lab_all)
            post_series[k].append(post_acc)
            gain_series[k].append(post_acc - accuracy(si_all, lab_all))
            f1_series[k].append(macro_f1(post_all, lab_all))
            # per-tier post@K for this fold
            tiers_t = tiers
            for tier in set(tiers_t):
                m = torch.tensor([t == tier for t in tiers_t])
                tier_series[tier][k].append(accuracy(post_all[m], lab_all[m]))

    result = dict(
        n_fold_estimates=len(si_series),
        SI=_agg(si_series),
        post={k: _agg(post_series[k]) for k in ks},
        gain={k: _agg(gain_series[k]) for k in ks},
        macro_f1={k: _agg(f1_series[k]) for k in ks},
        per_tier={tier: {k: _agg(v[k]) for k in ks}
                  for tier, v in tier_series.items()},
        ks=list(ks),
    )
    return result


def print_accuracy(name: str, r: Dict) -> None:
    ks = r["ks"]
    print(f"\n===== accuracy across CV: {name} "
          f"({r['n_fold_estimates']} fold estimates) =====")
    s = r["SI"]
    print(f"  SI (K=0)   : {s['mean']*100:5.1f}% ± {s['ci95']*100:.1f}")
    for k in ks:
        p, g = r["post"][k], r["gain"][k]
        print(f"  post@K={k:<2} : {p['mean']*100:5.1f}% ± {p['ci95']*100:.1f}  "
              f"(gain {g['mean']*100:+.1f})")
    print("  per severity tier (post@K largest K):")
    kmax = ks[-1]
    for tier, byk in sorted(r["per_tier"].items()):
        a = byk[kmax]
        print(f"    {tier:<10}: {a['mean']*100:5.1f}% ± {a['ci95']*100:.1f} (n={a['n']})")
