"""Few-shot enrollment evaluation — the accuracy/adaptation metrics panel.

Implements the SO1/SO4 metrics from spec Section 7 on a frozen encoder:

  * Pre-enrollment accuracy  — speaker-independent: prototypes built from the
    *training* speakers, evaluated on held-out speakers' utterances.
  * Post-enrollment accuracy @ K in {3,5,10} — speaker-adapted: prototypes
    built from K of the test speaker's *own* samples per intent, evaluated on
    that speaker's remaining utterances (enrollment samples excluded from the
    query set — the data-hygiene rule in spec Section 6).
  * Adaptation gain Δ = post − pre, per K.
  * Macro-F1 across the 36 intents.

All results are reported mean ± SD across the enrollment sampling (and, when the
caller aggregates several seed checkpoints, across seeds — spec Section 7).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from ..data import (CharTokenizer, TorgoDataset, Collate, map_prompt_to_intent,
                    speaker_disjoint_split)
from ..prototypical import PrototypicalNetwork, build_prototypes
from ..train.common import AcousticModel, get_device, load_checkpoint
from ..train.pretrain import build_model


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(model: AcousticModel, dataset, indices, device,
                       batch_size: int = 8):
    """Return (emb[N,D], intents[N], speakers[N]) for items that have an intent."""
    sub = Subset(dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        collate_fn=Collate(None))
    embs, intents, speakers = [], [], []
    model.eval()
    for batch in loader:
        wav = batch["wav"].to(device)
        wav_lengths = batch["wav_lengths"].to(device)
        e = model.embed(wav, wav_lengths).cpu()
        for i in range(e.size(0)):
            it = batch["intents"][i]
            if it is None:
                continue
            embs.append(e[i])
            intents.append(it)
            speakers.append(batch["speakers"][i])
    if not embs:
        return (torch.zeros(0, model.projector.proj.out_features),
                torch.zeros(0, dtype=torch.long), [])
    return torch.stack(embs), torch.tensor(intents, dtype=torch.long), speakers


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return (preds == labels).float().mean().item()


def macro_f1(preds: torch.Tensor, labels: torch.Tensor,
             n_classes: Optional[int] = None) -> float:
    if labels.numel() == 0:
        return float("nan")
    if n_classes is None:                         # derive from the labels present
        n_classes = int(max(int(labels.max()),
                            int(preds.max()) if preds.numel() else 0)) + 1
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        if tp + fp + fn == 0:
            continue                          # class absent from this eval set
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def _mean_sd(xs: List[float]):
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


# --------------------------------------------------------------------------- #
# Enrollment episodes
# --------------------------------------------------------------------------- #
def evaluate(model: AcousticModel, dataset: TorgoDataset, holdout: List[str],
             ks=(3, 5, 10), n_episodes: int = 20, distance: str = "euclidean",
             device=None, seed: int = 0, n_classes: Optional[int] = None) -> Dict:
    """Full panel on one frozen model. Returns nested dict of metrics.

    ``n_classes`` defaults to the number of classes present in the data, so any
    label scheme (36-intent taxonomy or one class per prompt) works unchanged.
    """
    device = device or get_device()
    proto = PrototypicalNetwork(distance=distance)
    gen = torch.Generator().manual_seed(seed)

    train_idx, eval_idx = speaker_disjoint_split(dataset, holdout)

    # speaker-independent prototypes from TRAIN speakers -> pre-enrollment
    tr_emb, tr_int, _ = extract_embeddings(model, dataset, train_idx, device)
    ev_emb, ev_int, ev_spk = extract_embeddings(model, dataset, eval_idx, device)

    if n_classes is None:
        hi = max(int(tr_int.max()) if tr_int.numel() else -1,
                 int(ev_int.max()) if ev_int.numel() else -1)
        n_classes = hi + 1 if hi >= 0 else 1

    results: Dict = {"pre": {}, "post": {}, "gain": {}, "macro_f1": {},
                     "n_eval": int(ev_int.numel())}

    # ---- pre-enrollment (speaker-independent) ----------------------------- #
    if tr_int.numel() and ev_int.numel():
        si_protos = build_prototypes(tr_emb, tr_int, n_classes)
        si_active = si_protos.norm(dim=1) > 0        # intents seen in train
        pre_preds = proto.classify(ev_emb, si_protos, si_active)
        pre_acc = accuracy(pre_preds, ev_int)
        pre_f1 = macro_f1(pre_preds, ev_int)
    else:
        pre_acc, pre_f1 = float("nan"), float("nan")
    results["pre"]["acc"] = pre_acc
    results["macro_f1"]["pre"] = pre_f1

    # ---- post-enrollment per K (speaker-adapted) -------------------------- #
    # group eval embeddings by speaker
    by_spk = defaultdict(list)
    for i, s in enumerate(ev_spk):
        by_spk[s].append(i)

    for k in ks:
        ep_accs, ep_f1s = [], []
        for _ in range(n_episodes):
            all_preds, all_labels = [], []
            for spk, idxs in by_spk.items():
                idxs_t = torch.tensor(idxs)
                spk_emb = ev_emb[idxs_t]
                spk_int = ev_int[idxs_t]
                # per intent, sample K enrollment; rest are queries. An intent
                # needs >K samples to both enroll AND be queried; intents with
                # <=K samples are excluded from BOTH sets this episode (they
                # can't be enrolled, so a query for them would be unanswerable).
                enroll_mask = torch.zeros(spk_int.numel(), dtype=torch.bool)
                enrolled_classes = []
                for c in spk_int.unique():
                    c_pos = (spk_int == c).nonzero(as_tuple=True)[0]
                    if c_pos.numel() <= k:
                        continue
                    perm = c_pos[torch.randperm(c_pos.numel(), generator=gen)]
                    enroll_mask[perm[:k]] = True
                    enrolled_classes.append(int(c))
                if not enrolled_classes:
                    continue
                enrolled = torch.tensor(enrolled_classes)
                active = torch.zeros(n_classes, dtype=torch.bool)
                active[enrolled] = True
                # queries: this speaker's non-enrolled samples whose intent WAS
                # enrolled (so the classifier has a prototype for the true class)
                query_mask = (~enroll_mask) & torch.isin(spk_int, enrolled)
                if query_mask.sum() == 0:
                    continue
                protos = build_prototypes(spk_emb[enroll_mask],
                                          spk_int[enroll_mask], n_classes)
                q_emb, q_lab = spk_emb[query_mask], spk_int[query_mask]
                preds = proto.classify(q_emb, protos, active)
                all_preds.append(preds)
                all_labels.append(q_lab)
            if not all_preds:
                continue
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            ep_accs.append(accuracy(preds, labels))
            ep_f1s.append(macro_f1(preds, labels))

        m_acc, sd_acc = _mean_sd(ep_accs)
        m_f1, _ = _mean_sd(ep_f1s)
        results["post"][k] = dict(acc=m_acc, acc_sd=sd_acc)
        results["macro_f1"][k] = m_f1
        results["gain"][k] = (m_acc - pre_acc) if pre_acc == pre_acc else float("nan")

    return results


def print_report(name: str, r: Dict) -> None:
    print(f"\n===== {name} =====")
    print(f"  eval utterances (with intent): {r['n_eval']}")
    print(f"  pre-enrollment acc : {r['pre']['acc']*100:5.1f}%  "
          f"(macro-F1 {r['macro_f1']['pre']*100:4.1f}%)")
    for k in sorted(x for x in r['post']):
        p = r['post'][k]
        print(f"  post @K={k:<2d}       : {p['acc']*100:5.1f}% "
              f"± {p['acc_sd']*100:4.1f}  | Δ {r['gain'][k]*100:+5.1f}  "
              f"| macro-F1 {r['macro_f1'][k]*100:4.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Few-shot enrollment benchmark")
    ap.add_argument("--encoder", required=True,
                    choices=["conformer", "ebranchformer", "ebranchformer_espnet", "zipformer", "moonshine", "moonshine_tiny", "moonshine_official", "dpwavlm"])
    ap.add_argument("--checkpoint", required=True, help="frozen finetuned model")
    ap.add_argument("--torgo-root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo")
    ap.add_argument("--holdout", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--distance", default="euclidean",
                    choices=["euclidean", "cosine"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    tokenizer = CharTokenizer()
    encoder_cfg = None
    if args.encoder == "dpwavlm":
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder_cfg = meta.get("meta", {}).get("encoder_cfg")
    model = build_model(args.encoder, tokenizer, encoder_cfg=encoder_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=False)

    ds = TorgoDataset(args.torgo_root, intent_map=map_prompt_to_intent)
    r = evaluate(model, ds, args.holdout, ks=tuple(args.ks),
                 n_episodes=args.episodes, distance=args.distance,
                 device=device, seed=args.seed)
    print_report(f"{args.encoder} (seed {args.seed})", r)


if __name__ == "__main__":
    main()
