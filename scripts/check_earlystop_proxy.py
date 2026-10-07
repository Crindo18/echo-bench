"""Diagnostic: does the CTC-val-loss proxy agree with real few-shot accuracy?

WHY THIS EXISTS
---------------
Early stopping (and best-checkpoint selection) key off **CTC validation loss**.
But the number the benchmark actually reports is **few-shot intent accuracy**
after freezing the encoder and meta-training the Prototypical projector. Those
two are correlated, not identical, so the epoch with the lowest CTC loss is not
guaranteed to be the epoch with the best downstream accuracy.

This script measures whether they agree, for ONE encoder, ONE seed. It runs the
**full** finetune epoch budget with NO early stopping, and at each probed epoch
it records both numbers:

  * CTC val loss   (cheap; computed anyway each epoch)
  * few-shot acc   (expensive; freeze a COPY of the current encoder, meta-train
                    a fresh projector, run the few-shot eval)

then reports how far apart the best-by-CTC epoch and the best-by-accuracy epoch
are, plus rank/linear correlation, and writes a JSON (and a plot if matplotlib
is present) so you can eyeball the two curves.

HOW TO READ THE RESULT
----------------------
  * best epochs close together + strong correlation  -> the CTC proxy is fine;
    keep the cheap pipeline (early stopping on CTC val loss is justified).
  * best epochs far apart / weak-or-negative correlation -> the proxy misleads;
    select checkpoints on the downstream metric instead (see the DEVELOPMENT_LOG
    discussion). Do that on the inner val fold, never the test fold.

This is a DIAGNOSTIC, not a benchmark number: one model, one seed, and the
few-shot accuracy here is measured on the finetune holdout speakers purely to
see whether the curves move together. It does not replace the real scorecard.

COST
----
Each probed epoch pays a meta-train + eval on top of the CTC epoch. Use
``--probe-every 2`` (or 3) and a single ``--probe-ks 5`` to keep it affordable
on an 8 GB card; ``--probe-episodes`` also trades speed for a smoother curve.

Example
-------
    # (echo importable via `pip install -e .`, else prefix PYTHONPATH=.)
    python scripts/check_earlystop_proxy.py --encoder conformer \
        --init runs/conformer/pretrain.pt --torgo-root C:/Users/ASUS/Desktop/ECHO_Datasets/torgo \
        --holdout F01 M01 F03 M03 --epochs 40 --seed 1 \
        --probe-every 2 --probe-ks 5 \
        --out-json runs/conformer/proxy_check_seed1.json \
        --out-plot runs/conformer/proxy_check_seed1.png
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from echo.data import (CharTokenizer, TorgoDataset, Collate, map_prompt_to_intent,
                       speaker_disjoint_split)
from echo.encoders import count_parameters
from echo.eval.benchmark import evaluate
from echo.train.common import (build_optimizer, get_device, load_init_weights,
                               set_seed)
from echo.train.metatrain import episodic_metatrain
from echo.train.pretrain import build_model, train_one_epoch, validate


# --------------------------------------------------------------------------- #
# Small stats helpers (numpy-free so the script has no extra hard dependency)
# --------------------------------------------------------------------------- #
def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _rank(vals: List[float]) -> List[float]:
    """Average-rank of each value (ties share the mean rank)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


# --------------------------------------------------------------------------- #
# The per-epoch few-shot probe (runs on a COPY so training is never perturbed)
# --------------------------------------------------------------------------- #
def _fewshot_probe(model, ds, train_idx, holdout, device, args) -> Dict[int, float]:
    """Freeze a copy of the current encoder, meta-train a fresh projector, and
    return {k: post-enrollment accuracy} on the holdout speakers.

    Uses a deep copy because (a) meta-training freezes the encoder and trains the
    projector, and (b) the finetune optimizer wraps *all* params including the
    projector — so probing in place would corrupt both the weights CTC training
    leaves at init and the optimizer state. The copy sidesteps all of that.
    """
    try:
        probe = copy.deepcopy(model)
    except Exception as e:  # pragma: no cover - defensive
        # Fallback: rebuild and copy weights if deepcopy chokes on a module.
        print(f"[proxy] deepcopy failed ({e}); rebuilding probe from state_dict")
        probe = build_model(args.encoder, CharTokenizer(),
                            encoder_cfg=getattr(args, "_encoder_cfg", None)).to(device)
        probe.load_state_dict(model.state_dict())

    probe.eval()
    # meta-train the projector on the train speakers (encoder frozen inside)
    episodic_metatrain(
        probe, ds, train_idx, device,
        epochs=args.embed_epochs, episodes_per_epoch=args.episodes_per_epoch,
        n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
        lr=args.embed_lr, seed=args.seed, verbose=False)

    r = evaluate(probe, ds, holdout, ks=tuple(args.probe_ks),
                 n_episodes=args.probe_episodes, device=device, seed=args.seed)

    accs = {int(k): float(r["post"][k]["acc"]) for k in r["post"]}
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return accs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check CTC-val-loss vs. few-shot-accuracy agreement per epoch")
    # ---- mirror finetune's essential setup -------------------------------- #
    ap.add_argument("--encoder", required=True,
                    choices=["conformer", "ebranchformer", "ebranchformer_espnet",
                             "zipformer", "moonshine", "moonshine_tiny",
                             "moonshine_official", "dpwavlm"])
    ap.add_argument("--init", default=None,
                    help="pretrained/compressed checkpoint to start from")
    ap.add_argument("--torgo-root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo")
    ap.add_argument("--holdout", nargs="+", required=True,
                    help="speaker ids held out for eval (speaker-disjoint)")
    ap.add_argument("--dysarthric-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=40,
                    help="run the FULL budget (no early stop) to see the curve")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-duration", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--encoder-cfg", default=None,
                    help="checkpoint whose meta.encoder_cfg rebuilds a DPWavLM student")
    # ---- meta-train settings for the probe (match finetune defaults) ------- #
    ap.add_argument("--embed-epochs", type=int, default=10)
    ap.add_argument("--episodes-per-epoch", type=int, default=200)
    ap.add_argument("--n-way", type=int, default=10)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-query", type=int, default=5)
    ap.add_argument("--embed-lr", type=float, default=1e-3)
    # ---- probe cadence / cost knobs --------------------------------------- #
    ap.add_argument("--probe-every", type=int, default=1,
                    help="run the few-shot probe every N epochs (>=2 is cheaper)")
    ap.add_argument("--probe-ks", nargs="+", type=int, default=[5],
                    help="enrollment K(s) to evaluate in the probe (fewer = faster)")
    ap.add_argument("--probe-episodes", type=int, default=20,
                    help="episodes per K in the probe eval (more = smoother curve)")
    ap.add_argument("--primary-k", type=int, default=5,
                    help="which K defines 'best-by-accuracy' in the summary")
    ap.add_argument("--out-json", default="proxy_check.json")
    ap.add_argument("--out-plot", default=None,
                    help="optional .png; needs matplotlib (skipped if absent)")
    args = ap.parse_args()

    if args.primary_k not in args.probe_ks:
        args.primary_k = args.probe_ks[0]
        print(f"[proxy] primary-k not probed; using K={args.primary_k}")

    set_seed(args.seed)
    device = get_device()
    tokenizer = CharTokenizer()
    print(f"[proxy] encoder={args.encoder} seed={args.seed} device={device} "
          f"| full {args.epochs} epochs, NO early stop, probe every "
          f"{args.probe_every} (K={args.probe_ks})")

    # ---- data (identical split to finetune) -------------------------------- #
    ds = TorgoDataset(args.torgo_root, dysarthric_only=args.dysarthric_only,
                      intent_map=map_prompt_to_intent)
    train_idx, eval_idx = speaker_disjoint_split(ds, args.holdout)
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, eval_idx)
    print(f"[proxy] train {len(train_ds)} / eval {len(val_ds)} utts; "
          f"holdout {args.holdout}")

    max_samples = int(args.max_duration * 16000) if args.max_duration else None
    collate = Collate(tokenizer, max_samples=max_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    # ---- model (same as finetune, incl. DPWavLM cfg) ----------------------- #
    encoder_cfg = None
    if args.encoder == "dpwavlm" and args.encoder_cfg:
        meta = torch.load(args.encoder_cfg, map_location="cpu", weights_only=False)
        encoder_cfg = meta.get("meta", {}).get("encoder_cfg")
    args._encoder_cfg = encoder_cfg  # for the deepcopy fallback
    model = build_model(args.encoder, tokenizer, encoder_cfg=encoder_cfg).to(device)

    if args.init:
        meta = load_init_weights(args.init, model, map_location=device)
        print(f"[proxy] initialised from {args.init} (stage={meta.get('stage','?')})")

    print(f"[proxy] encoder params: "
          f"{count_parameters(model.encoder, trainable_only=False)/1e6:.2f}M")

    ft_lr = args.lr if args.lr is not None else (
        0.015 if args.encoder == "zipformer" else 1e-4)
    optimizer, is_scaled = build_optimizer(model, args.encoder, lr=ft_lr)
    total_steps = args.epochs * max(1, len(train_loader))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return max(0.02, 0.5 * (1 + math.cos(prog * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- the loop: CTC epoch -> record val loss -> (maybe) probe ----------- #
    rows: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler,
                             device, tokenizer)
        va = validate(model, val_loader, device, tokenizer) if len(val_ds) else tr

        row: Dict = {"epoch": epoch, "ctc_train_loss": float(tr),
                     "ctc_val_loss": float(va), "fewshot_acc": None}
        if epoch % args.probe_every == 0 or epoch == args.epochs:
            accs = _fewshot_probe(model, ds, train_idx, args.holdout, device, args)
            row["fewshot_acc"] = accs
            primary = accs.get(args.primary_k, float("nan"))
            print(f"[proxy] epoch {epoch:3d} | CTC val {va:7.3f} | "
                  f"few-shot@K{args.primary_k} {primary*100:5.1f}% | "
                  f"{time.time()-t0:6.1f}s")
        else:
            print(f"[proxy] epoch {epoch:3d} | CTC val {va:7.3f} | "
                  f"(no probe) | {time.time()-t0:6.1f}s")
        rows.append(row)

    # ---- summary: do the two curves agree? -------------------------------- #
    probed = [r for r in rows if r["fewshot_acc"] is not None]
    epochs = [r["epoch"] for r in probed]
    val_losses = [r["ctc_val_loss"] for r in probed]
    accs = [r["fewshot_acc"][args.primary_k] for r in probed]

    # CTC val loss is recorded EVERY epoch, so the epoch early stopping would
    # actually pick is the argmin over ALL rows -- not just the probed grid.
    # (With --probe-every 1 the two coincide.)
    best_ctc = min(rows, key=lambda r: r["ctc_val_loss"])["epoch"]
    best_ctc_probed = min(probed, key=lambda r: r["ctc_val_loss"])["epoch"]
    ctc_min_was_probed = (best_ctc == best_ctc_probed)
    best_acc = max(probed, key=lambda r: r["fewshot_acc"][args.primary_k])["epoch"]
    gap = abs(best_ctc - best_acc)

    # correlation between accuracy and *negated* val loss: positive = they agree
    neg_val = [-v for v in val_losses]
    pearson = _pearson(neg_val, accs)
    spearman = _spearman(neg_val, accs)

    # Accuracy lost by trusting the CTC pick instead of the true-best epoch.
    # Regret needs few-shot acc AT the CTC-picked epoch, which only exists if
    # that epoch was probed. If the real CTC-min fell off the probe grid, fall
    # back to the best probed CTC epoch and flag the regret as grid-approximate.
    regret_epoch = best_ctc if ctc_min_was_probed else best_ctc_probed
    acc_at_ctc_pick = next(r["fewshot_acc"][args.primary_k]
                           for r in probed if r["epoch"] == regret_epoch)
    acc_at_best = max(r["fewshot_acc"][args.primary_k] for r in probed)
    regret = acc_at_best - acc_at_ctc_pick

    summary = {
        "encoder": args.encoder, "seed": args.seed, "primary_k": args.primary_k,
        "n_probed_epochs": len(probed),
        "best_ctc_val_epoch": best_ctc,               # over ALL epochs
        "best_ctc_val_epoch_probed": best_ctc_probed, # over the probe grid
        "ctc_min_was_probed": ctc_min_was_probed,
        "best_fewshot_acc_epoch": best_acc,
        "epoch_gap": gap,
        "regret_is_grid_exact": ctc_min_was_probed,
        "pearson_negval_vs_acc": pearson,
        "spearman_negval_vs_acc": spearman,
        "acc_at_ctc_pick": acc_at_ctc_pick,
        "acc_at_true_best": acc_at_best,
        "proxy_regret_acc": regret,  # accuracy given up by trusting the proxy
    }

    print("\n===== proxy check summary =====")
    print(f"  best CTC-val epoch      : {best_ctc}  (over all epochs)")
    print(f"  best few-shot@K{args.primary_k} epoch : {best_acc}")
    print(f"  gap                     : {gap} epoch(s)")
    print(f"  correlation (Pearson)   : {pearson:+.3f}")
    print(f"  correlation (Spearman)  : {spearman:+.3f}")
    print(f"  acc at CTC pick         : {acc_at_ctc_pick*100:.1f}%"
          f"{'' if ctc_min_was_probed else f' (probed epoch {best_ctc_probed})'}")
    print(f"  acc at true best        : {acc_at_best*100:.1f}%")
    print(f"  proxy regret            : {regret*100:.1f} pts"
          f"{'' if ctc_min_was_probed else ' (grid-approx)'}")
    if not ctc_min_was_probed:
        print(f"  NOTE: the CTC-min epoch ({best_ctc}) was not probed, so gap and "
              f"regret use the probe grid.\n        Re-run with --probe-every 1 "
              f"for an exact read.")
    # a deliberately conservative verdict; the plot is the real evidence
    if gap <= max(1, args.probe_every) and (spearman != spearman or spearman >= 0.5):
        print("  VERDICT: curves track -> CTC-val-loss proxy looks justified.")
    else:
        print("  VERDICT: curves diverge -> prefer selecting checkpoints on the "
              "downstream metric (on the inner val fold, not the test fold).")
    print("  (Diagnostic only: one model, one seed. The plot is the real read.)")

    out = {"summary": summary, "rows": rows,
           "config": {k: v for k, v in vars(args).items()
                      if not k.startswith("_")}}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[proxy] wrote {args.out_json}")

    # ---- optional plot ---------------------------------------------------- #
    if args.out_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(epochs, val_losses, "o-", color="tab:red",
                     label="CTC val loss")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("CTC val loss", color="tab:red")
            ax1.tick_params(axis="y", labelcolor="tab:red")
            ax1.axvline(best_ctc, color="tab:red", ls=":", alpha=0.6)

            ax2 = ax1.twinx()
            ax2.plot(epochs, [a * 100 for a in accs], "s-", color="tab:blue",
                     label=f"few-shot@K{args.primary_k}")
            ax2.set_ylabel(f"few-shot acc @K{args.primary_k} (%)",
                           color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            ax2.axvline(best_acc, color="tab:blue", ls=":", alpha=0.6)

            plt.title(f"{args.encoder} (seed {args.seed}): CTC proxy vs. "
                      f"downstream\nbest-CTC ep {best_ctc} vs best-acc ep "
                      f"{best_acc} (gap {gap}); Spearman {spearman:+.2f}")
            fig.tight_layout()
            fig.savefig(args.out_plot, dpi=130)
            print(f"[proxy] wrote {args.out_plot}")
        except Exception as e:
            print(f"[proxy] plot skipped ({e}); the JSON has all the numbers")


if __name__ == "__main__":
    main()
