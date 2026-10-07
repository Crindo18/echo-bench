"""Fine-tune a pretrained encoder on TORGO (atypical speech), then freeze it.

Stage 2 for all four models (spec Section 6). The pretrained (supervised) or
compressed (DPWavLM) encoder is adapted to dysarthric speech by continuing CTC
training on the speaker-disjoint TORGO/UASpeech train partition, then frozen.
The frozen encoder + ``EmbeddingProjector`` is what the Prototypical head
consumes at enrollment time — so this script's output checkpoint is exactly the
artifact the quantizer and the few-shot evaluator load.

Multi-seed support (``--seed``) is how the convergence / seed-variance metric
(spec Section 7) is produced: run this across >=5 seeds per model and compare.

Example
-------
    python -m echo.train.finetune --encoder ebranchformer \
        --init runs/ebranchformer/pretrain.pt \
        --torgo-root C:/Users/ASUS/Desktop/ECHO_Datasets/torgo --holdout F01 M01 \
        --epochs 30 --out runs/ebranchformer/finetune_seed1.pt --seed 1
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from ..data import (CharTokenizer, TorgoDataset, Collate, LabelScheme,
                    speaker_disjoint_split)
from ..encoders import count_parameters
from .common import (build_optimizer, get_device, load_checkpoint,
                     load_init_weights, save_checkpoint, set_seed)
from .pretrain import build_model, train_one_epoch, validate


def main() -> None:
    ap = argparse.ArgumentParser(description="Finetune on TORGO, then freeze")
    ap.add_argument("--encoder", required=True,
                    choices=["conformer", "ebranchformer", "ebranchformer_espnet", "zipformer", "moonshine", "moonshine_tiny", "moonshine_official", "dpwavlm"])
    ap.add_argument("--init", default=None,
                    help="pretrained/compressed checkpoint to start from")
    ap.add_argument("--torgo-root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo")
    ap.add_argument("--label-mode", default="intent", choices=["intent", "prompt"],
                    help="class space the Prototypical projector is meta-trained "
                         "on. MUST match the label mode you evaluate with in "
                         "measure.py: 'intent' clusters by the taxonomy, 'prompt' "
                         "by distinct prompt. Training on intent then scoring on "
                         "prompt (or vice-versa) is a train/eval mismatch that "
                         "understates accuracy.")
    ap.add_argument("--holdout", nargs="+", default=[],
                    help="speaker ids held out for eval (speaker-disjoint)")
    ap.add_argument("--dysarthric-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-duration", type=float, default=20.0,
                    help="cap each clip to this many seconds to bound GPU memory (0 = no cap)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop if val loss doesn't improve for this many "
                         "epochs (0 = disabled; runs the full --epochs)")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="minimum *relative* val-loss improvement to count as "
                         "progress, as a fraction of the current best (e.g. "
                         "0.005 = require a 0.5%% drop). 0.0 = any improvement "
                         "counts (original strict behaviour). Recommended here: "
                         "the TORGO holdout is small, so val CTC loss is noisy.")
    ap.add_argument("--lr", type=float, default=None,
                    help="finetune LR; default depends on optimizer "
                         "(AdamW 1e-4, ScaledAdam 0.015 for zipformer)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder-cfg", default=None,
                    help="path to a checkpoint whose meta.encoder_cfg rebuilds "
                         "a materialized DPWavLM student")
    # ---- Prototypical embedding-head meta-training (after CTC finetune) ----- #
    ap.add_argument("--skip-metatrain", action="store_true",
                    help="skip episodic meta-training of the embedding projector "
                         "(leaves it at init — not recommended)")
    ap.add_argument("--embed-epochs", type=int, default=10)
    ap.add_argument("--episodes-per-epoch", type=int, default=200)
    ap.add_argument("--n-way", type=int, default=10)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-query", type=int, default=5)
    ap.add_argument("--embed-lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = CharTokenizer()
    print(f"[finetune] encoder={args.encoder} seed={args.seed} device={device}")

    # ---- data (speaker-disjoint train/eval) ------------------------------- #
    # Label space for the projector's episodic meta-training. Must match the
    # eval label mode (measure.py --label-mode); see --label-mode help above.
    ds = TorgoDataset(args.torgo_root, dysarthric_only=args.dysarthric_only,
                      intent_map=LabelScheme(args.label_mode))
    if not args.holdout:
        # default: hold out ~20% of speakers if none named
        spk = ds.speakers()
        args.holdout = spk[: max(1, len(spk) // 5)]
        print(f"[finetune] no holdout given; holding out {args.holdout}")
    train_idx, eval_idx = speaker_disjoint_split(ds, args.holdout)
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, eval_idx)
    print(f"[finetune] train {len(train_ds)} / eval {len(val_ds)} utts; "
          f"holdout speakers {args.holdout}")

    max_samples = int(args.max_duration * 16000) if args.max_duration else None
    collate = Collate(tokenizer, max_samples=max_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    # ---- model ------------------------------------------------------------ #
    encoder_cfg = None
    if args.encoder == "dpwavlm" and args.encoder_cfg:
        meta = torch.load(args.encoder_cfg, map_location="cpu", weights_only=False)
        encoder_cfg = meta.get("meta", {}).get("encoder_cfg")
    model = build_model(args.encoder, tokenizer, encoder_cfg=encoder_cfg).to(device)

    if args.init:
        # load encoder (+ heads) weights, handling both full-model checkpoints
        # (supervised pretrain) and bare encoder-only checkpoints (DPWavLM
        # distill+prune); raises if nothing matched rather than silently
        # training from random init.
        meta = load_init_weights(args.init, model, map_location=device)
        print(f"[finetune] initialised from {args.init} "
              f"(stage={meta.get('stage','?')})")

    n_params = count_parameters(model.encoder, trainable_only=False)
    print(f"[finetune] encoder params: {n_params/1e6:.2f}M")

    # finetune LR is lower than pretrain; scale each optimizer's default down.
    ft_lr = args.lr if args.lr is not None else (
        0.015 if args.encoder == "zipformer" else 1e-4)
    optimizer, is_scaled = build_optimizer(model, args.encoder, lr=ft_lr)
    print(f"[finetune] optimizer: {'ScaledAdam' if is_scaled else 'AdamW'} "
          f"(lr={optimizer.param_groups[0]['lr']})")
    total_steps = args.epochs * max(1, len(train_loader))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return max(0.02, 0.5 * (1 + torch.cos(torch.tensor(prog * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best, no_improve = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler,
                             device, tokenizer)
        va = validate(model, val_loader, device, tokenizer) if len(val_ds) else tr
        print(f"[finetune] epoch {epoch:3d} | train {tr:7.3f} | "
              f"val {va:7.3f} | {time.time()-t0:6.1f}s")
        # "improved" = val beat the best by at least --min-delta (relative).
        # With min_delta=0 this is the original strict `va < best`.
        if va < best * (1.0 - args.min_delta):
            best, no_improve = va, 0
            save_checkpoint(args.out, model,
                            meta=dict(encoder=args.encoder, seed=args.seed,
                                      epoch=epoch, val_loss=va, stage="finetune",
                                      holdout=args.holdout, encoder_cfg=encoder_cfg))
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:  # early stopping
                print(f"[finetune] early stop: no val improvement in "
                      f"{args.patience} epochs (best val {best:.3f})")
                break

    # reload best CTC checkpoint before meta-training / freezing
    if best < float("inf"):
        load_checkpoint(args.out, model, map_location=device, strict=True)

    # ---- meta-train the Prototypical embedding head (encoder frozen) ------- #
    # CTC trained the encoder; the projector is not on the CTC graph, so it must
    # be trained episodically here or it stays random (see metatrain.py).
    if not args.skip_metatrain:
        from .metatrain import episodic_metatrain
        episodic_metatrain(
            model, ds, train_idx, device,
            epochs=args.embed_epochs, episodes_per_epoch=args.episodes_per_epoch,
            n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
            lr=args.embed_lr, seed=args.seed)
    else:
        model.freeze_encoder()
        print("[finetune] --skip-metatrain set; embedding head left at init.")

    # ---- freeze everything: this is the deployed feature extractor + head -- #
    model.freeze_encoder()
    for p in model.projector.parameters():
        p.requires_grad_(False)
    save_checkpoint(args.out.replace(".pt", "_frozen.pt"), model,
                    meta=dict(encoder=args.encoder, seed=args.seed,
                              stage="frozen", holdout=args.holdout,
                              encoder_cfg=encoder_cfg,
                              metatrained=not args.skip_metatrain))
    print(f"[finetune] done. best val {best:.3f}. "
          f"frozen model -> {args.out.replace('.pt', '_frozen.pt')}")


if __name__ == "__main__":
    main()
