"""CTC pretraining on LibriSpeech (typical speech).

Stage 1 of the supervised track (spec Section 6): each of Conformer,
E-Branchformer and Zipformer is pretrained on the *same* typical-speech corpus
with the *same* recipe and the *same* ~15M budget, so the only difference
between them is the block design. DPWavLM does not use this script — it
inherits WavLM Base+ pretraining and is compressed instead (see
``distill_prune.py``).

Example
-------
    python -m echo.train.pretrain --encoder ebranchformer \
        --data-root C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech --split train-clean-100 \
        --epochs 30 --out runs/ebranchformer/pretrain.pt --seed 1
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from ..data import CharTokenizer, Collate, LibriSpeechDataset
from ..encoders import build_encoder, count_parameters
from .common import (
    AcousticModel,
    AverageMeter,
    build_optimizer,
    get_device,
    save_checkpoint,
    set_seed,
)


def build_model(
    encoder_name: str, tokenizer: CharTokenizer, encoder_cfg: dict | None = None
) -> AcousticModel:
    encoder = build_encoder(encoder_name, **(encoder_cfg or {}))
    return AcousticModel(encoder, vocab_size=tokenizer.vocab_size)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    tokenizer,
    grad_clip: float = 5.0,
    log_every: int = 50,
) -> float:
    model.train()
    meter = AverageMeter()
    n_bad = 0
    for step, batch in enumerate(loader):
        wav = batch["wav"].to(device)
        wav_lengths = batch["wav_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # drop degenerate items whose transcript encoded to empty (CTC needs a
        # non-empty target). targets is the flattened concat of per-item label
        # sequences, so filter by splitting on target_lengths and re-concatenating
        # the kept segments — not just computing a mask and ignoring it.
        if (target_lengths == 0).any():
            keep = target_lengths > 0
            if keep.sum() == 0:
                continue
            segs = torch.split(targets, target_lengths.tolist())
            targets = torch.cat([s for s, k in zip(segs, keep) if k])
            wav, wav_lengths = wav[keep], wav_lengths[keep]
            target_lengths = target_lengths[keep]

        _, loss = model.forward_ctc(wav, wav_lengths, targets, target_lengths)
        # Skip non-finite steps instead of back-propagating them. A single NaN/Inf
        # loss would write NaN grads -> NaN weights -> every later loss AND the
        # running avg NaN forever. Zipformer (ScaledAdam at a high LR) is the usual
        # culprit; if these skips are frequent, lower its LR or lengthen --warmup
        # rather than leaning on this guard. The scheduler still advances so the LR
        # trajectory stays on schedule.
        if not torch.isfinite(loss):
            n_bad += 1
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            continue
        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gnorm):  # exploded/NaN grads: don't corrupt weights
            n_bad += 1
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        meter.update(loss.item(), wav.size(0))
        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  step {step:5d} | loss {loss.item():7.3f} | "
                f"avg {meter.avg:7.3f} | lr {lr:.2e}"
            )
    if n_bad:
        print(
            f"  [warn] skipped {n_bad} non-finite (NaN/Inf) step(s) this epoch. "
            f"If frequent, lower the LR (esp. Zipformer/ScaledAdam) or raise "
            f"--warmup."
        )
    return meter.avg


@torch.no_grad()
def validate(model, loader, device, tokenizer) -> float:
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        wav = batch["wav"].to(device)
        wav_lengths = batch["wav_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        if (target_lengths == 0).all():
            continue
        _, loss = model.forward_ctc(wav, wav_lengths, targets, target_lengths)
        if not torch.isfinite(loss):  # don't let a bad batch make the val avg NaN
            continue  # (best-checkpoint selection depends on it)
        meter.update(loss.item(), wav.size(0))
    return meter.avg


def main() -> None:
    ap = argparse.ArgumentParser(description="CTC pretrain on LibriSpeech")
    ap.add_argument(
        "--encoder",
        required=True,
        choices=[
            "conformer",
            "ebranchformer",
            "ebranchformer_espnet",
            "zipformer",
            "moonshine",
            "moonshine_tiny",
            "moonshine_official",
        ],
    )
    ap.add_argument(
        "--data-root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech"
    )
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--val-split", default="dev-clean")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="cap each clip to this many seconds to bound GPU memory (0 = no cap)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early stop if val loss doesn't improve for this many "
        "epochs (0 = disabled; runs the full --epochs)",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="minimum *relative* val-loss improvement to count as "
        "progress, as a fraction of the current best (e.g. "
        "0.005 = require a 0.5%% drop). 0.0 = any improvement "
        "counts, reproducing the original strict behaviour. "
        "Useful on noisy val sets so trivial downticks don't "
        "keep resetting --patience.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="LR; default depends on optimizer (AdamW 3e-4, "
        "ScaledAdam 0.04 for zipformer)",
    )
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = CharTokenizer()

    print(f"[pretrain] encoder={args.encoder} seed={args.seed} device={device}")
    train_ds = LibriSpeechDataset(
        args.data_root, url=args.split, download=not args.no_download
    )
    val_ds = LibriSpeechDataset(
        args.data_root, url=args.val_split, download=not args.no_download
    )
    max_samples = int(args.max_duration * 16000) if args.max_duration else None
    collate = Collate(tokenizer, max_samples=max_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model = build_model(args.encoder, tokenizer).to(device)
    n_params = count_parameters(model.encoder, trainable_only=False)
    print(f"[pretrain] encoder params: {n_params / 1e6:.2f}M")

    # Zipformer -> ScaledAdam (its native optimizer); others -> AdamW.
    lr_override = args.lr if args.lr is not None else None
    optimizer, is_scaled = build_optimizer(model, args.encoder, lr=lr_override)
    print(
        f"[pretrain] optimizer: {'ScaledAdam' if is_scaled else 'AdamW'} "
        f"(lr={optimizer.param_groups[0]['lr']})"
    )
    total_steps = args.epochs * max(1, len(train_loader))

    def lr_lambda(step):  # linear warmup then cosine-ish decay
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return max(0.02, 0.5 * (1 + torch.cos(torch.tensor(prog * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best, no_improve = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, tokenizer
        )
        va = validate(model, val_loader, device, tokenizer)
        dt = time.time() - t0
        print(
            f"[pretrain] epoch {epoch:3d} | train {tr:7.3f} | "
            f"val {va:7.3f} | {dt:6.1f}s"
        )
        # "improved" = val beat the best by at least --min-delta (relative).
        # With min_delta=0 this is the original strict `va < best`. best starts
        # at +inf, so the first finite epoch always counts.
        if va < best * (1.0 - args.min_delta):
            best, no_improve = va, 0
            save_checkpoint(
                args.out,
                model,
                meta=dict(
                    encoder=args.encoder,
                    seed=args.seed,
                    epoch=epoch,
                    val_loss=va,
                    stage="pretrain",
                ),
                optimizer=optimizer,
            )
            print(f"[pretrain] saved best -> {args.out} (val {va:.3f})")
        else:
            no_improve += 1
            # early stopping: stop if val loss hasn't improved for --patience
            # epochs (0 disables it). The best checkpoint is already saved.
            if args.patience and no_improve >= args.patience:
                print(
                    f"[pretrain] early stop: no val improvement in "
                    f"{args.patience} epochs (best val {best:.3f})"
                )
                break
    print(f"[pretrain] done. best val loss {best:.3f}")


if __name__ == "__main__":
    main()
