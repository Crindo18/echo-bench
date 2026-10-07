"""DPWavLM: joint distillation + structured pruning to the edge budget.

This is the "shrink-big" track and the one calculated gamble (spec Sections 5,
9). It follows the DPHuBERT recipe applied to a WavLM Base+ teacher:

  1. Initialise the student from the teacher (best-effort weight inheritance).
  2. Jointly distill (student hidden states -> teacher targets: L1 + L2 +
     cosine) *and* structurally prune (HardConcrete gates over layers, heads
     and FFN units) under an L0 budget constraint enforced with a Lagrange
     multiplier, targeting <= ~15M params (more aggressive than DPHuBERT's
     ~24M default — the risk-bearing step).
  3. Run the Go/No-Go gate (size / deployability / accuracy-floor).
  4. If Go: ``materialize()`` the compact static student and run a short
     recovery distillation pass.

Only the compressed student then enters the shared finetune -> quantize -> eval
pipeline exactly like the supervised encoders.

Example
-------
    python -m echo.train.distill_prune \
        --data-root C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech --split train-clean-100 \
        --target-params 15e6 --steps 20000 \
        --out runs/dpwavlm/compressed.pt
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import CharTokenizer, LibriSpeechDataset, Collate
from ..encoders import (DPWavLMStudent, WavLMTeacher, init_student_from_teacher,
                        count_parameters)
from .common import AverageMeter, get_device, save_checkpoint, set_seed


# --------------------------------------------------------------------------- #
# Distillation head: map chosen student layers -> teacher layers
# --------------------------------------------------------------------------- #
class DistillProjector(nn.Module):
    """Linear maps from student hidden dim to teacher hidden dim for each
    (student_layer -> teacher_layer) pair being matched. WavLM Base+ has 12
    teacher layers; by default we match a spread of student layers to teacher
    layers 4/8/12 (indices 3/7/11) plus the final layer, following the
    multi-layer distillation used in DPHuBERT."""

    def __init__(self, student_dim: int, teacher_dim: int, n_pairs: int):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Linear(student_dim, teacher_dim) for _ in range(n_pairs)]
        )

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]


def distill_loss(student_feats, teacher_feats, lengths=None):
    """L1 + L2 + (1 - cosine), averaged over matched layer pairs and time.
    This is the standard hidden-state regression objective used to transfer the
    teacher's representation into the pruned student.

    ``lengths`` (valid frames per item) masks padded frames so the teacher's
    activations over padding — the WavLM teacher is run unmasked, since its
    attention cannot take a mask — do not contaminate the target.
    """
    total = 0.0
    for s, t in zip(student_feats, teacher_feats):
        T = min(s.size(1), t.size(1))
        s, t = s[:, :T], t[:, :T]
        if lengths is not None:
            m = (torch.arange(T, device=s.device)[None, :]
                 < lengths.to(s.device).clamp_max(T)[:, None])       # [B, T]
            mf = m.unsqueeze(-1).float()
            denom = mf.sum().clamp_min(1.0)
            l1 = (F.l1_loss(s, t, reduction="none") * mf).sum() / (denom * s.size(-1))
            l2 = (F.mse_loss(s, t, reduction="none") * mf).sum() / (denom * s.size(-1))
            cos_frame = F.cosine_similarity(s, t, dim=-1)            # [B, T]
            cos = 1.0 - (cos_frame * m.float()).sum() / m.float().sum().clamp_min(1.0)
        else:
            l1 = F.l1_loss(s, t)
            l2 = F.mse_loss(s, t)
            cos = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        total = total + l1 + l2 + cos
    return total / max(1, len(student_feats))


# --------------------------------------------------------------------------- #
# Layer-pair selection
# --------------------------------------------------------------------------- #
def pick_layer_pairs(n_student_layers: int, n_teacher_layers: int):
    """Match evenly-spaced student layers to evenly-spaced teacher layers,
    always including the final layer of each. Returns list of (s_idx, t_idx)."""
    k = min(4, n_student_layers, n_teacher_layers)
    s_idx = torch.linspace(0, n_student_layers - 1, k).round().long().tolist()
    t_idx = torch.linspace(0, n_teacher_layers - 1, k).round().long().tolist()
    return list(zip(s_idx, t_idx))


# --------------------------------------------------------------------------- #
# Go/No-Go gate (spec Section 5)
# --------------------------------------------------------------------------- #
def go_no_go_gate(student: DPWavLMStudent, size_mb_limit: float = 16.0,
                  param_limit: float = 15e6, verbose: bool = True) -> dict:
    """Cheap, early check run *before* committing weeks to DPWavLM.

    We evaluate the two objective, offline-checkable gates here:
      * Size gate: materialised param count implies <= 16 MB INT8 ONNX
        (INT8 ~1 byte/param, so params-in-millions ~= MB).
      * Deployability gate: the materialised student runs a forward pass and is
        structurally exportable (a full ONNX export + Pi-5 latency check is done
        by ``echo.quantize`` / ``echo.eval.latency`` on the target device).
    The accuracy floor is assessed downstream on a held-out check; we surface a
    placeholder hook so the caller can plug in that number.
    """
    materialized = student.materialize()
    n_params = count_parameters(materialized, trainable_only=False)
    est_mb = n_params / 1e6  # INT8 ~ 1 byte/param

    # deployability smoke test: a forward pass must succeed
    deployable = True
    try:
        materialized.eval()
        with torch.no_grad():
            wav = torch.randn(1, 16000)
            lengths = torch.tensor([16000])
            materialized(wav, lengths)
    except Exception as exc:  # pragma: no cover - reported, not raised
        deployable = False
        if verbose:
            print(f"[gate] deployability forward pass FAILED: {exc}")

    size_ok = (n_params <= param_limit) and (est_mb <= size_mb_limit)
    decision = "GO" if (size_ok and deployable) else "NO-GO"
    result = dict(decision=decision, n_params=int(n_params), est_mb=est_mb,
                  size_ok=bool(size_ok), deployable=bool(deployable),
                  n_layers=materialized.cfg["n_layers"])
    if verbose:
        print(f"[gate] materialized: {n_params/1e6:.2f}M params "
              f"(~{est_mb:.1f} MB INT8), layers={materialized.cfg['n_layers']}")
        print(f"[gate] size_ok={size_ok} deployable={deployable} -> {decision}")
        if decision == "NO-GO":
            print("[gate] Per spec Section 5: do NOT sink weeks in. Report at "
                  "the smallest reachable size as an out-of-budget reference "
                  "row, or drop DPWavLM. The other three still stand.")
    return result


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def distill_prune(
    data_root: str, split: str, out: str,
    target_params: float = 15e6,
    steps: int = 20000, recovery_steps: int = 4000,
    batch_size: int = 8, max_duration: float = 20.0,
    lr: float = 2e-4, gate_lr: float = 2e-2,
    lambda_lr: float = 1.0, seed: int = 1, no_download: bool = False,
    allow_random_teacher: bool = False,
    log_every: int = 50,
):
    set_seed(seed)
    device = get_device()
    tokenizer = CharTokenizer()  # only for the shared Collate signature
    print(f"[dpwavlm] device={device} target={target_params/1e6:.1f}M "
          f"steps={steps}")

    # ---- teacher + student ------------------------------------------------ #
    teacher = WavLMTeacher(pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    # Guard: WavLMTeacher silently falls back to a RANDOM-init backbone if the
    # pretrained WavLM Base+ download fails. Distilling toward a random teacher
    # is meaningless yet would still pass the size/deployability gate, so refuse
    # unless the user explicitly opted into a no-teacher smoke run.
    if not teacher.pretrained and not allow_random_teacher:
        raise RuntimeError(
            "WavLM Base+ pretrained weights did not load, so the teacher is "
            "randomly initialised -- distillation would inherit nothing. Fix the "
            "download (network/torchaudio) and re-run, or pass "
            "--allow-random-teacher for a deliberate pipeline smoke test "
            "(the result is NOT a valid DPWavLM model)."
        )
    student = DPWavLMStudent().to(device)          # Base+ dims, gated
    # strict inheritance: abort if the student would end up mostly random because
    # torchaudio's WavLM key names don't match (see init_student_from_teacher).
    init_student_from_teacher(student, teacher, strict=not allow_random_teacher)
    dense_params = count_parameters(student, trainable_only=False)
    print(f"[dpwavlm] dense student: {dense_params/1e6:.2f}M "
          f"(target {target_params/1e6:.1f}M)")

    pairs = pick_layer_pairs(student.cfg["n_layers"],
                             teacher_layers := 12)
    projector = DistillProjector(student.cfg["d_model"], 768, len(pairs)).to(device)
    s_layer_idx = [s for s, _ in pairs]
    t_layer_idx = [t for _, t in pairs]
    print(f"[dpwavlm] distilling student layers {s_layer_idx} -> "
          f"teacher layers {t_layer_idx}")

    # ---- data ------------------------------------------------------------- #
    ds = LibriSpeechDataset(data_root, url=split, download=not no_download)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4,
                        collate_fn=Collate(None, max_samples=int(max_duration*16000) if max_duration else None),
                        drop_last=True)

    # ---- optimisers: separate main weights, gates, projector -------------- #
    main_opt = torch.optim.AdamW(
        list(student.main_parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-4)
    gate_opt = torch.optim.AdamW(student.gate_parameters(), lr=gate_lr)
    # Lagrange multiplier for the param-budget constraint (ascent).
    lam = torch.zeros((), device=device, requires_grad=True)
    lam_opt = torch.optim.Adam([lam], lr=lambda_lr, maximize=True)

    def run_phase(n_steps: int, prune: bool, tag: str):
        student.train()
        meter, sparsity_meter = AverageMeter(), AverageMeter()
        step = 0
        while step < n_steps:
            for batch in loader:
                if step >= n_steps:
                    break
                wav = batch["wav"].to(device)
                wav_lengths = batch["wav_lengths"].to(device)
                with torch.no_grad():
                    t_feats, _ = teacher.hidden_states(wav, wav_lengths)
                    t_sel = [t_feats[i] for i in t_layer_idx]
                _, s_out_len, s_hiddens = student(wav, wav_lengths, return_hidden=True)
                s_sel = projector([s_hiddens[i] for i in s_layer_idx])
                d_loss = distill_loss(s_sel, t_sel, lengths=s_out_len)

                if prune:
                    exp_params = student.expected_num_params()
                    # constraint: exp_params <= target  ->  violation >= 0
                    violation = (exp_params - target_params) / dense_params
                    loss = d_loss + lam.detach() * violation
                else:
                    violation = torch.zeros((), device=device)
                    loss = d_loss

                main_opt.zero_grad()
                if prune:
                    gate_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                main_opt.step()
                if prune:
                    gate_opt.step()
                    # Lagrange ascent on the (recomputed) violation
                    lam_opt.zero_grad()
                    exp_params2 = student.expected_num_params()
                    lam_viol = (exp_params2.detach() - target_params) / dense_params
                    (lam * lam_viol).backward()
                    lam_opt.step()
                    lam.data.clamp_(min=0.0)
                    with torch.no_grad():
                        cur = student.expected_num_params().item()
                    sparsity_meter.update(cur)

                meter.update(d_loss.item(), wav.size(0))
                if step % log_every == 0:
                    msg = (f"[{tag}] step {step:5d} | distill {d_loss.item():6.3f} "
                           f"| avg {meter.avg:6.3f}")
                    if prune:
                        msg += (f" | Eparams {sparsity_meter.avg/1e6:5.2f}M "
                                f"| lam {lam.item():6.3f}")
                    print(msg)
                step += 1
        return meter.avg

    # ---- Phase 1: joint distill + prune ----------------------------------- #
    t0 = time.time()
    run_phase(steps, prune=True, tag="prune")
    print(f"[dpwavlm] prune phase done in {time.time()-t0:.0f}s")

    # ---- Go/No-Go gate ---------------------------------------------------- #
    gate = go_no_go_gate(student, param_limit=target_params)
    save_checkpoint(out.replace(".pt", "_gated_student.pt"), student,
                    meta=dict(stage="pre_materialize", gate=gate, seed=seed))
    if gate["decision"] == "NO-GO":
        print("[dpwavlm] gate NO-GO — stopping before recovery. See spec §5.")
        return gate

    # ---- Materialize + recovery distillation ------------------------------ #
    compact = student.materialize().to(device)
    print(f"[dpwavlm] materialized to "
          f"{count_parameters(compact, trainable_only=False)/1e6:.2f}M; "
          f"running {recovery_steps}-step recovery distillation")

    # rebuild projector/pairs for the (smaller) compact student and recover
    pairs2 = pick_layer_pairs(compact.cfg["n_layers"], teacher_layers)
    projector = DistillProjector(compact.cfg["d_model"], 768, len(pairs2)).to(device)
    s_layer_idx = [s for s, _ in pairs2]
    t_layer_idx = [t for _, t in pairs2]
    student = compact  # reuse run_phase closure vars via reassignment
    main_opt = torch.optim.AdamW(
        list(student.main_parameters()) + list(projector.parameters()),
        lr=lr * 0.5, weight_decay=1e-4)
    run_phase(recovery_steps, prune=False, tag="recover")

    final_params = count_parameters(student, trainable_only=False)
    save_checkpoint(out, student,
                    meta=dict(stage="compressed", seed=seed,
                              n_params=int(final_params),
                              encoder_cfg=dict(student.cfg),
                              gate=gate))
    print(f"[dpwavlm] done. compressed student {final_params/1e6:.2f}M -> {out}")
    print("[dpwavlm] next: python -m echo.train.finetune --encoder dpwavlm "
          f"--init {out} --encoder-cfg {out} ...")
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(description="DPWavLM distill+prune to budget")
    ap.add_argument("--data-root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech")
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--target-params", type=float, default=15e6)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--recovery-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-duration", type=float, default=20.0,
                    help="cap each clip to this many seconds to bound GPU memory")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--allow-random-teacher", action="store_true",
                    help="proceed even if pretrained WavLM Base+ fails to load "
                         "(teacher/student would be random) -- pipeline smoke "
                         "test only; the result is NOT a valid DPWavLM model.")
    args = ap.parse_args()
    distill_prune(
        data_root=args.data_root, split=args.split, out=args.out,
        target_params=args.target_params, steps=args.steps,
        recovery_steps=args.recovery_steps, batch_size=args.batch_size,
        max_duration=args.max_duration,
        lr=args.lr, seed=args.seed, no_download=args.no_download,
        allow_random_teacher=args.allow_random_teacher,
    )


if __name__ == "__main__":
    main()
