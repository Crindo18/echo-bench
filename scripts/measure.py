#!/usr/bin/env python
"""End-to-end data-hygiene + measurement driver.

One command turns a corpus and a set of trained/frozen encoders into a decision:

  corpus scan  ->  CV plan (nested, severity-stratified, speaker-independent)
               ->  leakage contract (must pass, or we stop)
               ->  enrollment-K coverage
               ->  per-model cross-validated accuracy (+ per-severity tier)
               ->  footprint (INT8 size / budget) + latency (if ONNX present)
               ->  ranked scorecard + recommendation

All artifacts are written as plain JSON under --out so the run is reproducible
and auditable without any database or service.

Example:
  PYTHONPATH=. python scripts/measure.py \
      --corpus torgo --root C:/Users/ASUS/Desktop/ECHO_Datasets/torgo \
      --speaker-table configs/datasets/torgo_speakers.csv \
      --models conformer,ebranchformer,zipformer,moonshine,dpwavlm \
      --ckpt-dir runs --export-dir export \
      --n-folds 5 --seeds 1,2,3 --ks 3,5,10 --out measure
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from echo.data import (CharTokenizer, LabelScheme, scan_corpus,
                       UtteranceDataset, make_cv_plan, verify_no_leakage,
                       print_leakage_report, coverage, print_coverage)
from echo.eval.protocol import cross_validate, print_accuracy
from echo.eval.sweep import metatrain_eval_sweep, print_sweep
from echo.eval.scorecard import (card_from_measurements, build_scorecard,
                                 print_scorecard, weights_without_severity)
from echo.eval.latency import profile_onnx
from echo.encoders import count_parameters
from echo.train.common import get_device
from echo.train.pretrain import build_model


def _load_model(name, ckpt_path, tokenizer, device):
    encoder_cfg = None
    if os.path.isfile(ckpt_path):
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_cfg = meta.get("meta", {}).get("encoder_cfg")
    model = build_model(name, tokenizer, encoder_cfg=encoder_cfg).to(device)
    if os.path.isfile(ckpt_path):
        # strict=False so a benign extra/missing head tensor doesn't abort, but
        # capture the incompatibility and WARN LOUDLY: silently loading a partly
        # random model would misattribute its degraded accuracy to the encoder.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            total = sum(p.numel() for p in model.state_dict().values())
            miss_n = sum(model.state_dict()[k].numel() for k in missing
                         if k in model.state_dict())
            print(f"  [WARN] checkpoint/model mismatch for '{name}': "
                  f"{len(missing)} missing, {len(unexpected)} unexpected tensors "
                  f"(~{100.0*miss_n/max(1,total):.1f}% of params left at init). "
                  f"Accuracy below may NOT reflect the trained encoder — check that "
                  f"the checkpoint and encoder architecture match.")
            if missing:
                print(f"         e.g. missing: {missing[:3]}")
    else:
        print(f"  [warn] no checkpoint at {ckpt_path}; using randomly-init model")
    return model


def _footprint(model, int8_path):
    params_M = count_parameters(model.encoder, trainable_only=False) / 1e6
    int8_mb = float("nan")
    if os.path.isfile(int8_path):
        int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
    return dict(encoder_params_M=params_M, onnx_int8_MB=int8_mb,
                params_within_budget=params_M <= 16.0,
                onnx_within_budget=(int8_mb <= 16.0) if int8_mb == int8_mb else None)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    tok = CharTokenizer()

    # 1. scan + plan + leakage + coverage -----------------------------------
    labeler = LabelScheme(args.label_mode)     # 'intent' taxonomy or 'prompt' (any)
    records = scan_corpus(args.root, args.corpus, args.speaker_table,
                          intent_map=labeler,
                          primary_only=not args.all_channels)
    print(f"[measure] scanned {len(records)} primary utterances "
          f"({len({r.speaker for r in records})} speakers); {labeler.describe()}")
    plan = make_cv_plan(records, n_folds=args.n_folds,
                        seeds=[int(s) for s in args.seeds.split(",")],
                        stratify=not args.no_severity)
    if args.no_severity:
        print("[measure] severity stratification OFF: folds are not balanced by "
              "tier and hardest-tier does not affect the ranking; per-tier "
              "accuracy is still reported where labels exist.")
    plan.to_json(os.path.join(args.out, "cv_plan.json"))

    leak = verify_no_leakage(plan, records)
    print_leakage_report(leak)
    with open(os.path.join(args.out, "leakage.json"), "w") as fh:
        json.dump(leak, fh, indent=2)
    if not leak["passed"]:
        raise SystemExit("[measure] LEAKAGE DETECTED — stopping. See leakage.json")

    ks = [int(k) for k in args.ks.split(",")]
    cov = coverage(records, ks=ks)
    print_coverage(cov)
    with open(os.path.join(args.out, "coverage.json"), "w") as fh:
        json.dump(cov, fh, indent=2)

    ds = UtteranceDataset(records)

    # 2. per-model measurement ----------------------------------------------
    results, cards, sweeps = {}, [], {}
    for name in args.models.split(","):
        name = name.strip()
        print(f"\n[measure] === {name} ===")
        ckpt = os.path.join(args.ckpt_dir, name,
                            f"finetune_seed{args.ckpt_seed}_frozen.pt")
        model = _load_model(name, ckpt, tok, device)

        acc = cross_validate(model, ds, plan, device, ks=ks, distance=args.distance)
        print_accuracy(name, acc)
        results[name] = acc

        # optional comprehensive meta-train-K x eval-K sweep (re-trains the head
        # per fold at each meta-K; leakage-free). Off unless --metatrain-ks given.
        if args.metatrain_ks:
            mks = [int(k) for k in args.metatrain_ks.split(",")]
            sw = metatrain_eval_sweep(model, ds, plan, device,
                                      meta_ks=mks, eval_ks=ks, distance=args.distance)
            print_sweep(sw, name)
            sweeps[name] = sw

        int8 = os.path.join(args.export_dir, f"{name}_int8.onnx")
        fp = _footprint(model, int8)
        lat = None
        if os.path.isfile(int8):
            lat = profile_onnx(int8, seconds=args.latency_seconds, runs=args.latency_runs,
                               accepts_waveform=model.accepts_waveform)
        cards.append(card_from_measurements(name, fp, acc, lat, primary_k=args.primary_k))

    # 3. scorecard -----------------------------------------------------------
    weights = weights_without_severity() if args.no_severity else None
    sc = build_scorecard(cards, weights=weights)
    print_scorecard(sc, primary_k=args.primary_k)

    serialisable = dict(
        weights=sc["weights"], recommendation=sc["recommendation"],
        disqualified=sc["disqualified"],
        models=[dict(name=r["name"], score=r["score"],
                     contributions=r["contributions"], card=vars(r["card"]))
                for r in sc["rows"]],
        accuracy=results,
    )
    with open(os.path.join(args.out, "scorecard.json"), "w") as fh:
        json.dump(serialisable, fh, indent=2, default=str)

    written = "cv_plan, leakage, coverage, scorecard"
    if sweeps:
        with open(os.path.join(args.out, "sweep.json"), "w") as fh:
            json.dump(sweeps, fh, indent=2, default=str)
        written += ", sweep"
    print(f"\n[measure] artifacts written to {args.out}/ ({written}).json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="torgo", choices=["torgo", "uaspeech"])
    ap.add_argument("--root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo",
                    help="corpus root (TORGO/UASpeech). Default points at the ECHO_Datasets folder")
    ap.add_argument("--speaker-table", required=True)
    ap.add_argument("--models", default="conformer,ebranchformer,zipformer,moonshine,dpwavlm")
    ap.add_argument("--ckpt-dir", default="runs")
    ap.add_argument("--ckpt-seed", type=int, default=1)
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--ks", default="3,5,10")
    ap.add_argument("--primary-k", type=int, default=5)
    ap.add_argument("--metatrain-ks", default=None,
                    help="also run a meta-train-K x eval-K sweep (e.g. '3,5,10'): "
                         "for each meta-train K, re-train the head per fold "
                         "(leakage-free) and evaluate at every --ks; writes a "
                         "matrix to sweep.json. Off by default.")
    ap.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--label-mode", default="intent", choices=["intent", "prompt"],
                    help="what the classifier matches against: 'intent' (36-intent "
                         "taxonomy) or 'prompt' (each distinct prompt/word is its "
                         "own class — no taxonomy needed)")
    ap.add_argument("--all-channels", action="store_true",
                    help="keep every microphone (default: primary channel only)")
    ap.add_argument("--no-severity", action="store_true",
                    help="do NOT stratify folds by severity and drop hardest-tier "
                         "from the ranking; per-tier accuracy is still reported "
                         "where speaker severity labels exist")
    ap.add_argument("--latency-seconds", type=float, default=4.0)
    ap.add_argument("--latency-runs", type=int, default=30)
    ap.add_argument("--out", default="measure")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
