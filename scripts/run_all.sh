#!/usr/bin/env bash
# End-to-end benchmark driver for the supervised encoders.
# DPWavLM has its own entrypoint (run_dpwavlm.sh) because it compresses a
# teacher instead of pretraining from scratch (spec §5).
#
# Usage: scripts/run_all.sh <encoder> [seed]
#   encoder in {conformer, ebranchformer, ebranchformer_espnet, zipformer,
#               moonshine, moonshine_tiny, moonshine_official}
#
# GPU memory: BATCH and MAX_DUR below are tuned for ~6 GB of VRAM (safe on an
# 8 GB card). Peak memory scales with BATCH x longest-clip, so both are capped.
# Override per run without editing the file, e.g.:
#     BATCH=4 MAX_DUR=12 scripts/run_all.sh conformer 1     # for <6 GB / OOM
#     BATCH=16 scripts/run_all.sh conformer 1               # for a big GPU
set -euo pipefail

ENC="${1:?encoder required: conformer|ebranchformer|ebranchformer_espnet|zipformer|moonshine|moonshine_tiny|moonshine_official}"
SEED="${2:-1}"
RUN="runs/${ENC}"
mkdir -p "${RUN}" export

# Edit these to your data locations / speaker split.
LIBRISPEECH_ROOT="C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech"
TORGO_ROOT="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo"
HOLDOUT="F01 M01 F03 M03"     # speaker-disjoint eval speakers

# Training knobs (all env-overridable, no file edit needed):
#   BATCH      per-step batch size            MAX_DUR   cap each clip to N seconds
#   PRE_EPOCHS pretrain epochs                FT_EPOCHS finetune epochs
#   PATIENCE   early-stop after N epochs with no val improvement (0 = off)
#   MIN_DELTA  min relative val-loss drop to count as improvement (0 = strict).
#              Only meaningful with PATIENCE>0; noisy val sets want ~0.005.
# e.g.  PRE_EPOCHS=15 FT_EPOCHS=20 PATIENCE=5 MIN_DELTA=0.005 scripts/run_all.sh conformer 1
BATCH="${BATCH:-6}"
MAX_DUR="${MAX_DUR:-16}"
PRE_EPOCHS="${PRE_EPOCHS:-30}"
FT_EPOCHS="${FT_EPOCHS:-30}"
PATIENCE="${PATIENCE:-0}"
MIN_DELTA="${MIN_DELTA:-0}"
# Class space for the projector. MUST match measure.py --label-mode, or the
# embedding head is trained on one label space and scored on another.
LABEL_MODE="${LABEL_MODE:-intent}"
echo "[run_all] ${ENC} seed=${SEED} | batch=${BATCH} max_dur=${MAX_DUR}s | epochs=${PRE_EPOCHS}/${FT_EPOCHS} patience=${PATIENCE} min_delta=${MIN_DELTA} label_mode=${LABEL_MODE}"

echo "== [1/4] pretrain ${ENC} on LibriSpeech (typical speech) =="
python -m echo.train.pretrain --encoder "${ENC}" \
  --data-root "${LIBRISPEECH_ROOT}" --split train-clean-100 \
  --batch-size "${BATCH}" --max-duration "${MAX_DUR}" \
  --epochs "${PRE_EPOCHS}" --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
  --seed "${SEED}" --out "${RUN}/pretrain.pt"

echo "== [2/4] finetune ${ENC} on TORGO (atypical speech), then freeze =="
python -m echo.train.finetune --encoder "${ENC}" \
  --init "${RUN}/pretrain.pt" --torgo-root "${TORGO_ROOT}" \
  --batch-size "${BATCH}" --max-duration "${MAX_DUR}" \
  --holdout ${HOLDOUT} --epochs "${FT_EPOCHS}" --patience "${PATIENCE}" \
  --min-delta "${MIN_DELTA}" --label-mode "${LABEL_MODE}" \
  --seed "${SEED}" --out "${RUN}/finetune_seed${SEED}.pt"

echo "== [3/4] INT8 quantize + ONNX export =="
python -m echo.quantize.quantize_export --encoder "${ENC}" \
  --checkpoint "${RUN}/finetune_seed${SEED}_frozen.pt" \
  --out-dir export

echo "== [4/4] few-shot enrollment benchmark + latency =="
python -m echo.eval.benchmark --encoder "${ENC}" \
  --checkpoint "${RUN}/finetune_seed${SEED}_frozen.pt" \
  --torgo-root "${TORGO_ROOT}" --holdout ${HOLDOUT} --seed "${SEED}"
python -m echo.eval.latency --onnx "export/${ENC}_int8.onnx"
