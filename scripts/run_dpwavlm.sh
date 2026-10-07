#!/usr/bin/env bash
# DPWavLM track: compress WavLM Base+ to the edge budget, run the Go/No-Go gate,
# then (if GO) finetune -> quantize -> eval like the others (spec §5).
#
# GPU memory: BATCH / MAX_DUR tuned for ~6 GB VRAM (safe on 8 GB). Override, e.g.:
#     BATCH=4 MAX_DUR=12 scripts/run_dpwavlm.sh 1
set -euo pipefail

SEED="${1:-1}"
RUN="runs/dpwavlm"
mkdir -p "${RUN}" export
LIBRISPEECH_ROOT="C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech"
TORGO_ROOT="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo"
HOLDOUT="F01 M01 F03 M03"

BATCH="${BATCH:-6}"
MAX_DUR="${MAX_DUR:-16}"
FT_EPOCHS="${FT_EPOCHS:-30}"
PATIENCE="${PATIENCE:-0}"     # early-stop finetune after N no-improve epochs (0=off)
MIN_DELTA="${MIN_DELTA:-0}"  # min relative val-loss drop to count (0=strict; noisy TORGO wants ~0.005)
LABEL_MODE="${LABEL_MODE:-intent}"  # projector class space; MUST match measure.py --label-mode
echo "[run_dpwavlm] seed=${SEED} | batch=${BATCH} max_dur=${MAX_DUR}s | ft_epochs=${FT_EPOCHS} patience=${PATIENCE} min_delta=${MIN_DELTA} label_mode=${LABEL_MODE}"

echo "== [1/4] distill + structured-prune WavLM Base+ -> <=15M (gated) =="
python -m echo.train.distill_prune \
  --data-root "${LIBRISPEECH_ROOT}" --split train-clean-100 \
  --batch-size "${BATCH}" --max-duration "${MAX_DUR}" \
  --target-params 15e6 --steps 20000 --recovery-steps 4000 \
  --seed "${SEED}" --out "${RUN}/compressed.pt"
# If the Go/No-Go gate prints NO-GO, stop here (see spec §5) — the other three
# models still form a complete benchmark.

echo "== [2/4] finetune compressed student on TORGO, then freeze =="
python -m echo.train.finetune --encoder dpwavlm \
  --init "${RUN}/compressed.pt" --encoder-cfg "${RUN}/compressed.pt" \
  --batch-size "${BATCH}" --max-duration "${MAX_DUR}" \
  --torgo-root "${TORGO_ROOT}" --holdout ${HOLDOUT} \
  --epochs "${FT_EPOCHS}" --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
  --label-mode "${LABEL_MODE}" \
  --seed "${SEED}" --out "${RUN}/finetune_seed${SEED}.pt"

echo "== [3/4] INT8 quantize + ONNX export =="
python -m echo.quantize.quantize_export --encoder dpwavlm \
  --checkpoint "${RUN}/finetune_seed${SEED}_frozen.pt" --out-dir export

echo "== [4/4] few-shot enrollment benchmark + latency =="
python -m echo.eval.benchmark --encoder dpwavlm \
  --checkpoint "${RUN}/finetune_seed${SEED}_frozen.pt" \
  --torgo-root "${TORGO_ROOT}" --holdout ${HOLDOUT} --seed "${SEED}"
python -m echo.eval.latency --onnx "export/dpwavlm_int8.onnx"
