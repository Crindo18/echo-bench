#!/usr/bin/env pwsh
# PowerShell port of run_dpwavlm.sh -- the shrink-big arm: compress WavLM Base+
# to the edge budget, run the Go/No-Go gate, then (if GO) finetune -> quantize
# -> eval like the others (spec Section 5).
#
# Usage:  .\scripts\run_dpwavlm.ps1 [seed]
# Knobs are env-overridable, e.g.:
#     $env:BATCH=4; $env:MAX_DUR=12; .\scripts\run_dpwavlm.ps1 1
# If .ps1 execution is blocked:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
param(
    [int]$Seed = 1
)
$ErrorActionPreference = "Stop"

$Run = "runs\dpwavlm"
New-Item -ItemType Directory -Force -Path $Run, "export" | Out-Null

$LibriRoot = "C:\Users\ASUS\Desktop\ECHO_Datasets\librispeech"
$TorgoRoot = "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo"
$Holdout   = @("F01","M01","F03","M03")

$Batch    = if ($env:BATCH)     { $env:BATCH }     else { "6" }
$MaxDur   = if ($env:MAX_DUR)   { $env:MAX_DUR }   else { "16" }
$FtEpochs = if ($env:FT_EPOCHS) { $env:FT_EPOCHS } else { "30" }
$Patience = if ($env:PATIENCE)  { $env:PATIENCE }  else { "0" }
$MinDelta = if ($env:MIN_DELTA) { $env:MIN_DELTA } else { "0" }
$LabelMode = if ($env:LABEL_MODE) { $env:LABEL_MODE } else { "intent" }  # match measure.py --label-mode
Write-Host "[run_dpwavlm] seed=$Seed | batch=$Batch max_dur=${MaxDur}s | ft_epochs=$FtEpochs patience=$Patience min_delta=$MinDelta label_mode=$LabelMode"

function Assert-Ok($Label) {
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

Write-Host "== [1/4] distill + structured-prune WavLM Base+ -> <=15M (gated) =="
python -m echo.train.distill_prune `
  --data-root $LibriRoot --split train-clean-100 `
  --batch-size $Batch --max-duration $MaxDur `
  --target-params 15e6 --steps 20000 --recovery-steps 4000 `
  --seed $Seed --out "$Run\compressed.pt"
Assert-Ok "distill_prune"
# If the Go/No-Go gate prints NO-GO, stop here (spec Section 5) -- the other
# three models still form a complete benchmark.

Write-Host "== [2/4] finetune compressed student on TORGO, then freeze =="
python -m echo.train.finetune --encoder dpwavlm `
  --init "$Run\compressed.pt" --encoder-cfg "$Run\compressed.pt" `
  --batch-size $Batch --max-duration $MaxDur `
  --torgo-root $TorgoRoot --holdout $Holdout `
  --epochs $FtEpochs --patience $Patience --min-delta $MinDelta `
  --label-mode $LabelMode `
  --seed $Seed --out "$Run\finetune_seed${Seed}.pt"
Assert-Ok "finetune"

Write-Host "== [3/4] INT8 quantize + ONNX export =="
python -m echo.quantize.quantize_export --encoder dpwavlm `
  --checkpoint "$Run\finetune_seed${Seed}_frozen.pt" --out-dir export
Assert-Ok "quantize"

Write-Host "== [4/4] few-shot enrollment benchmark + latency =="
python -m echo.eval.benchmark --encoder dpwavlm `
  --checkpoint "$Run\finetune_seed${Seed}_frozen.pt" `
  --torgo-root $TorgoRoot --holdout $Holdout --seed $Seed
Assert-Ok "benchmark"
python -m echo.eval.latency --onnx "export\dpwavlm_int8.onnx"
Assert-Ok "latency"
