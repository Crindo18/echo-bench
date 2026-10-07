#!/usr/bin/env pwsh
# PowerShell port of run_all.sh -- end-to-end driver for the supervised encoders.
# (DPWavLM has its own entrypoint, run_dpwavlm.ps1.)
#
# Usage:  .\scripts\run_all.ps1 <encoder> [seed]
#   encoder in {conformer, ebranchformer, ebranchformer_espnet, zipformer,
#               moonshine, moonshine_tiny, moonshine_official}
#
# Knobs are env-overridable, same names/defaults as the bash driver, e.g.:
#     $env:BATCH=4; $env:MAX_DUR=12; .\scripts\run_all.ps1 conformer 1   # <6 GB / OOM
#     $env:PATIENCE=5; $env:MIN_DELTA=0.005; .\scripts\run_all.ps1 conformer 1
# If .ps1 execution is blocked:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("conformer","ebranchformer","ebranchformer_espnet","zipformer",
                 "moonshine","moonshine_tiny","moonshine_official")]
    [string]$Encoder,
    [int]$Seed = 1
)
$ErrorActionPreference = "Stop"   # stop on PowerShell errors (native exit codes checked below)

$Run = "runs\$Encoder"
New-Item -ItemType Directory -Force -Path $Run, "export" | Out-Null

# Edit these to your data locations / speaker split.
$LibriRoot = "C:\Users\ASUS\Desktop\ECHO_Datasets\librispeech"
$TorgoRoot = "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo"
$Holdout   = @("F01","M01","F03","M03")   # speaker-disjoint eval speakers

# env-overridable knobs (same defaults as run_all.sh)
$Batch     = if ($env:BATCH)      { $env:BATCH }      else { "6" }
$MaxDur    = if ($env:MAX_DUR)    { $env:MAX_DUR }    else { "16" }
$PreEpochs = if ($env:PRE_EPOCHS) { $env:PRE_EPOCHS } else { "30" }
$FtEpochs  = if ($env:FT_EPOCHS)  { $env:FT_EPOCHS }  else { "30" }
$Patience  = if ($env:PATIENCE)   { $env:PATIENCE }   else { "0" }
$MinDelta  = if ($env:MIN_DELTA)  { $env:MIN_DELTA }  else { "0" }
# Projector class space; MUST match measure.py --label-mode.
$LabelMode = if ($env:LABEL_MODE) { $env:LABEL_MODE } else { "intent" }
Write-Host "[run_all] $Encoder seed=$Seed | batch=$Batch max_dur=${MaxDur}s | epochs=$PreEpochs/$FtEpochs patience=$Patience min_delta=$MinDelta label_mode=$LabelMode"

# mirror bash 'set -e' for native commands: throw on any non-zero python exit.
function Assert-Ok($Label) {
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

Write-Host "== [1/4] pretrain $Encoder on LibriSpeech (typical speech) =="
python -m echo.train.pretrain --encoder $Encoder `
  --data-root $LibriRoot --split train-clean-100 `
  --batch-size $Batch --max-duration $MaxDur `
  --epochs $PreEpochs --patience $Patience --min-delta $MinDelta `
  --seed $Seed --out "$Run\pretrain.pt"
Assert-Ok "pretrain"

Write-Host "== [2/4] finetune $Encoder on TORGO (atypical speech), then freeze =="
python -m echo.train.finetune --encoder $Encoder `
  --init "$Run\pretrain.pt" --torgo-root $TorgoRoot `
  --batch-size $Batch --max-duration $MaxDur `
  --holdout $Holdout --epochs $FtEpochs --patience $Patience `
  --min-delta $MinDelta --label-mode $LabelMode `
  --seed $Seed --out "$Run\finetune_seed${Seed}.pt"
Assert-Ok "finetune"

Write-Host "== [3/4] INT8 quantize + ONNX export =="
python -m echo.quantize.quantize_export --encoder $Encoder `
  --checkpoint "$Run\finetune_seed${Seed}_frozen.pt" --out-dir export
Assert-Ok "quantize"

Write-Host "== [4/4] few-shot enrollment benchmark + latency =="
python -m echo.eval.benchmark --encoder $Encoder `
  --checkpoint "$Run\finetune_seed${Seed}_frozen.pt" `
  --torgo-root $TorgoRoot --holdout $Holdout --seed $Seed
Assert-Ok "benchmark"
python -m echo.eval.latency --onnx "export\${Encoder}_int8.onnx"
Assert-Ok "latency"
