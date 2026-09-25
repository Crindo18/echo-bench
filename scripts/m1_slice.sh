#!/usr/bin/env bash
# The whole M1 slice in one command (blueprint section 7.2, M1 exit criterion), desktop only:
#   model recipe -> random-weight model -> ONNX FP32 + INT8 -> register (gates G1, G2a)
#   -> perf (latency, memory) -> report (table + figure)
#
# Run from the repository root:
#   scripts/m1_slice.sh                                      # perf settings: perf_smoke.yaml
#   scripts/m1_slice.sh configs/benchmarks/perf_full.yaml    # the full protocol (slower)
#   MODEL_CONFIG=configs/models/my_recipe.yaml scripts/m1_slice.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PERF_CONFIG="${1:-configs/benchmarks/perf_smoke.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/ebf12m.yaml}"
NAME="$(awk '/^name:/ {print $2; exit}' "$MODEL_CONFIG")-rand"
OUT="artifacts/models/$NAME"
PINNED="results/m1_slice_$(basename "$PERF_CONFIG")"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1/5 Build, export and quantize: $MODEL_CONFIG"
(cd packages/echo-train && uv run echo-train build-random --config "../../$MODEL_CONFIG" --out "../../$OUT")

step "2/5 Results database"
uv run echo-bench db init

step "3/5 Register the model files (records gates G1 and G2a)"
uv run echo-bench model register "$OUT"

step "4/5 Time them: $PERF_CONFIG"
# Point the config's '$NAME:<variant>' entries at the exact files built in step 1.
uv run python - "$PERF_CONFIG" "$OUT" "$NAME" "$PINNED" <<'PY'
import json, sys
from pathlib import Path
import yaml
perf_config, out, name, pinned = sys.argv[1:]
config = yaml.safe_load(Path(perf_config).read_text())
built = json.loads((Path(out) / "last_build.json").read_text())["artifacts"]
models = []
for ref in config["models"]:
    ref_name, _, variant = ref.partition(":")
    models.append(built[variant][:16] if ref_name == name and variant in built else ref)
config["models"] = models
Path(pinned).write_text(yaml.safe_dump(config, sort_keys=False))
print("Timing:", ", ".join(models))
PY
pin=()
if command -v taskset >/dev/null 2>&1 && [ "$(uname -m)" = "x86_64" ]; then
    # 4 different physical cores, as a stand-in for the Pi 5's 4 cores (section 6.1).
    # On CPUs with 2 threads per core, "0-3" can mean only 2 real cores.
    cpus="$(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}' | head -n 4 | paste -sd, -)"
    pin=(taskset -c "${cpus:-0-3}")
    echo "Pinned to logical CPUs ${cpus:-0-3}"
fi
"${pin[@]}" uv run echo-bench perf --config "$PINNED" --name "$(basename "$PERF_CONFIG" .yaml)"

step "5/5 Report"
uv run echo-bench report perf --latest
