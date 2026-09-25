#!/usr/bin/env bash
# One-time setup of a Raspberry Pi 5 for ECHO-Bench (blueprint M0 and section 11.1).
#
# Before running: flash Raspberry Pi OS Lite (64-bit), log in over SSH, then
#   sudo apt update && sudo apt install -y git
#   git clone <your-repo-url> echo-bench && cd echo-bench
#   bash scripts/pi/setup_pi.sh
set -euo pipefail

cd "$(dirname "$0")/../.."  # the repository root

if ! grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
  echo "Warning: this does not look like a Raspberry Pi 5. Continuing anyway."
fi

echo "==> System packages (audio libraries are for the M8 end-to-end runs)"
sudo apt-get update
sudo apt-get install -y git curl libportaudio2 libsndfile1

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Python 3.12 managed by uv (the system Python is left alone)"
uv python install 3.12

echo "==> echo-core and echo-bench only: no PyTorch, no pandas"
uv sync --package echo-bench

echo "==> Results database and environment check"
uv run --package echo-bench echo-bench db init
uv run --package echo-bench echo-bench doctor --cooling active_cooler --power-source official_27w_psu

cat <<'MSG'

Setup finished. On the Pi, always start commands with `uv run --package echo-bench`:
    uv run --package echo-bench echo-bench doctor
A plain `uv run echo-bench ...` would install the desktop-only analysis
packages (pandas, matplotlib) onto the Pi.
Change --cooling and --power-source to match the setup you are actually using.
MSG
