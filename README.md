# ECHO-Bench

Benchmarking harness for the ECHO INT8 E-Branchformer, desktop first and then
Raspberry Pi 5 (4 GB). The design is in `docs/ECHO_Bench_Blueprint.md`; this
repository is at milestone **M0 (foundations)**.

## Layout

```text
packages/
  echo-core/      ships on the device (no torch, espnet, librosa, pandas)
  echo-bench/     the harness: CLI, results database, fingerprints (desktop + Pi)
  echo-analysis/  statistics, figures, thesis tables (desktop only)
  echo-train/     build, export, quantize (desktop only, its OWN uv.lock)
tests/            run with `uv run pytest`
scripts/pi/       Raspberry Pi setup
docs/             the blueprint
alembic.ini       for creating new database migrations
```

Never committed (see `.gitignore`): `data/`, `artifacts/`, `results/`,
`reports/` and any `*.db`. Participant data is restricted (RA 10173).

## Desktop setup (Ubuntu 24.04 or WSL2)

```bash
uv sync --all-packages          # installs echo-core, echo-bench, echo-analysis + dev tools
uv run pytest                   # all tests should pass
uv run echo-bench db init       # creates results/bench_<hostname>.db
uv run echo-bench doctor        # checks this machine
uv run pre-commit install       # automatic checks before every commit
```

The training environment is separate:

```bash
cd packages/echo-train
uv sync                         # PyTorch 2.11 + ESPnet: a few GB the first time
uv run python scripts/echo_practice_run.py
```

## Raspberry Pi 5 setup

```bash
sudo apt update && sudo apt install -y git
git clone <your-repo-url> echo-bench && cd echo-bench
bash scripts/pi/setup_pi.sh
```

On the Pi, always run commands as `uv run --package echo-bench echo-bench ...`.
A plain `uv run` installs the desktop-only analysis packages.

## Everyday commands

| Command | What it does |
|---|---|
| `uv run echo-bench doctor` | Hardware/software fingerprints and environment checks |
| `uv run echo-bench doctor --save` | Same, and stores the fingerprints in the database |
| `uv run echo-bench db init` | Creates the database, or brings it up to date |
| `uv run echo-bench db upgrade` | Applies new migrations to an existing database |
| `uv run pytest` | Runs every test |
| `uv run pre-commit run --all-files` | Runs lint, format, type and import checks |

## M0 checklist (blueprint section 7.2)

- [ ] `uv run pytest` passes on the desktop
- [ ] `echo-bench db init` and `echo-bench doctor` succeed on the desktop
- [ ] Repository pushed to a **private** GitHub repo
- [ ] Pi flashed with Raspberry Pi OS Lite (64-bit), `setup_pi.sh` finished
- [ ] `echo-bench db init` and `echo-bench doctor` succeed on the Pi
