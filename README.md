# ECHO-Bench

Benchmarking harness for the ECHO INT8 E-Branchformer, desktop first and then
Raspberry Pi 5 (4 GB). The design is in `docs/ECHO_Bench_Blueprint.md`; this
repository is at milestone **M1 (MVP: architecture-level performance on the desktop)**.

## Layout

```text
packages/
  echo-core/      ships on the device (no torch, espnet, librosa, pandas)
  echo-bench/     the harness: CLI, results database, fingerprints (desktop + Pi)
  echo-analysis/  statistics, figures, thesis tables (desktop only)
  echo-train/     build, export, quantize (desktop only, its OWN uv.lock)
tests/            run with `uv run pytest`
configs/          model recipes, benchmark settings, gate thresholds
scripts/          m1_slice.sh (the whole M1 run), pi/ (Raspberry Pi setup)
docs/             the blueprint, feasibility.md (can it run on the Pi?), references.md (Moonshine)
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
| `uv run echo-bench model register <folder>` | Checks model files against their cards, records gates G1/G2a |
| `uv run echo-bench model list` | Every registered model |
| `uv run echo-bench model inspect <name:variant>` | Size, parameters, operators, quantization coverage, compute per utterance |
| `uv run echo-bench perf --config <file>` | Latency, memory and temperature per model and input length |
| `uv run echo-bench report perf --latest` | Table and figure for the newest perf campaign (desktop only) |
| `uv run pytest` | Runs every test |
| `uv run pre-commit run --all-files` | Runs lint, format, type and import checks |

## M0 checklist (blueprint section 7.2)

- [ ] `uv run pytest` passes on the desktop
- [ ] `echo-bench db init` and `echo-bench doctor` succeed on the desktop
- [ ] Repository pushed to a **private** GitHub repo
- [ ] Pi flashed with Raspberry Pi OS Lite (64-bit), `setup_pi.sh` finished
- [ ] `echo-bench db init` and `echo-bench doctor` succeed on the Pi

## M1: the whole slice in one command (desktop)

```bash
scripts/m1_slice.sh
```

It runs five steps; each one also works on its own:

1. `echo-train build-random`: builds the encoder from `configs/models/ebf12m.yaml` with
   random weights, prints parameters per module, exports ONNX FP32, checks PyTorch-vs-ONNX
   parity at 1-10 s (gate G2a), and makes INT8 versions. Files go to
   `artifacts/models/ebf-12m-rand/<sha8>/`, each with a `model_card.json`.
2. `echo-bench db init`
3. `echo-bench model register artifacts/models/ebf-12m-rand` (gates G1 and G2a)
4. `echo-bench perf` with `configs/benchmarks/perf_smoke.yaml`, pinned to 4 physical cores
5. `echo-bench report perf --latest`: console tables and `reports/perf_<id>_latency_vs_length.png`

Rebuilding the same recipe reproduces byte-identical files (same `<sha8>` folders).
Desktop numbers show trends only; the Pi numbers come in M2 (section 6.1).

### Refining the model

1. Edit `configs/models/ebf12m.yaml` (for example `num_blocks`, `output_size`, `linear_units`).
   Change one setting per run so you know what caused the difference.
2. Run `scripts/m1_slice.sh`. It times exactly the files it just built.
3. Compare the size (G1) and latency with earlier campaigns.

To compare two recipes side by side, copy the recipe to a new file with a new `name:`
(for example `ebf-6blk`), build both, and list both in a perf config:
`[ebf-12m-rand:onnx_int8_static, ebf-6blk-rand:onnx_int8_static]`. A `name:variant`
entry means the most recently registered file with that name; a SHA-256 prefix
(at least 6 characters, from `echo-bench model list`) picks one exact file.

More INT8 versions of an existing FP32 model:

```bash
cd packages/echo-train
uv run echo-train quantize --fp32 ../../artifacts/models/ebf-12m-rand/<sha8> \
    --method static_qdq --calibration Entropy
```

### Findings built into the code (M1)

- ESPnet's default `max_pos_emb_len` (5000) stores a 10 MB table in the ONNX file;
  500 covers about 20 s of audio and keeps INT8 under 16 MB.
- ONNX Runtime preprocessing needs `skip_symbolic_shape=True` on this model.
- Entropy and Percentile calibration need equal-length clips (3 s is used).
- ESPnet's attention mask constant (-3.4e38) breaks histogram calibration, so static
  quantization leaves the mask (`Where`) nodes in float for every calibration method.
- On x86, dynamic INT8 (all operators) is slower than FP32; static QDQ is the fastest.
  Check again on the Pi in M2.

## M1 checklist (blueprint section 7.2)

- [ ] `scripts/m1_slice.sh` finishes on the desktop
- [ ] G2a passes (export parity) and G1 is recorded for each INT8 file
- [ ] `echo-bench model inspect` confirms the parameter count (about 13.06 M) and compute (about 3.1 billion MACs at 5 s)
- [ ] `docs/feasibility.md` numbers checked against your own run
- [ ] Figure saved under `reports/`; numbers noted for the M1 write-up
