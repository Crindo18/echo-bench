# ECHO-Bench

Benchmarking harness for the ECHO INT8 E-Branchformer, desktop first and then
Raspberry Pi 5 (4 GB). The design is in `docs/ECHO_Bench_Blueprint.md`; this
repository is at milestone **M3 (data layer and cross-validation plan)**; M2 waits for the Pi.

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
| `uv run echo-bench data ingest torgo\|uaspeech --root <folder>` | Scan a corpus into a manifest and the database |
| `uv run echo-bench data list / validate / coverage` | Loaded datasets; leakage rules L1-L2; feasible K per speaker |
| `uv run echo-bench cv plan --config <file>` / `cv summary` | Make the speaker-independent folds; fold table (desktop only) |
| `uv run echo-bench fewshot quick --embeddings <npz> ...` | Early accuracy check: Prototypical Networks on encoder embeddings |
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

Rebuilding the same recipe on the same machine reproduces byte-identical files (same
`<sha8>` folders). Another processor can round a few starting weights differently in the
last digit (differences around 3e-8), which changes the fingerprint but not the model's
size, speed or behaviour. To share a model between machines, copy its folder instead of
rebuilding it.
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

## M3: data layer and cross-validation plan (desktop)

**Get the corpora** and keep them outside this repository (licensed data is never committed):

- TORGO: free for academic, non-profit use from the University of Toronto
  (http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html); cite Rudzicz et al. (2012).
  Unpack F, FC, M and MC into one folder, for example `~/corpora/TORGO`.
- UASpeech: request access from the University of Illinois. Point `--root` at **one** version
  of the audio (original or noise-reduced); the ingester refuses duplicate copies. Optionally
  export the official word list to a CSV with columns `code,word` and pass it with `--wordlist`.

**Check the speaker tables first.** `configs/datasets/torgo_speakers.csv` and
`uaspeech_speakers.csv` hold each speaker's cohort and severity tier, with their sources in the
header. The tiers decide how the folds are stratified, so verify them against the corpus
documentation.

```bash
uv run echo-bench db upgrade          # adds the M3 tables; existing results are kept
uv run echo-bench data ingest torgo --root ~/corpora/TORGO
uv run echo-bench data ingest uaspeech --root ~/corpora/UASpeech/audio
uv run echo-bench data list
uv run echo-bench cv plan --config configs/cv/sgkf5-severity-s42.yaml
uv run echo-bench data validate       # must end with "No leakage found"
uv run echo-bench data coverage       # which K (3, 5, 10) each speaker's recordings support
uv run echo-bench cv summary          # the fold table for Chapters 3 and 5, saved under reports/
```

How it works:

- **One recording, one group.** The same utterance captured by several microphones shares a
  `recording_group_id`, and exactly one microphone is primary (TORGO: head mic; UASpeech: M5, the
  microphone used for per-word prototypes in prior prototype-based UASpeech work). By default
  only primary files are ingested; `--all-channels` adds the others.
- **Datasets and plans are immutable.** Re-ingesting identical files changes nothing; changed
  files need a new `--version`. A CV plan can't be edited, only replaced by one with a new name,
  so every trained model can point at the exact folds it used (rule L4).
- **The folds** (thesis §1.7.4.4): scikit-learn's `StratifiedGroupKFold` over speakers, stratified
  by corpus and severity tier. In fold i, group i is the test set, group i+1 the validation set,
  and the rest is training.
- **Manifests** (`data/manifests/*.jsonl`, git-ignored) list every file with its SHA-256.

## M3 checklist (blueprint section 7.2)

- [ ] Speaker tables checked against the corpus documentation
- [ ] TORGO ingested; UASpeech ingested once access is granted
- [ ] `echo-bench data validate` is clean
- [ ] Fold summary table saved (`echo-bench cv summary`) for Chapters 3 and 5

## Early accuracy check (before training our own model)

Question: do E-Branchformer embeddings plus Prototypical Networks separate dysarthric words at
all, and how does a Conformer compare? Published ESPnet encoders trained on the same 100 hours of
LibriSpeech answer that now, with no training. Run after `echo-bench data ingest torgo`:

```bash
cd packages/echo-train
M=../../data/manifests/torgo-v1.jsonl; E=../../data/embeddings
uv run echo-train embed --manifest $M --root ~/corpora/TORGO --out $E/torgo-v1__ls100-ebf.npz \
    --model hf:pyf98/librispeech_100_e_branchformer
uv run echo-train embed --manifest $M --root ~/corpora/TORGO --out $E/torgo-v1__ls100-conformer.npz \
    --model hf:pyf98/librispeech_100h_conformer
uv run echo-train embed --manifest $M --root ~/corpora/TORGO --out $E/torgo-v1__ebf-random.npz \
    --model hf:pyf98/librispeech_100_e_branchformer --random-weights
cd ../..
uv run echo-bench fewshot quick --embeddings data/embeddings/torgo-v1__ls100-ebf.npz \
    --embeddings data/embeddings/torgo-v1__ls100-conformer.npz \
    --embeddings data/embeddings/torgo-v1__ebf-random.npz
```

Add `--limit 200` to an `embed` command for a quick trial, and `--device cuda` if PyTorch sees
your GPU. The first run downloads each model once (into the Hugging Face cache).

**Word pool.** By default (`--pool matched`) SI and every K draw the same words in each
episode: words the speaker recorded at least max(K) + 1 times. On UASpeech that leaves out the
300 uncommon words, which are recorded once and could never be used for K. Before this option,
SI also drew those words, so SI and K were measured on different words; `--pool all` reproduces
those older numbers exactly, for comparison only.

**Paired comparison.** With several `--embeddings` files, the first is the reference and every
other encoder is compared with it speaker by speaker (dysarthric speakers only): mean difference,
how many speakers were better/worse, and a two-sided Wilcoxon signed-rank test, exact up to 20
speakers, with Holm's correction across the table. Put the thesis model first. `--csv <file>`
saves every speaker's accuracy for Chapter 5.

Reading the result:

- **SI** uses prototypes from other speakers only (the K = 0 baseline); **K=k** uses k of the
  speaker's own recordings per class. K above SI is the personalization effect SO4 is about.
- **Random weights** is the floor: the same E-Branchformer, untrained. The trained encoders
  should be well above it.
- **E-Branchformer vs Conformer**, both trained on identical data, is the fair architecture
  comparison. A clear Conformer lead would be a reason to revisit the model choice.
- Caveats: these are 12-block encoders (about 25M parameters, above the 16 MB budget), trained
  on typical read speech and not fine-tuned on dysarthric speech, and TORGO repeats each word
  only a few times per speaker (see `echo-bench data coverage`). An early signal, not a thesis
  result. The embedder refuses a checkpoint whose weights don't all load, because ESPnet would
  otherwise skip them silently and leave part of the encoder random.
