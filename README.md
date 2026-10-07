# ECHO Encoder Benchmark

A complete, runnable PyTorch benchmark of five speech-encoder architectures
(eight registry entries, including upstream/official implementations) for the
**ECHO Encoder Benchmark — Design Specification**. Every encoder pretrains on
typical speech (LibriSpeech), finetunes to atypical/dysarthric speech (TORGO),
and feeds a shared few-shot **Prototypical Network** intent classifier, all under
one matched edge budget (**~15M params / ≤16MB INT8 ONNX**, targeting a
Raspberry Pi 5).

**New here? Jump to [How to use / run](#how-to-use--run) for install →
data → train → compare, or the copy-paste [command cheat-sheet](#if-i-were-running-it-the-exact-commands-in-order)
at the end. [Troubleshooting](#troubleshooting) covers the common install/GPU
snags.**

The benchmark exists to answer one question (spec §1): for edge intent
recognition on atypical speech, is it better to **build a small model** (supervised
encoders trained at the budget) or to **shrink a big one** (a compressed speech
foundation model)?

| Track | Encoder (registry name) | Implementation | Front-end | Params |
|-------|---------|------|-----------|-----------------|
| Build small | Conformer (`conformer`) | this repo | log-Mel | 14.58M |
| Build small | E-Branchformer (`ebranchformer`) | this repo | log-Mel | 14.71M |
| Build small | E-Branchformer (`ebranchformer_espnet`) | **upstream ESPnet** | log-Mel | ~13M† |
| Build small | Zipformer (`zipformer`) | this repo | log-Mel | 14.57M |
| Build small | Moonshine (`moonshine`) | this repo, matched budget | waveform | 14.62M |
| Build small | Moonshine (`moonshine_tiny`) | this repo, official geometry | waveform | 7.68M |
| Build small | Moonshine (`moonshine_official`) | **official (HF transformers)** | waveform | 7.68M† |
| Shrink big | DPWavLM (`dpwavlm`) | this repo (no upstream exists) | waveform | 94.4M → ~15M* |

† `ebranchformer_espnet` and `moonshine_official` are the real upstream models —
optional deps (`pip install -e ".[espnet]"` / `".[moonshine]"`); their exact
params come from the released config. Use these for the deployment answer and to
validate the reimplementations. See "Own vs. official" below.

\* DPWavLM starts as a 94.4M-param WavLM-Base+-style student and is pruned to the
budget by structured L0 pruning; `materialize()` produces the compact static
model. Reaching ≤15M is gated by a Go/No-Go check (see below and spec §5).

## Inference pipeline

```
audio (16 kHz)
   → log-Mel spectrogram            (echo/features.py; supervised models)
   → FIRST ML STAGE: encoder        (the swapped variable — Conformer / E-Branchformer
                                      / Zipformer / Moonshine / DPWavLM; Moonshine and
                                      DPWavLM eat raw waveform directly, no Mel stage)
   → masked mean-pool + projector   (256-d L2-normalised utterance embedding)
   → Prototypical Network           (nearest-prototype match against the user's
                                      K-shot enrollment prototypes)
   → predicted class                (highest-match prototype)
```

The final class is a **toggle** (`--label-mode`, see Step 8): it can be an
**intent** (utterances grouped by the 36-intent taxonomy) or **any prompt** (each
distinct word/phrase is its own class, no taxonomy needed). The pipeline is
identical either way — only the label attached to each utterance changes — and the
number of classes is derived from the data, so nothing is hard-wired to 36.

This benchmark builds and measures exactly the **first ML stage + Prototypical
Network + INT8 quantization**, so you can decide (a) *which* encoder to deploy as
the first stage and (b) whether it fits the edge budget. The two learned pieces
are trained in two stages: the encoder is trained by CTC (LibriSpeech pretrain →
TORGO finetune); the projector — the embedding head the Prototypical Network
matches on — is then trained episodically (see "Training stages" below). At
deployment, enrollment is a forward pass: encode the user's K samples per class,
average to prototypes, and classify a query by its nearest prototype.

## Training stages

1. **CTC pretrain** (supervised encoders) on LibriSpeech — learns the acoustic
   representation. DPWavLM skips this: it inherits WavLM Base+ pretraining and is
   distilled+pruned instead.
2. **CTC finetune** on TORGO — adapts the encoder to atypical speech, then the
   encoder is frozen (it is the deployed feature extractor).
3. **Episodic meta-training of the Prototypical embedding head** on the
   intent-labelled TORGO *training* speakers (speaker-disjoint from the
   enrollment-eval speakers). The CTC objective never touches the projector, so
   without this stage the embedding would be a random projection. Meta-training
   runs N-way/K-shot Prototypical episodes with the **encoder frozen** — cheap
   (pooled features are cached once) and *identical for all five encoders*, so
   the only thing that differs downstream is the frozen representation. Runs
   automatically inside `finetune.py`; disable with `--skip-metatrain`.
4. **INT8 quantize + ONNX export**, then the **few-shot enrollment benchmark**.

---

## What this codebase is (and one important design decision)

The spec describes drawing each encoder from its "native" toolkit (ESPnet,
icefall/k2, S3PRL/DPHuBERT). **This implementation instead provides all five
encoders as clean, self-contained PyTorch modules behind a single common
`EncoderBase` interface.** This is a deliberate deviation, and the reasoning is:

- The benchmark's core methodological claim is *"swap only the encoder, hold
  everything else constant"* (§4). A single interface (`(features|waveform,
  lengths) → (hidden, out_lengths)`, fixed `embed_dim=256`) makes that literally
  true and testable in one harness, rather than reconciling three toolkits' data
  loaders, feature stacks, and training loops.
- It keeps the whole pipeline — front-end, CTC pretraining, freeze, Prototypical
  head, INT8 export, ONNX-Runtime latency — identical across models, which is
  exactly the matched-conditions requirement.

The trade-off: the encoders are **faithful reimplementations**, not the upstream
checkpoints. Where an upstream design has training machinery orthogonal to the
architecture, it is simplified with the simplification documented (below). If you
need the exact upstream recipes, this repo's harness is still useful as the
common evaluation and export layer.

---

## Architecture and per-model notes

All encoders live in `echo/encoders/` and share `EncoderBase`, a `Conv2dSubsampling`
4× stem (supervised models), relative-position multi-head attention, and the
same parameter-counting/masking utilities.

- **Conformer** (`conformer.py`) — sequential Macaron-FFN → MHSA → ConvModule →
  FFN blocks. `d=256, d_ff=1024, 8 layers`.
- **E-Branchformer** (`ebranchformer.py`) — parallel attention ∥ cgMLP (CSGU)
  branches with a learned merge. `d=256, d_ff=1024, cgmlp_ff=1536, 6 layers`.
  A second option, **`ebranchformer_espnet`**, wraps the *upstream ESPnet*
  E-Branchformer (the exact implementation the echo-bench harness uses) behind
  the same interface — use it when you want the community-trusted implementation
  and/or ESPnet's pretrained checkpoint for your headline baseline. It is an
  **optional dependency**: `pip install -e ".[espnet]"`. See "Using the upstream
  ESPnet E-Branchformer" below.
- **Zipformer** (`zipformer.py`) — U-Net-style temporal downsampling stacks
  (`blocks=(1,2,2,2,1)`, factors `(1,2,4,2,1)`), BiasNorm, and SwooshR
  activations. Trained with the machinery Zipformer is designed for:
  **ScaledAdam** (`echo/train/scaled_adam.py`, its scale-invariant optimizer —
  selected automatically for this encoder) and per-block **Balancer + Whitener**
  regularizers (`echo/encoders/regularizers.py`). These are *training-only* —
  identity at inference, zero added parameters or footprint — so the deployed,
  exported, INT8 model is unchanged. The one remaining deviation from icefall is
  fidelity of constants (the regularizers reproduce the mechanism, not the exact
  thresholds), not missing components.
- **Moonshine** (`moonshine.py`) — raw-waveform 3-conv stem (kernels 127/7/3,
  strides 64/3/2 → 384× compression) followed by pre-norm Transformer blocks with
  **partial RoPE** (0.9 of the head dim, `theta=10000`), GELU MLP, and bias-free
  projections/norms, matching the Moonshine encoder. There are **three registry
  entries**, for three purposes:
  - `moonshine` — this repo, sized *up* to the matched ~14.6M budget
    (`d_model=384, layers=8`), for a same-capacity architecture comparison.
  - `moonshine_tiny` — this repo, at the *official* tiny geometry
    (`d_model=288, layers=6` → **7.68M**), used **only** for the
    reimplementation-validation study (own-vs-official at matched size).
  - `moonshine_official` — the **real released** Moonshine encoder via HF
    `transformers` (~7.68M), the one you'd actually deploy.

  If you are not doing the own-vs-official validation study, you only need
  `moonshine` (architecture comparison) and/or `moonshine_official` (deployment);
  `moonshine_tiny` exists solely to make that validation a clean, same-size check.
- **DPWavLM** (`dpwavlm.py`) — a self-contained WavLM-Base+-style **student** with
  native structured-pruning gates (HardConcrete/L0) on attention heads, FFN
  units, and whole layers. It follows the DPHuBERT recipe: distill from a frozen
  **WavLM Base+ teacher** while an L0 Lagrangian drives the differentiable
  `expected_num_params()` down to the target, then `materialize()` bakes the open
  gates into a compact dense model, followed by a short recovery-distillation
  phase.

### Front-end caveat (matched conditions)

The three log-Mel encoders (Conformer, E-Branchformer, Zipformer) share the
**same log-Mel front-end** (80-mel, 25ms/10ms), held constant across models per
§4. **Moonshine and DPWavLM are the documented "origin" exceptions**: neither can
consume log-Mel features — DPWavLM inherits WavLM's learned conv extractor, and
Moonshine's design principle is to skip hand-engineered features entirely (a
learned 3-conv stem on raw waveform). Both are wired through the same
`AcousticModel` on the waveform path (`accepts_waveform = True`), which routes
them around the shared front-end. The genuinely matched points across *all five*
are therefore (a) the 256-d embedding interface into the identical Prototypical
head, and (b) the INT8-ONNX footprint/latency measured through the same ONNX
Runtime. This is called out in code and is the honest scope of "matched."

### DPWavLM teacher weights + student inheritance

`WavLMTeacher` loads real WavLM Base+ weights via
`torchaudio.pipelines.WAVLM_BASE_PLUS`. If the download is blocked it falls back
to the same architecture with random init and warns — but `distill_prune` now
**refuses to run against a random teacher** (distilling toward noise is
meaningless), so a blocked download fails fast with a clear message instead of
silently producing junk. Pass `--allow-random-teacher` only for a deliberate
offline pipeline smoke test (the result is *not* a valid model).

`init_student_from_teacher` performs the DPHuBERT **"student = teacher"** init:
it structurally copies the teacher's conv feature extractor, feature projection,
and every transformer layer's attention (q/k/v/out), FFN, and LayerNorm weights
into the full-size student, so the student genuinely starts as WavLM Base+. The
copy runs before any gate is learned, so it is a same-shape state-dict copy, not
a sliced one; structured pruning happens later in `materialize()`. The **one
inherited-capacity gap** is WavLM's gated *relative-position* bias, which this
student does not model (it uses a positional conv instead) — that part is not
transferable and is learned during distillation; the weight-normed positional
conv is likewise left to training.

Robustness: teacher tensors are matched by role within each `layers.{i}` block
(tolerant of torchaudio prefix differences), coverage is measured, and if it
falls below 70% — meaning your torchaudio's WavLM key names differ from the
expected layout — the init **raises** rather than silently degrading to a mostly
random student. `smoke_test.py` reports this coverage as a check, so run it first
to confirm inheritance lands on your torchaudio version (expect ~90%+).

---

## How to use / run

There are two layers. **`run_all.sh` / `run_dpwavlm.sh`** *train* each model and
give a quick per-model number. **`measure.py`** then *compares* the trained
models under leakage-free cross-validation and prints a ranked scorecard — the
"which one to deploy" answer. The normal order is: run the trainers for every
model, then run the comparison once.

> **Windows users — three things.** The commands below are written for
> Linux/macOS `bash`. On Windows: (1) run commands **one at a time** — don't
> chain with `&&` in older PowerShell. (2) After `pip install -e .` you can
> **drop the `PYTHONPATH=.` prefix** entirely (the `echo` package is installed),
> so `PYTHONPATH=. python scripts/x.py` becomes just `python scripts\x.py`; if
> you skipped the install, set it first with `$env:PYTHONPATH="."`. (3) The
> `.sh` training drivers (`run_all.sh`, `run_dpwavlm.sh`) are shell scripts —
> on Windows use the **native PowerShell ports `run_all.ps1` / `run_dpwavlm.ps1`**
> (same knobs, set as `$env:BATCH=…` etc.), run them under **Git Bash** / **WSL**,
> *or* call the underlying module commands directly (see "Run individual stages").
> Everything else (`smoke_test.py`, `measure.py`) is Python and runs the same on
> every OS.

### Step 0 — Put the code under git / GitHub (optional)

This is a normal Python package, so any standard git workflow applies. The
included `.gitignore` keeps corpora and generated artifacts (`data/`, `runs/`,
`export/`, `measure/`, `*.pt`, `*.onnx`) out of version control — you commit
source only.

**As a new GitHub repository.** Create an *empty* repo on github.com first (no
README/licence/gitignore, to avoid a merge conflict), then:

```bash
cd echo_benchmark
git init
git add .
git commit -m "ECHO encoder benchmark: initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

**Into an existing repository** (e.g. next to your thesis harness) — pick one:

```bash
# Option 1 — plain subdirectory (simplest; single repo & history)
cd <existing-repo>
cp -r /path/to/echo_benchmark ./echo_benchmark
git add echo_benchmark && git commit -m "Add ECHO encoder benchmark" && git push
#   then make sure the existing repo ignores the generated dirs, e.g.:
#   printf 'echo_benchmark/data/\necho_benchmark/runs/\necho_benchmark/export/\necho_benchmark/measure/\n' >> .gitignore

# Option 2 — git subtree (keep this project independently updatable in-tree)
cd <existing-repo>
git subtree add  --prefix=echo_benchmark <this-project-url> main --squash
git subtree pull --prefix=echo_benchmark <this-project-url> main --squash   # later, to update

# Option 3 — git submodule (histories stay fully separate)
cd <existing-repo>
git submodule add <this-project-url> echo_benchmark
git commit -m "Add echo_benchmark as submodule"
#   collaborators then run:  git submodule update --init --recursive
```

Use **Option 1** for a one-off drop-in, **Option 2/3** if this project lives in
its own repo and you want to pull future updates. For the uv-workspace
integration described in Step 1, add `echo_benchmark` to the workspace's
`[tool.uv.workspace] members` after placing it (Option 1 or 2).

### Step 1 — Set up the environment (venv **or** uv)

Pick **one** of the two paths. Both give a working, isolated environment. Plain
`venv` ships with Python and is the simplest match for this single-package
project; `uv` is far faster and the better choice if you will integrate with the
`echo-bench` workspace (see the end of this step). Installing the project
(`-e .`) makes the `echo` package importable, so you can drop the `PYTHONPATH=.`
prefix used elsewhere in this README.

Run the commands **one line at a time** (don't chain them with `&&` — that is not
valid in older PowerShell). Pick your OS block.

**Path A — venv + pip** (built into Python, nothing extra to install):

Linux / macOS (bash):
```bash
cd echo_benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .                     # installs the echo package + its deps (CPU torch)
#   GPU-ready pinned deps instead:  pip install -r requirements.txt  (installs a
#   CUDA cu126 torch build; then add the package with  pip install -e . --no-deps)
```

Windows (PowerShell):
```powershell
cd echo_benchmark
python -m venv .venv
.venv\Scripts\Activate.ps1           # if blocked, first run:
                                     #   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -e .                     # installs the echo package + its deps (CPU torch)
#   GPU-ready pinned deps instead:  pip install -r requirements.txt  (installs a
#   CUDA cu126 torch build; then add the package with  pip install -e . --no-deps)
```

If activation is blocked and you'd rather not change the policy, skip it and call
the venv binaries directly: `.venv\Scripts\python.exe -m pip install -e .`, then
run everything as `.venv\Scripts\python.exe scripts\smoke_test.py`.

**Path B — uv** (10–100× faster installs, reproducible):

Linux / macOS (bash):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # one-time install of uv
cd echo_benchmark
uv venv
source .venv/bin/activate            # or prefix commands with `uv run`
uv pip install -e .
```

Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # one-time install of uv
cd echo_benchmark
uv venv
.venv\Scripts\Activate.ps1           # or prefix commands with `uv run`
uv pip install -e .
```

**PyTorch note (both paths):** `pip install -e .` pulls a default **CPU** build of
`torch`/`torchaudio`, which is enough to run `smoke_test.py` and small checks. For
a **CUDA** build (needed to train at real speed) you have two options:

- **`pip install -r requirements.txt`** — pins a guaranteed CUDA build
  (`torch==2.14.0+cu126` / `torchaudio==2.11.0+cu126`; the `+cu126` local tag
  exists only on PyTorch's index, so pip can't silently fall back to a CPU
  wheel). It does *not* install the `echo` package, so follow with
  `pip install -e . --no-deps`. If it reports "No matching distribution", that
  torch has no `cu126` wheel — bump the tag to `cu128` in `requirements.txt`
  (both pins **and** the `--extra-index-url` line), or edit both to `+cpu`/`cpu`.
- **PyTorch's official selector** (https://pytorch.org/get-started/locally/) for
  any other CUDA version, installed *before* `pip install -e .`, e.g.
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126`
  (older tags like `cu121` are retired and fail with "No matching distribution").

Verify either way: `python -c "import torch; print(torch.cuda.is_available())"`
should print `True`.
Also note PyTorch only ships wheels for specific **Python versions** — if install
reports "from versions: none", your Python may be too new; use 3.11 or 3.12. On
the **Raspberry Pi 5** you only need `onnxruntime` (already a dependency) — not
torch — for the latency / RAM numbers from the exported INT8 ONNX.

**Integrating into the `echo-bench` uv workspace (optional).** If you keep this
alongside the `echo-bench` thesis harness (which is a uv workspace), add this
project as a workspace member so both share one reproducible, locked environment:
add `echo_benchmark` under `[tool.uv.workspace] members = [...]` in the
workspace's root `pyproject.toml`, then run `uv sync` at the workspace root. The
`echo` package and its CLIs then resolve inside the same `uv.lock`, so
`echo-bench`'s harness can call this project's training/quantize/measure entry
points directly.

### Step 2 — Sanity check (no data, no GPU)

Confirm the whole pipeline runs in your environment before committing to
downloads or training:

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

This builds every encoder on synthetic audio and exercises every stage — CTC
train step, freeze + embedding extraction, few-shot enrollment with adaptation
gain, INT8 static export + footprint, ONNX-Runtime latency, both DPWavLM gate
decisions, and the full data-hygiene + measurement layer (CV plan, leakage
contract, coverage, scorecard, severity toggle). It should end with
`ALL SMOKE TESTS PASSED`.

### Step 3 — Get the data

**LibriSpeech** (pretraining, typical speech) downloads automatically via
torchaudio on first run — nothing to do but point a path at a writable dir.

**TORGO** (fine-tuning, atypical speech) is four `.bz2` archives you download
manually from https://www.cs.toronto.edu/~complingweb/data/TORGO/ — `F`, `FC`,
`M`, `MC` (dysarthric females, female controls, dysarthric males, male controls;
15 speakers total). Extract **all four into one folder** so each speaker sits
directly under it. The built-in default location is
`C:\Users\ASUS\Desktop\ECHO_Datasets\torgo` — extract there and no path flag is
needed; extract elsewhere and pass `--torgo-root` (or edit the driver vars).

```powershell
# Windows PowerShell (tar ships with Windows 10+; it auto-detects .bz2):
mkdir "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo"
foreach ($f in "F","FC","M","MC") { tar -xf "$env:USERPROFILE\Downloads\$f.tar.bz2" -C "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo" }
```
```bash
# Linux / macOS (the C:\ default won't resolve here — extract anywhere and pass
# --torgo-root <path> to the commands, or edit the driver vars):
mkdir -p data/torgo
for f in F FC M MC; do tar -xjf ~/Downloads/$f.tar.bz2 -C data/torgo; done
```

The scanner expects speaker folders directly under `data/torgo/`, each with
`Session*/wav_*/*.wav` and matching `Session*/prompts/*.txt`:

```
data/torgo/
├── F01/  F03/  F04/          # dysarthric females
├── FC01/ FC02/ FC03/         # female controls
├── M01/ … M05/               # dysarthric males
└── MC01/ … MC04/             # male controls
```

**If the archives unpacked with a wrapper folder** (i.e. you see
`…/torgo/F/F01/…` instead of `…/torgo/F01/…`), flatten it:

```powershell
# Windows PowerShell:
cd "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo"
Get-ChildItem F,FC,M,MC -Directory | Move-Item -Destination .
Remove-Item F,FC,M,MC
```
```bash
# Linux / macOS (from wherever you extracted):
cd <your-torgo-root> && mv F/* FC/* M/* MC/* . && rmdir F FC M MC
```

**Verify the layout** (should list the speaker folders, then some `.wav` files):

```powershell
ls "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo"                          # -> F01 F03 F04 FC01 ... MC04
ls "C:\Users\ASUS\Desktop\ECHO_Datasets\torgo\F01\Session*\wav_*\"      # -> *.wav files
```

Notes on how these speakers are used:
- The **8 dysarthric speakers** (F01, F03, F04, M01–M05) are what the benchmark
  evaluates; CV folds are built over them, stratified by severity.
- The **7 controls** (FC*/MC*) are scanned but excluded from the folds
  (`dysarthric_only=True`) — usable as extra fine-tuning data, but they never
  skew the dysarthric evaluation.
- TORGO prompts mix literal words/sentences with stimulus pointers (bracketed
  descriptions, image refs); the scanner **skips the non-literal ones** for
  labelling, so the usable utterance count is lower than the raw `.wav` count —
  that is correct, not data loss.

**Quick data check (no training needed)** — confirm the extraction worked and see
which K each speaker supports. This is a plain script, so it runs the same on
every OS:

```bash
python scripts/check_torgo.py
#   options: --root C:/Users/ASUS/Desktop/ECHO_Datasets/torgo  --label-mode prompt|intent  --ks 3,5,10
```

It prints the utterance count, the speakers it found, and per-tier coverage. If it
says "0 utterances" or "folder not found", re-check the layout above (speaker
folders directly under `data/torgo/`).

(TORGO repeats each word only a handful of times, so expect K=3/5 well-supported
and K=10 sparse — the coverage report tells you exactly, per severity tier.)

**UASpeech** (optional second fine-tuning corpus) is supported by the same tools;
see the UASpeech note in *Two remaining substitutions* below.

### Step 4 — Configure

The drivers (and every module's CLI default) ship pointing at
`C:\Users\ASUS\Desktop\ECHO_Datasets`:

```bash
LIBRISPEECH_ROOT="C:/Users/ASUS/Desktop/ECHO_Datasets/librispeech"  # auto-downloads here
TORGO_ROOT="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo"              # where you extracted TORGO
HOLDOUT="F01 M01 F03 M03"                                           # speaker-disjoint eval speakers
```

Edit the top of `scripts/run_all.sh` / `run_dpwavlm.sh` (and the `$LibriRoot` /
`$TorgoRoot` lines in the `.ps1` ports) if your data lives elsewhere. Every
module also takes `--data-root` / `--torgo-root` / `--root` to override per-run,
so non-Windows users (where the `C:\…` default won't resolve) just pass the path
explicitly.

The `HOLDOUT` speakers must be folder names that exist in your TORGO copy.

**Training knobs — all environment-overridable (no file edit):**

| Var | Default | What it does |
|-----|---------|--------------|
| `BATCH` | 6 | per-step batch size |
| `MAX_DUR` | 16 | cap each clip to N seconds |
| `PRE_EPOCHS` | 30 | pretrain (LibriSpeech) epochs |
| `FT_EPOCHS` | 30 | finetune (TORGO) epochs |
| `PATIENCE` | 0 | early-stop after N epochs with no val-loss improvement (0 = off) |
| `MIN_DELTA` | 0 | min *relative* val-loss drop that counts as improvement (e.g. `0.005` = 0.5%); pairs with `PATIENCE` so trivial downticks on a noisy val set don't keep resetting it. Recommended for the small-holdout TORGO finetune |

```bash
# tighter memory + fewer epochs + early stopping (saves GPU time on convergence):
BATCH=4 MAX_DUR=12 PRE_EPOCHS=15 FT_EPOCHS=20 PATIENCE=5 MIN_DELTA=0.005 scripts/run_all.sh conformer 1
BATCH=16 scripts/run_all.sh conformer 1               # big GPU, defaults otherwise
```

`GPU memory`: peak VRAM scales with `BATCH × longest clip`, so both are capped;
the defaults are safe on ~6 GB. If you hit `CUDA out of memory`, lower `BATCH`
(6 → 4 → 2) and/or `MAX_DUR` (16 → 12). `Epochs`: the defaults (30/30) are sound,
slightly-conservative starting points, and the pipeline always keeps the *best*
checkpoint by validation loss, so extra epochs cost GPU time but not model
quality. Treat the counts as provisional: set the final values by where the
slowest-converging model's val-loss curve actually flattens (identical across all
models, for fairness). On a small GPU, set `PATIENCE` (e.g. 5) so a converged
model stops early instead of burning epochs, and/or lower the counts.
When running the module stages directly, these are the `--batch-size`,
`--max-duration`, `--epochs`, `--patience`, and `--min-delta` flags.

### Step 5 — Train + test one model

The `.sh` drivers are bash — run them in **Git Bash / WSL** on Windows. Each
encoder runs the full pipeline (pretrain → fine-tune → freeze → INT8 export →
few-shot benchmark + latency) in one command:

```bash
scripts/run_all.sh moonshine_official 1        # <encoder> <seed>
```

Produces `runs/moonshine_official/finetune_seed1_frozen.pt` (the deployed model),
`export/moonshine_official_int8.onnx`, and prints the accuracy panel (SI + post@K
+ gain + macro-F1) and latency. (Pure PowerShell without Git Bash → run the
module stages directly, see "Run individual stages".)

### Step 6 — Train + test the models you'll compare

The deployment set uses the *real* upstream models (Git Bash / WSL):

```bash
scripts/run_all.sh conformer            1
scripts/run_all.sh ebranchformer_espnet 1     # real ESPnet ( pip install -e ".[espnet]" )
scripts/run_all.sh zipformer            1
scripts/run_all.sh moonshine_official   1     # real Moonshine ( pip install -e ".[moonshine]" )
scripts/run_dpwavlm.sh                  1     # shrink-big arm, own entrypoint
```

Swap in the reimplementation names (`ebranchformer`, `moonshine`) instead if you
are running the own-vs-official validation study or don't want the optional deps.

DPWavLM is different: instead of pretraining it distills + prunes WavLM Base+ to
the budget and runs a **Go/No-Go gate**. If the gate prints `NO-GO`, stop there
for DPWavLM — the other four still form a complete benchmark (spec §5).

### Step 7 — Multiple seeds (for stability)

Robustness comes from several seeds per model. The `.sh` drivers use bash — run
this in **Git Bash / WSL** (swap in whichever encoders you're comparing):

```bash
for s in 1 2 3 4 5; do
  for m in conformer ebranchformer_espnet zipformer moonshine_official; do
    scripts/run_all.sh $m $s
  done
  scripts/run_dpwavlm.sh $s
done
```
In pure PowerShell (no Git Bash), loop the module stages instead — see
"Run individual stages".

### Step 8 — Compare the models (leakage-free scorecard)

Once the models are trained/frozen/exported (Steps 5–6), run the comparison. It
scans the corpus, builds speaker-independent severity-stratified CV folds,
**verifies no leakage** (and stops if it finds any), checks K-coverage,
cross-validates each model's few-shot accuracy with a **per-severity-tier**
breakdown, folds in footprint / budget / latency, and prints a ranked
**scorecard** with a recommendation:

```powershell
python scripts\measure.py `
    --corpus torgo --root C:\Users\ASUS\Desktop\ECHO_Datasets\torgo `
    --speaker-table configs\datasets\torgo_speakers.csv `
    --models moonshine_official,ebranchformer_espnet,zipformer,dpwavlm `
    --ckpt-dir runs --export-dir export `
    --n-folds 5 --seeds 1,2,3 --ks 3,5,10 --primary-k 5 `
    --label-mode intent --out measure
```

**Choosing what to match against** (`--label-mode`): the classifier can match to
an **intent** or to **any prompt**. This is the key toggle for getting numbers:

```bash
# (a) match against the 36-intent taxonomy — needs a real intent list
python scripts/measure.py ... --label-mode intent --out measure_intent

# (b) match against ANY prompt — each distinct word/phrase is its own class,
#     no taxonomy required; run this to get real numbers TODAY, before you have
#     finalised your intent list
python scripts/measure.py ... --label-mode prompt --out measure_prompt
```

In `intent` mode, prompts that don't match the taxonomy are skipped; in `prompt`
mode every distinct prompt becomes a class (case/whitespace-normalised) and the
class count comes from the data. Everything else — folds, leakage checks,
per-tier accuracy, scorecard — is identical, so the two runs are directly
comparable in structure.

> **Match the training label mode to the eval label mode.** The Prototypical
> projector is meta-trained during fine-tuning on a *class space* too, set by
> `LABEL_MODE` on the run drivers (→ `finetune --label-mode`). It must equal the
> `--label-mode` you pass to `measure.py`: training the embedding head to cluster
> by **intent** and then scoring it on **prompt** separability (or vice-versa) is
> a train/eval mismatch that understates accuracy. So for a prompt-mode scorecard,
> train with `LABEL_MODE=prompt` too:
> `LABEL_MODE=prompt scripts/run_all.sh conformer 1` … then `measure.py … --label-mode prompt`.
> Both default to `intent`, so the default run is already consistent.

Artifacts land in `--out` as JSON: `cv_plan`, `leakage`, `coverage`,
`scorecard`. Useful flags:

- `--label-mode {intent,prompt}` — what the classifier matches against.
  `intent` (default) groups utterances by the 36-intent taxonomy; `prompt` makes
  **each distinct prompt/word its own class** (no taxonomy needed — benchmark on
  the raw vocabulary, and skip the placeholder-taxonomy dependency). The number
  of classes is derived from the data, so nothing is hard-wired to 36.
- `--metatrain-ks 3,5,10` — also run a **meta-train-K × eval-K sweep**: for each
  K the embedding head could be *trained* at, re-train a fresh head **per fold**
  (leakage-free, on train speakers only) and evaluate it at every `--ks`. Prints a
  matrix and writes `sweep.json`, so you can see the full comparison — whether the
  head needs to be trained at the same K it's deployed at, and which combination
  wins. Off by default. (The single meta-train K for the *deployed* model is set
  separately by `--k-shot` in `finetune.py`.)
- `--no-severity` — turn off severity stratification and drop hardest-tier from
  the ranking (per-tier accuracy is still reported). See *What the measurement
  produces* below.
- `--all-channels` — keep every microphone instead of the primary channel only.
- `--distance cosine` — cosine instead of Euclidean prototype matching.
- `--models conformer,moonshine` — compare any subset.

### Step 9 — Run individual stages (optional)

`run_all.sh` is just these module CLIs chained together; run them directly to
resume or customise. Each has `--help`:

```bash
python -m echo.train.pretrain --help          # LibriSpeech CTC pretrain
python -m echo.train.finetune --help          # TORGO fine-tune + freeze + meta-train
python -m echo.train.distill_prune --help     # DPWavLM distill + structured prune
python -m echo.quantize.quantize_export --help# INT8 PTQ + ONNX export + footprint
python -m echo.eval.benchmark --help          # single-holdout few-shot benchmark
python -m echo.eval.latency --help            # ONNX-Runtime latency / RTF / RAM
```

### Where things land

```
runs/<encoder>/pretrain.pt                     LibriSpeech-pretrained checkpoint
runs/<encoder>/finetune_seed<S>_frozen.pt      deployed model (fine-tuned + frozen)
runs/dpwavlm/compressed.pt                      distilled+pruned DPWavLM student
export/<encoder>_int8.onnx                      INT8 ONNX for deployment / latency
measure/{cv_plan,leakage,coverage,scorecard}.json   the comparison artifacts
measure/sweep.json                              meta-train-K x eval-K matrix (with --metatrain-ks)
```

---

## If I were running it: the exact commands, in order

A copy-paste path from a fresh clone to the ranked scorecard, in **PowerShell**
(the training drivers in step 4 are `.sh` — run those in **Git Bash / WSL**, as
noted). Steps 1–3 run on any machine; training (4) needs a GPU.

```powershell
# 1. Environment
python -m venv .venv
.venv\Scripts\Activate.ps1              # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -e .                        # add the official encoders:  pip install -e ".[espnet,moonshine]"
#   GPU torch (for training):  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. Sanity check — no data, no GPU (should end "ALL SMOKE TESTS PASSED")
python scripts\smoke_test.py

# 3. Get data — extract TORGO into C:\Users\ASUS\Desktop\ECHO_Datasets\torgo (LibriSpeech auto-downloads)
mkdir C:\Users\ASUS\Desktop\ECHO_Datasets\torgo
foreach ($f in "F","FC","M","MC") { tar -xf "$env:USERPROFILE\Downloads\$f.tar.bz2" -C C:\Users\ASUS\Desktop\ECHO_Datasets\torgo }
```

**4. Train + export the models you'll compare (GPU).** Edit `TORGO_ROOT` /
`HOLDOUT` at the top of the drivers first, then pick the *real* models for the
deployment benchmark. In **PowerShell**, use the native `.ps1` ports:

```powershell
$env:BATCH=6; $env:MAX_DUR=16      # optional; add $env:PATIENCE=5; $env:MIN_DELTA=0.005 for early stopping
$env:LABEL_MODE="prompt"           # match the prompt-mode scorecard in step 5 (train the projector on the same class space)
foreach ($m in "conformer","ebranchformer_espnet","zipformer","moonshine_official") {
    .\scripts\run_all.ps1 $m 1     # add more seeds (…$m 2, …$m 3) for stability
}
.\scripts\run_dpwavlm.ps1 1        # the shrink-big arm (no upstream model exists)
```

The equivalent `.sh` drivers run in **Git Bash / WSL** (`for m in conformer …;
do scripts/run_all.sh $m 1; done`). No drivers at all? Run the per-model stages
directly — see "Run individual stages".

**5. Compare them — the leakage-free scorecard** (the "which to deploy" answer),
back in PowerShell (note the backtick `` ` `` line continuation):

```powershell
python scripts\measure.py `
    --corpus torgo --root C:\Users\ASUS\Desktop\ECHO_Datasets\torgo `
    --speaker-table configs\datasets\torgo_speakers.csv `
    --models moonshine_official,ebranchformer_espnet,dpwavlm,conformer,zipformer `
    --ckpt-dir runs --export-dir export `
    --n-folds 5 --seeds 1,2,3 --ks 3,5,10 --primary-k 5 `
    --label-mode prompt --out measure_deploy
#   -> prints size + accuracy + per-tier + latency + recommendation, and writes
#      measure_deploy\{cv_plan,leakage,coverage,scorecard}.json
```

**Reimplementation-validation study** (optional — own vs. official at matched
size; see "Own vs. official"):

```powershell
python scripts\measure.py --corpus torgo --root C:\Users\ASUS\Desktop\ECHO_Datasets\torgo `
    --speaker-table configs\datasets\torgo_speakers.csv `
    --models moonshine_tiny,moonshine_official --out measure_val_moon
python scripts\measure.py --corpus torgo --root C:\Users\ASUS\Desktop\ECHO_Datasets\torgo `
    --speaker-table configs\datasets\torgo_speakers.csv `
    --models ebranchformer,ebranchformer_espnet --out measure_val_ebf
```

Common variations of the compare command:

```powershell
--label-mode intent            # match against your 36-intent taxonomy instead of prompts
--metatrain-ks 3,5,10          # also print the meta-train-K x eval-K accuracy matrix
--no-severity                  # unstratified folds, no hardest-tier weighting
--models zipformer,moonshine   # compare only a subset
```

To see the output *shape* before training, run step 5 without trained
checkpoints — it warns and uses random models (accuracy ≈ chance) but shows the
exact console + JSON layout you'll get with real numbers.

---

## Own vs. official implementations (two questions, two tables)

For E-Branchformer and Moonshine the harness carries **both** this repo's
reimplementation and the **real upstream model** (ESPnet / HF Moonshine). That is
deliberate — they answer two different questions, and mixing them into one table
muddies both:

1. **Deployment benchmark (which model do we ship?).** Use the *real* models —
   you deploy the trusted artifact, not a reimplementation. Compare
   `ebranchformer_espnet`, `moonshine_official`, your **DPWavLM** (the shrink-big
   arm, for which no off-the-shelf model exists), and Conformer/Zipformer at their
   **native sizes**. Sizes need not match: the scorecard's footprint and latency
   criteria already reward the smaller/faster model, so a 7.68M Moonshine that
   ties a 13M E-Branchformer correctly ranks *higher*.

   ```powershell
   python scripts\measure.py --corpus torgo --root C:\Users\ASUS\Desktop\ECHO_Datasets\torgo `
       --speaker-table configs\datasets\torgo_speakers.csv `
       --models moonshine_official,ebranchformer_espnet,dpwavlm,conformer,zipformer `
       --ckpt-dir runs --export-dir export --label-mode prompt --out measure_deploy
   ```

2. **Reimplementation validation (are our reimplementations faithful?).** A small
   **matched-size** study: this repo's encoder vs. the official one at the *same*
   geometry. If they track within a point or two, that validates every number the
   reimplementations produce. `moonshine_tiny` is this repo's Moonshine at the
   official tiny geometry (7.68M) precisely for this check.

   ```powershell
   python scripts\measure.py ... --models moonshine_tiny,moonshine_official --out measure_val_moon
   python scripts\measure.py ... --models ebranchformer,ebranchformer_espnet --out measure_val_ebf
   ```

Report the two separately: table 1 picks the model, table 2 earns trust in the
method. DPWavLM has no "official" counterpart, so it only appears in table 1 — the
reimplementation *is* the contribution there.

### Or: benchmark everything at once (all 8)

For a thorough, thesis-grade sweep you can train and compare **all** encoders —
own and official — in one go. Train each (needs the optional deps for the two
official ones: `pip install -e ".[espnet,moonshine]"`):

```bash
for m in conformer ebranchformer ebranchformer_espnet zipformer \
         moonshine moonshine_tiny moonshine_official; do
  scripts/run_all.sh $m 1        # Git Bash / WSL; BATCH/MAX_DUR keep each ~6 GB
done
scripts/run_dpwavlm.sh 1
```

Then one combined scorecard across all 8:

```powershell
python scripts\measure.py --corpus torgo --root C:\Users\ASUS\Desktop\ECHO_Datasets\torgo `
    --speaker-table configs\datasets\torgo_speakers.csv `
    --models conformer,ebranchformer,ebranchformer_espnet,zipformer,moonshine,moonshine_tiny,moonshine_official,dpwavlm `
    --ckpt-dir runs --export-dir export --label-mode prompt --out measure_all
```

**Read the all-8 table in two passes** (it mixes the two questions above): (1)
across the *distinct* models — Conformer vs. an E-Branchformer vs. Zipformer vs. a
Moonshine vs. DPWavLM — to pick what to deploy (prefer the official rows); (2)
down the *pairs* — `ebranchformer` vs `ebranchformer_espnet`, `moonshine_tiny` vs
`moonshine_official` (both 7.68M) — to judge reimplementation fidelity. The
scorecard scores footprint and latency, so the smaller 7.68M models get proper
credit — a reimpl landing within a point or two of its official twin is a
*validation* result, not a loss. To make that point explicit, also pull the pairs
into their own small tables (same runs, viewed two ways):

```powershell
python scripts\measure.py ... --models moonshine_tiny,moonshine_official --out measure_val_moon
python scripts\measure.py ... --models ebranchformer,ebranchformer_espnet --out measure_val_ebf
```

## Using the upstream ESPnet E-Branchformer (optional)

The `ebranchformer` above is this repo's from-scratch reimplementation. For a
more thesis-defensible headline baseline you can instead use the **upstream
ESPnet** E-Branchformer — the same implementation the echo-bench harness uses,
with access to ESPnet's pretrained checkpoints — via the `ebranchformer_espnet`
adapter, which wraps ESPnet's `EBranchformerEncoder` behind the identical
`EncoderBase` interface (`echo/encoders/ebranchformer_espnet.py`). Everything
downstream (shared log-Mel front-end, Prototypical head, CV, scorecard) is
unchanged, so it swaps in like any other encoder.

**Install and run.** ESPnet is an optional dep (large; on Windows install it in
WSL/Linux). Then `ebranchformer_espnet` is just another registry name — train,
export, and compare it exactly like any encoder (see *How to use / run*):

```bash
pip install -e ".[espnet]"                 # installs ESPnet + huggingface_hub
scripts/run_all.sh ebranchformer_espnet 1  # then add it to measure.py --models
```

**Start from ESPnet's pretrained checkpoint** (the main reason to use ESPnet) —
build the adapter sized to the checkpoint's config, then load its weights:

```python
from echo.encoders import build_encoder
enc = build_encoder("ebranchformer_espnet", d_model=256, num_blocks=12)   # match the ckpt
enc.load_pretrained("pyf98/librispeech_100_e_branchformer")   # HF id, or a local .pth
```

`load_pretrained` copies only the matching `encoder.*` tensors, reports how many,
and errors if it loads zero (config mismatch — resize and retry).

Two things for a fair comparison: **(1) Consistency** — don't mix your reimpl and
ESPnet supervised encoders in one deployment table (see *Own vs. official*).
**(2) Pretrained front-end** — ESPnet checkpoints expect ESPnet's feature
normalisation (global MVN); this repo applies utterance CMVN, which is fine for
random-init training but means a *pretrained* checkpoint needs ESPnet-matched
features to transfer. The adapter is import-safe without ESPnet and verified in
`smoke_test.py` against a mock, but the ESPnet-specific behaviour (encoder
internals, exact param count, checkpoint keys) must be validated in your ESPnet
environment.

## Using the official Moonshine (optional)

The `moonshine_official` adapter wraps the **released Moonshine** encoder (Useful
Sensors) via HuggingFace `transformers`, behind the same `EncoderBase` interface
(`echo/encoders/moonshine_official.py`) — same idea as the ESPnet adapter, same
strict-loading discipline (it refuses a partially-loaded checkpoint).

```bash
pip install -e ".[moonshine]"                 # installs transformers + huggingface_hub
scripts/run_all.sh moonshine_official 1       # uses the real moonshine-tiny weights
```

By default it loads `UsefulSensors/moonshine-tiny` (~7.68M encoder). Options via
the constructor: `source=` (any HF id or local folder, e.g. `moonshine-base`) and
`pretrained=False` (the official architecture with random weights — the untrained
"floor" baseline echo-train uses). Like the ESPnet adapter, it is import-safe
without `transformers`, its wiring is verified in `smoke_test.py` against a mock,
and the real-model behaviour (exact frames, checkpoint keys) should be validated
in your environment. Compare it against this repo's `moonshine_tiny` (same 7.68M
geometry) for the reimplementation-validation study above.

## Footprint / INT8 quantization

Export and quantization live in `echo/quantize/quantize_export.py`.

- **What is exported.** The deployed artifact is the frozen encoder → pooled,
  projected 256-d embedding (the input to the Prototypical head), exported from
  each encoder's **native input**: log-Mel features `[B,T,80]` for the supervised
  encoders, raw waveform for DPWavLM. The log-Mel filterbank is a fixed on-device
  DSP stage (not a learned part of the encoder, §4), and `torch.stft` is not
  cleanly ONNX-exportable across runtimes — so keeping the filterbank outside the
  graph is both correct and portable. Compute log-Mel on-device (the same
  `echo/features.py` config) and feed features to the supervised graphs.
- **Static vs dynamic INT8.** `quantize_int8_static` (default) uses ONNX
  Runtime's **QDQ** static format and quantizes **both Conv and MatMul** (plus
  calibrated activations), so the whole encoder runs INT8 and the on-disk size is
  the true deployed footprint. `quantize_int8_onnx` (dynamic) is the
  calibration-free alternative but leaves Conv in FP32, so conv-heavy encoders
  stay large under it — use it only when you can't ship a calibration set. QDQ is
  chosen over the QOperator format because QOperator's shape-inference cannot
  handle Zipformer's SwooshR/BiasNorm and E-Branchformer's cgMLP activation
  subgraphs; on the Pi-5 CPU EP, ORT fuses QDQ islands into true INT8 kernels at
  load, so the two formats run equivalently.
- **Budget verification caveat.** The ≤16MB ground-truth number is realised on a
  **trained** checkpoint quantized with **representative calibration data** (a
  held-out slice of TORGO/LibriSpeech features — pass a real
  `CalibrationDataReader`). On untrained weights with random calibration, static
  INT8 sizes are inflated and meaningless. The stable, calibration-independent
  budget proxy is the **parameter count** (~1 byte/param at INT8), which
  `footprint_report` reports as `params_within_budget` and which all three
  supervised encoders (~14.6M) pass. Read `onnx_within_budget` off your trained
  models.

### DPWavLM Go/No-Go gate (spec §5)

`echo/train/distill_prune.py:go_no_go_gate` materializes the current student and
checks it against the budget. DPWavLM carries a hard floor (~9M) from its conv
feature extractor (~4.2M) + positional conv (~4.7M), so reaching ≤15M requires
aggressive layer *and* width pruning — this is exactly the "risk-bearing" nature
§5 describes. The gate returns **GO** only when a materialized, deployable model
is within budget; otherwise **NO-GO** with the §5 guidance ("don't sink weeks in;
report the smallest reachable size as an out-of-budget reference row, or drop
DPWavLM — the other three still form a complete benchmark").

---

## What the measurement produces (and how it stays honest)

Running Step 8 (`measure.py`) turns the trained models into a decision, writing
`cv_plan.json`, `leakage.json`, `coverage.json`, `scorecard.json` (and
`sweep.json` with `--metatrain-ks`). It reports, per model:

- **Footprint** — encoder parameters, FP32 / INT8 ONNX size (MB), and whether
  it meets the ≤16 MB budget.
- **Accuracy** — SI (K=0), post-enrollment accuracy at K∈{3,5,10}, **adaptation
  gain** (post − SI on identical queries) and macro-F1, each as mean ± ~95% CI
  over (seed × fold), plus a **per-severity-tier** breakdown so a model that
  collapses on *severe* speech can't hide behind a good average.
- **Speed / memory** — ONNX-Runtime latency (mean / p90), RTF, peak RAM.
- **Scorecard** — all of the above combined into one ranked, tunable weighted
  score with a recommended encoder (over-budget models are shown but excluded).

Why the numbers can be trusted:

- **Leakage-free by construction and by proof.** Folds are speaker-independent
  (a speaker lives wholly in one fold) and severity-stratified; the
  `verify_no_leakage` contract then *checks* two invariants — no speaker shared
  across train/val/test, and no multi-mic utterance split across sides — and
  **stops the run** if either fails. One channel per utterance is kept by
  default, so a word said once counts once.
- **Only honest K's are reported.** The `coverage` check reports how many
  speakers per tier can support each K (needs > K reps); unsupported K's return
  NaN rather than a faked number.
- **No inflation.** Prototypes are built only from enrolled samples (never from
  query labels), and a random (untrained) model on noise scores at chance — so
  accuracy above chance is earned, not manufactured.

Enrollment detail: for each K, a class needs more than K samples to be both
enrolled and queried; classes with ≤K samples are excluded from *both* sets that
episode, and the classifier only predicts classes that were actually enrolled.

The `--no-severity`, `--label-mode`, and `--metatrain-ks` toggles (Step 8) change
*what* is measured without changing these guarantees.

## Repository layout

```
echo/
  features.py            log-Mel front-end (+ SpecAugment)
  heads.py               CTC head, pooling, 256-d embedding projector
  prototypical.py        Prototypical Network head (enroll / classify / episodic loss)
  encoders/              base, modules, conformer, ebranchformer(+espnet), zipformer, moonshine(+tiny+official), dpwavlm
  data/                  tokenizer, LibriSpeech/TORGO datasets, intent taxonomy,
                         label scheme (intent | prompt), corpus scan,
                         CV splits + leakage contract, coverage
  train/                 common (AcousticModel), pretrain, finetune, metatrain, distill_prune
  quantize/              ONNX export + INT8 PTQ + footprint report
  eval/                  few-shot benchmark, cross-validation protocol,
                         meta-train-K x eval-K sweep, ONNX-Runtime latency,
                         model-selection scorecard
configs/datasets/        editable TORGO / UASpeech speaker cohort+severity tables
scripts/                 run_all.sh + run_dpwavlm.sh (bash) and run_all.ps1 +
                         run_dpwavlm.ps1 (PowerShell) training drivers, measure.py,
                         check_torgo.py, check_earlystop_proxy.py, smoke_test.py
pyproject.toml           installable package (pip/uv, workspace integration)
requirements.txt         pinned deps + a guaranteed CUDA (cu126) torch build
.gitignore               keeps corpora + generated artifacts out of git
```

## Two remaining substitutions to make for the real study

- **Intent taxonomy (only needed for `--label-mode intent`).**
  `echo/data/intents.py` ships a **placeholder** 36-intent / 6-category set with a
  prompt→intent mapper. Replace `INTENT_SET` (and the mapper) with the real ECHO
  taxonomy and the corpus prompt strings before reporting intent-mode
  accuracy/macro-F1. **You can skip this entirely with `--label-mode prompt`,**
  which needs no taxonomy — so you can get real, structurally-identical comparison
  numbers before the intent list is finalised.
- **Trained numbers.** Training here was verified on synthetic tensors only (the
  development environment had no network access to the corpora or pretrained
  weights). The architecture, budgets, pipeline, export, and gate logic are
  verified; run the scripts on real LibriSpeech/TORGO in your environment for the
  benchmark's actual accuracy, footprint, and latency figures.

---

## Troubleshooting

Common setup/run issues (most are environment drift, not the code):

**`CUDA out of memory` during training.** Peak VRAM scales with batch size × the
longest audio clip. Lower the knobs (they're env-overridable, no file edit):
```bash
BATCH=4 MAX_DUR=12 scripts/run_all.sh conformer 1     # ~6 GB defaults are 6 / 16
```
Drop `BATCH` (6 → 4 → 2) and/or `MAX_DUR` (16 → 12 → 8) until it fits; check free
VRAM with `nvidia-smi`. Running the module stages directly? Same as
`--batch-size` / `--max-duration`.

**`ModuleNotFoundError: No module named 'torchcodec'`** (at data loading). The
project loads audio (LibriSpeech FLAC + TORGO WAV) via **soundfile**, not
`torchaudio.load`, precisely so a recent torchaudio (which routes `load()`
through TorchCodec) doesn't require TorchCodec. So the fix is just to ensure
soundfile is installed — it's a declared dependency, so `pip install -e .` (or
`pip install soundfile`) resolves it; **no `torchcodec` needed**. (If you hit
this on an *older* copy of the project, update to this version.)

**`Could not find a version that satisfies the requirement torch` (from cu121).**
The `cu121` CUDA index is retired. Use a current one — `cu126` or `cu128`:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```
If it still fails, your Python may be too new for prebuilt wheels — use Python
3.11 or 3.12, or use the selector at https://pytorch.org/get-started/locally/.

**Training says `device=cpu` but you have a GPU.** You installed the CPU build of
torch. Confirm the GPU is visible, then reinstall a CUDA build:
```bash
python -c "import torch; print(torch.cuda.is_available())"    # want True
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```
The device is auto-selected — no flag to change; installing CUDA torch is the fix.

**PowerShell: `The token '&&' is not a valid statement separator` / `Missing
opening '(' after keyword 'for'`.** You copied a bash command into PowerShell. Run
lines one at a time (not chained with `&&`), use PowerShell loops
(`foreach ($x in "a","b") { ... }`), `\` → `` ` `` for line continuation, and
`.venv\Scripts\Activate.ps1` (not `source .venv/bin/activate`). Each command block
in this README has a Windows PowerShell form.

**PowerShell: `Activate.ps1 cannot be loaded because running scripts is
disabled`.** Allow scripts for the session, then activate:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```
Or skip activation and call the venv binaries directly:
`.venv\Scripts\python.exe scripts\smoke_test.py`.

**The `.sh` training drivers won't run in PowerShell.** They're bash scripts —
run them in **Git Bash** or **WSL** (e.g. `& "C:\Program Files\Git\bin\bash.exe"
scripts/run_all.sh conformer 1`), or run the per-model `python -m echo.train.…`
stages directly in PowerShell (see "Run individual stages").

**`ImportError: … needs ESPnet / transformers`.** You built an optional encoder
(`ebranchformer_espnet` / `moonshine_official`) without its dependency. Install
the extra: `pip install -e ".[espnet]"` or `pip install -e ".[moonshine]"`. The
other encoders work without them.
