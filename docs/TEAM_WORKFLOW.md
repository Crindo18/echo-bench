# Team workflow: one repository, no duplicated work

Sir Melvin asked that every reported result be traceable to a specific configuration and test.
That only works if everyone runs the same code from this repository,
https://github.com/Crindo18/echo-bench, and never from zip copies.

## 1. The repository and its branches

| Branch | What it is | Status |
|---|---|---|
| `main` | CJ's harness (M0, M1, M3, early accuracy check, skip too-short recordings) | Base |
| `uaspeech-moonshine` | `main` + Brendan's Moonshine loader and encoder-size notes | Merge into `main` |
| `official-fixes` | `uaspeech-moonshine` + matched word pool, paired test, this document | Merge into `main` after it |
| `drei_eme` | Branched before "Skip too-short recordings"; its loading fix is superseded by `main`'s | Retire; don't run results from it |

**The repository must be private** (Settings -> General -> Danger Zone -> Change visibility; only
the owner can do this). The blueprint and the corpus licences require it, and the participants'
recordings will be protected under RA 10173. `.gitignore` keeps `.wav`, `.flac`, `.npz`, `.db`,
`data/` and `results/` out of git; check `git status` before every commit anyway.

## 2. Everyone, once

```bash
git clone https://github.com/Crindo18/echo-bench.git
cd echo-bench
uv sync --all-packages && uv run pytest     # every test should pass
```

Copy your existing `data/` folder (manifests, embeddings) into the clone if you have one; it is
never in git, so share it by copying.

## 3. Every task: branch, commit, merge

```bash
git switch main && git pull                 # start from the newest code
git switch -c <short-task-name>             # e.g. torgo-moonshine
# ... work ...
uv run pytest                               # before every commit
git add -A && git commit -m "What changed and why"
git push -u origin <short-task-name>        # then open a pull request on GitHub
```

Results in the thesis should name the commit they came from (`git rev-parse --short HEAD`);
the results database records it automatically once the code is a git clone.

## 4. Who does what

Fill in one owner per task in the group chat before starting; one owner means nobody repeats it.

| Task | Needs | Owner |
|---|---|---|
| Rerun the UASpeech check with the matched word pool and paired test (command below) | Brendan's saved `data/` folder | |
| Same check on TORGO, including Moonshine | TORGO ingested | |
| Moonshine INT8 encoder: accuracy vs its FP32 version (gate G2c) | transformers + onnxruntime | |
| A test for the Moonshine loader in `packages/echo-train/tests/` | training environment | |
| Verify `configs/datasets/uaspeech_speakers.csv` against README_extra.docx | the UASpeech download | |
| UASpeech-only fold plan (copy of `sgkf5-severity-s42.yaml`, `datasets: [uaspeech]`, new name) | UASpeech ingested | |
| Pi 5: `scripts/pi/setup_pi.sh`, then `perf_full.yaml` and the thread sweep (M2) | the Pi | |
| Correct Section 6 of the UASpeech/Moonshine progress report to describe `legacy_renames()` | - | |
| Chapters 3-5: setup, datasets, preprocessing, configurations, quantization, metrics, results | the above | |

## 5. The UASpeech rerun (seconds, no audio needed)

From the repository root, with the four `.npz` files and `uaspeech-v1.jsonl` in `data/`. Put the
thesis model first: every other encoder is compared with it speaker by speaker.

```bash
uv run echo-bench fewshot quick --k 1 --k 2 --episodes 100 \
    --embeddings data/embeddings/uaspeech-v1__ls100-ebf.npz \
    --embeddings data/embeddings/uaspeech-v1__ls100-conformer.npz \
    --embeddings data/embeddings/uaspeech-v1__moonshine-tiny.npz \
    --embeddings data/embeddings/uaspeech-v1__ebf-random.npz \
    --csv reports/uaspeech-v1_fewshot_per_speaker.csv
```

Add `--pool all` to reproduce the earlier report's tables exactly (to show what changed).
Report the Holm-corrected p-values when claiming one encoder is better.
