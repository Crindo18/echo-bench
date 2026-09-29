# UASpeech early accuracy check: E-Branchformer vs Conformer vs Moonshine Tiny vs random weights

**Run date:** 2026-09-29 · **Commit:** `b6a4074` · **Dataset version:** `uaspeech-v1` (manifest sha256 `bb5a4d362892…`)

> **Scope.** This is a *quick check*, not the thesis protocol (M5, speaker-independent 5-fold CV).
> All three trained encoders are **pretrained on LibriSpeech (typical read English) with no
> dysarthria training**. They are used frozen: nothing is trained or tuned on UASpeech. Each
> recording becomes one embedding (masked mean of the encoder frames), and words are classified
> with Prototypical Networks in 10-way episodes (chance ≈ 10%).

## 1. Headline results (matched word pool, 15 dysarthric speakers)

| Encoder | SI | K=1 | K=2 |
|---|---|---|---|
| E-Branchformer (LibriSpeech 100 h), **reference** | 63.6% | 66.8% | 74.1% |
| Conformer (LibriSpeech 100 h) | 65.0% | 70.2% | 78.4% |
| Moonshine Tiny (encoder only, 7.68 M params) | **76.5%** | **71.1%** | **79.2%** |
| E-Branchformer, random weights (floor) | 35.6% | 41.1% | 51.7% |

1. **E-Branchformer is not the best of the pretrained encoders in this check.** Conformer
   beats it at K=1 and K=2 (+3.4 and +4.3 points, 14 of 15 speakers, Holm p = 0.002), and
   Moonshine Tiny beats it in every condition (+12.9 points SI, +4.3 K=1, +5.1 K=2, Holm
   p ≤ 0.005). At SI, E-Branchformer and Conformer are not significantly different (Holm p = 0.454).
2. **Pretraining matters a lot.** Every trained encoder is 22 to 41 points above the
   random-weight floor, and the random-weight version of E-Branchformer is significantly worse
   than the trained one in all conditions (Holm p ≤ 0.017).
3. **Personalization helps the most severe speakers.** With E-Branchformer, very-low
   intelligibility speakers go from 23.5% (SI) to 43.5% (K=2) and low-intelligibility speakers
   from 57.7% to 78.4%. High-intelligibility speakers and controls are already at 95 to 98%
   with SI, so there is little room to gain.
4. **K=1 is not reliably better than SI.** With E-Branchformer, K=1 beats SI for 6 of 15
   dysarthric speakers and K=2 for 9 of 15. For Moonshine Tiny, whose SI is strongest, K=1 is
   *lower* than SI on average (71.1% vs 76.5%). One recording per word is a noisy prototype.
   Other speakers' recordings, pooled together, often make a better one. K=1 wins mainly for
   the most severe speakers, whose speech is least like anyone else's.

**What this means for the thesis.** The data does not support "E-Branchformer is the best
encoder for dysarthric few-shot recognition" *as a pretrained, frozen encoder*. The thesis
claim may still hold after fine-tuning on dysarthric speech (M5), or once size and latency on
the Raspberry Pi are weighed in. Moonshine Tiny, the smallest model here, is a strong
competitor on both accuracy and size, and should be included in the M5 comparison.

## 2. Results by severity group (matched word pool)

Intelligibility tiers from the UASpeech speaker table: very_low 0–25%, low 26–50%, mid 51–75%, high 76–100%.

**E-Branchformer (reference)**

| Group | Speakers | SI | K=1 | K=2 |
|---|---|---|---|---|
| very_low | 4 | 23.5% | 35.2% | 43.5% |
| low | 3 | 57.7% | 68.0% | 78.4% |
| mid | 3 | 69.0% | 67.0% | 75.6% |
| high | 5 | 95.8% | 91.2% | 95.1% |
| control | 13 | 97.6% | 97.3% | 98.6% |
| **all dysarthric** | 15 | 63.6% | 66.8% | 74.1% |

**Conformer**

| Group | Speakers | SI | K=1 | K=2 |
|---|---|---|---|---|
| very_low | 4 | 24.1% | 38.0% | 49.1% |
| low | 3 | 61.2% | 71.3% | 82.2% |
| mid | 3 | 72.6% | 73.8% | 83.6% |
| high | 5 | 95.3% | 93.2% | 96.4% |
| control | 13 | 97.8% | 97.6% | 98.7% |
| **all dysarthric** | 15 | 65.0% | 70.2% | 78.4% |

**Moonshine Tiny**

| Group | Speakers | SI | K=1 | K=2 |
|---|---|---|---|---|
| very_low | 4 | 40.3% | 42.0% | 52.3% |
| low | 3 | 77.2% | 72.5% | 82.6% |
| mid | 3 | 88.5% | 73.0% | 82.5% |
| high | 5 | 97.8% | 92.3% | 96.6% |
| control | 13 | 98.8% | 98.2% | 99.2% |
| **all dysarthric** | 15 | 76.5% | 71.1% | 79.2% |

**E-Branchformer, random weights**

| Group | Speakers | SI | K=1 | K=2 |
|---|---|---|---|---|
| very_low | 4 | 27.0% | 32.8% | 42.3% |
| low | 3 | 36.4% | 48.5% | 59.3% |
| mid | 3 | 34.2% | 34.8% | 44.1% |
| high | 5 | 42.8% | 47.0% | 59.3% |
| control | 13 | 51.6% | 70.7% | 80.0% |
| **all dysarthric** | 15 | 35.6% | 41.1% | 51.7% |

## 3. Paired per-speaker test (matched pool)

Each encoder minus E-Branchformer, per dysarthric speaker (n = 15). Wilcoxon signed-rank,
two-sided, exact; Holm-corrected across all 9 rows. "Better/worse/same" counts speakers where
the other encoder is better than, worse than, or equal to E-Branchformer.

| Encoder | Condition | Mean diff | Better/worse/same | p | p (Holm) |
|---|---|---|---|---|---|
| Conformer | SI | +1.4 pts | 6/9/0 | 0.454 | 0.454 |
| Conformer | K=1 | +3.4 pts | 14/1/0 | <0.001 | 0.002 |
| Conformer | K=2 | +4.3 pts | 13/1/1 | <0.001 | 0.002 |
| Moonshine Tiny | SI | +12.9 pts | 13/2/0 | <0.001 | 0.004 |
| Moonshine Tiny | K=1 | +4.3 pts | 12/3/0 | 0.001 | 0.005 |
| Moonshine Tiny | K=2 | +5.1 pts | 13/2/0 | <0.001 | 0.003 |
| E-Branchformer (random) | SI | −28.0 pts | 4/11/0 | 0.008 | 0.017 |
| E-Branchformer (random) | K=1 | −25.7 pts | 3/12/0 | <0.001 | 0.004 |
| E-Branchformer (random) | K=2 | −22.4 pts | 3/12/0 | 0.001 | 0.005 |

This test compares **encoders**. It does not test SI against K within one encoder; the
K-vs-SI observations in section 1 are descriptive (speaker counts), not significance tests.

## 4. Old setup (`--pool all`) and what changed

All dysarthric speakers, `--pool all` (this reproduces the setup of the earlier UASpeech report):

| Encoder | SI | K=1 | K=2 |
|---|---|---|---|
| E-Branchformer | 63.2% | 67.2% | 73.8% |
| Conformer | 64.8% | 70.5% | 77.5% |
| Moonshine Tiny | 73.3% | 70.8% | 78.3% |
| E-Branchformer, random weights | 33.8% | 41.1% | 51.2% |

Paired test under `--pool all` (each minus E-Branchformer, Holm p): Conformer +1.6 SI (0.015),
+3.3 K=1 (0.002), +3.7 K=2 (0.002); Moonshine +10.1 SI (0.003), +3.6 K=1 (0.013), +4.5 K=2
(0.003); random −29.4 SI (0.003), −26.2 K=1 (0.003), −22.6 K=2 (0.003).

**What changed, in plain language.** UASpeech has two kinds of words:
- **155 words recorded three times each** by every speaker (digits, letters, computer
  commands, common words)
- **300 "uncommon" words recorded only once each**

A personal test (K=1 or K=2) needs at least one extra recording to use as the test item, so
it can only use the 155 repeated words. Under the old setup, the SI test drew from **all 455
words** while K drew from **only the 155**. So "K minus SI" mixed two things: the benefit of
the speaker's own recordings, and a change in which words were being tested.

The corrected setup (`matched`, the default) makes SI and every K use **the same 155 words, and
the same 10 words in each episode**. The difference between SI and K then measures
personalization and nothing else.

**How big the effect was here.** With these encoders the old setup moved the averages only a
little. E-Branchformer's SI went from 63.2% to 63.6%, and Moonshine's from 73.3% to 76.5%. The
ranking of encoders is the same under both setups. The one notable change is the SI
comparison of Conformer vs E-Branchformer: under the old setup it looked significant
(Holm p = 0.015), but on matched words it is not (p = 0.454). **Under `--pool all`, do not
compare SI with K**; compare encoders within one column only.

Brendan's earlier numbers came from different embedding files, so compare them with the
`--pool all` table above, not with section 1. Any difference beyond rounding would point to a
difference in data, model loading or code version.

## 5. Data and ingest

| Check | Expected | Result |
|---|---|---|
| Speaker folders in `audio/original` | 28 (15 dysarthric + 13 control) | 28 ✓ |
| Noisereduce / normalized copies inside | none | none ✓ |
| Files used (primary microphone M5) | 21,420 | 21,420 ✓ |
| Speakers | 28 | 28 ✓ (control 13, very_low 4, low 3, mid 3, high 5) |
| Classes | 455 | 455 ✓ |
| Recordings per speaker | 765 | 765 for every speaker ✓ |
| `echo-bench data validate` | no leakage | "No leakage found" ✓ |
| Recordings skipped as too short | — | 0 for every encoder |

`data validate` reported 0 CV plans checked, since no plan is stored yet, so speaker-overlap
rule L1 had nothing to check. The quick check does not train anything. Its own rules hold by
construction: SI prototypes exclude the target speaker (L6), and there is one microphone per
recording, so support and query never share a recording (L2).

## 6. Speaker-table check

`README_extra.docx` contains **no intelligibility or severity figures**. It lists which
speakers are distributed and why F01, M02, M03, M06 and M13 are excluded. The corpus's own
speaker table is the "Speaker" sheet of `doc/speaker_wordlist.xls`, as `readme_UASpeech.txt`
points out. It matches `configs/datasets/uaspeech_speakers.csv` for **all 15 dysarthric speakers**:

| Speaker | Official (speaker_wordlist.xls) | CSV | Match |
|---|---|---|---|
| **M01** | Very low (**15%**) | very_low, **15** | ✓ (17% from another source is not the official figure) |
| M04 | Very low (2%) | very_low, 2 | ✓ |
| F03 | Very low (6%) | very_low, 6 | ✓ |
| M12 | Very low (7.4%) | very_low, 7.4 | ✓ |
| M07 | Low (28%) | low, 28 | ✓ |
| F02 | Low (29%) | low, 29 | ✓ |
| M16 | Low (43%) | low, 43 | ✓ |
| M05 | Mid (58%) | mid, 58 | ✓ |
| F04 | Mid (62%) | mid, 62 | ✓ |
| M11 | Mid (62%) | mid, 62 | ✓ |
| M09 | High (86%) | high, 86 | ✓ |
| M14 | High (90.4%) | high, 90.4 | ✓ |
| M08 | High (93%) | high, 93 | ✓ |
| M10 | High (93%) | high, 93 | ✓ |
| F05 | High (95%) | high, 95 | ✓ |

No change to the CSV is needed. Its "VERIFY" comment can cite `doc/speaker_wordlist.xls` as the source.

## 7. Encoders and embedding runs

| Encoder | Source | Weights check | Embedding size | Time (21,420 files) |
|---|---|---|---|---|
| E-Branchformer | `hf:pyf98/librispeech_100_e_branchformer` | all 503 encoder tensors loaded (2 renamed from an older ESPnet layout) | 256 | 2,174 s |
| Conformer | `hf:pyf98/librispeech_100h_conformer` | all 491 encoder tensors loaded (2 renamed) | 256 | 1,528 s |
| Moonshine Tiny | `moonshine:moonshine-ai/moonshine-tiny` | all 68 encoder tensors loaded; 7.68 M encoder parameters | 288 | 947 s |
| E-Branchformer, random | same architecture, `--random-weights` | n/a | 256 | 1,281 s |

"Renamed" tensors come from `legacy_renames()`: layers whose names changed between ESPnet
versions while the layer stayed the same. A tensor is renamed only when the name and shape
match exactly one missing tensor; otherwise loading refuses.

## 8. Machine and software

- **Hardware:** Intel Core i5-12450H, 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB, driver 581.83)
- **OS:** Windows 11 Home 10.0.26200 (native Windows, not WSL)
- **Software:** Python 3.12.14, uv 0.12.20, numpy 2.5.3; echo-train: torch 2.11.0+cu128, ESPnet 202610.post2, transformers 5.17.0

**Deviations from the Linux instructions, none of which changed code or config:**
- **PyTorch:** PyPI's Windows torch wheels are CPU-only, so torch/torchaudio 2.11.0+cu128 were
  installed into `packages/echo-train/.venv` from the PyTorch index. echo-train commands
  therefore run with `uv run --no-sync`, because a plain `uv run` would reinstall the CPU build.
- **Environment variables:** all commands ran with `PYTHONUTF8=1` (the Windows console could
  not print ✓) and `COLUMNS=200`. At the default width, one test
  (`test_perf_refuses_a_changed_model_file`) fails only because long Windows paths wrap its
  error message. With `COLUMNS=200`, all 79 tests pass.
- **Background run:** the four embedding runs ran as one detached process instead of `nohup`.
  Logs are in `results/logs/`.

## 9. Commands

From the repository root unless noted. `~/corpora/UASpeech/audio/original` = `C:\Users\grego\corpora\UASpeech\audio\original`.

```bash
export PYTHONUTF8=1 COLUMNS=200
uv sync --all-packages
uv run pytest
uv run echo-bench db init
uv run echo-bench data ingest uaspeech --root ~/corpora/UASpeech/audio/original
uv run echo-bench data validate

cd packages/echo-train
UV_HTTP_TIMEOUT=600 uv sync
uv pip install "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
M=../../data/manifests/uaspeech-v1.jsonl; R=~/corpora/UASpeech/audio/original; E=../../data/embeddings
uv run --no-sync echo-train embed --manifest $M --root $R --out $E/uaspeech-v1__ls100-ebf.npz       --model hf:pyf98/librispeech_100_e_branchformer --device cuda
uv run --no-sync echo-train embed --manifest $M --root $R --out $E/uaspeech-v1__ls100-conformer.npz --model hf:pyf98/librispeech_100h_conformer --device cuda
uv run --no-sync echo-train embed --manifest $M --root $R --out $E/uaspeech-v1__moonshine-tiny.npz  --model moonshine:moonshine-ai/moonshine-tiny --device cuda
uv run --no-sync echo-train embed --manifest $M --root $R --out $E/uaspeech-v1__ebf-random.npz      --model hf:pyf98/librispeech_100_e_branchformer --random-weights --device cuda
cd ../..

uv run echo-bench fewshot quick --k 1 --k 2 --episodes 100 \
  --embeddings data/embeddings/uaspeech-v1__ls100-ebf.npz \
  --embeddings data/embeddings/uaspeech-v1__ls100-conformer.npz \
  --embeddings data/embeddings/uaspeech-v1__moonshine-tiny.npz \
  --embeddings data/embeddings/uaspeech-v1__ebf-random.npz \
  --csv reports/uaspeech-v1_fewshot_per_speaker.csv
uv run echo-bench fewshot quick --k 1 --k 2 --episodes 100 --pool all \
  --embeddings data/embeddings/uaspeech-v1__ls100-ebf.npz \
  --embeddings data/embeddings/uaspeech-v1__ls100-conformer.npz \
  --embeddings data/embeddings/uaspeech-v1__moonshine-tiny.npz \
  --embeddings data/embeddings/uaspeech-v1__ebf-random.npz \
  --csv reports/uaspeech-v1_fewshot_pool-all.csv
```

## 10. Files

| File | Contents |
|---|---|
| `reports/uaspeech-v1_fewshot_per_speaker.csv` | Matched pool, per speaker × encoder (SI, K=1, K=2) |
| `reports/uaspeech-v1_fewshot_pool-all.csv` | Old `--pool all` setup, same layout |
| `results/logs/fewshot_matched.log`, `fewshot_pool-all.log` | Full printed output of both runs |
| `results/logs/embed_*.log`, `embed_status.txt` | Embedding runs and timings |
| `data/embeddings/uaspeech-v1__*.npz` | The four embedding files (cached; rerunning the few-shot step takes seconds) |

None of these files are committed: `data/`, `results/` and `reports/` are git-ignored, and the
corpus is licensed.
