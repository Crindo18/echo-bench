# Can the encoder run on a Raspberry Pi 5 (4 GB)?

Evidence from the desktop (M1), September 2026, for the random-weight 8-block
E-Branchformer in `configs/models/ebf12m.yaml` (13.06 M parameters). Pi speed is an
estimate until M2 measures it. Regenerate the numbers with `scripts/m1_slice.sh` and
`echo-bench model inspect <name:variant>`.

## What "fits" means, and what the desktop can already answer

| Question | Gate | Where it can be answered | Answer so far |
|---|---|---|---|
| Is the file small enough? | G1: INT8 file ≤ 16 MB | Anywhere: a file is the same size on every machine | Yes: 14.97 MB (dynamic INT8), 15.32 MB (static INT8). Margin under 1.1 MB. |
| Does it fit in memory? | G6: whole system ≤ 2 GB of the 4 GB | Desktop, closely: same file, same ONNX Runtime | Yes: about 150 MB peak for static INT8, including Python and ONNX Runtime (about 200 MB for FP32 or dynamic INT8), with 1-10 s inputs. |
| How much arithmetic per utterance? | Predicts speed | Anywhere: a property of the model | 0.59 / 1.84 / 3.14 / 5.16 billion multiply-accumulates (MACs) for 1 / 3 / 5 / 8 s. |
| Is it fast enough on the Pi? | G5a: end-to-end p95 ≤ 1.5 s | Only the Pi (M2) | Estimated encoder time: about 50-100 ms for a 3 s phrase and 130-260 ms for 8 s (method below). |
| Does the ONNX file compute what PyTorch computes? | G2a: ≤ 1e-4 | Desktop | Yes: largest difference 1.2e-6 over 1-10 s inputs. |
| Does it behave the same on the Pi? | G2d: ≥ 99.5% agreement | Desktop + Pi (M4) | Same file and same ONNX Runtime version; the small rounding differences are measured in M4. |
| Is it accurate on dysarthric speech? | G2c, G3a-G3d | Trained weights + data (M3-M5) | Unknown. This is the main open question, and it does not depend on the Pi. |

## How the speed estimate was made

Moonshine Tiny (reference R1, `docs/references.md`) publishes its encoder time on a
Raspberry Pi 5 with 4 threads: 50.7 ms for a 5 s window. Counting that encoder's
arithmetic from its published architecture (width 288, 6 layers, feed-forward 1152,
plus its convolutional audio stem) gives about 2.0 billion MACs, so the Pi 5 sustained
about 40 billion MACs per second on it. ECHO's encoder needs 3.14 billion MACs for
5 s, about 80 ms at the same efficiency. The ranges in the table allow ECHO to run up
to twice as slowly, because the runtime differs (ONNX Runtime vs LiteRT) and
E-Branchformer's depthwise convolutions and gating take time without doing much
arithmetic. M2 replaces the estimate with measurements.

## Where the arithmetic goes (5 s input)

About half is in the input stage, the convolutional subsampling that shrinks the
spectrogram before the blocks: 45% in its two convolutions and 5% in its output
projection, although it holds only 14% of the parameters. The eight blocks take the
other half: attention 16%, feed-forward 17%, convolution branch 13%, merge 4%. If the
Pi is slower than estimated, a slimmer input stage is the first thing to try (it needs
retraining).

## What about Zipformer?

Zipformer (Yao et al., ICLR 2024) is a faster encoder design: its middle layers run at
lower frame rates. It does not change the size picture:

- The binding limit is our own 16 MB objective (G1, SO1 D1), not the Pi. Memory is far
  from the limit: this encoder uses about 150 MB of the 4 GB.
- Ready-made Zipformer models are larger than 16 MB. In sherpa-onnx's catalogue the
  smallest listed streaming model (14 M parameters, Chinese) has a 21 MB INT8 encoder,
  and the English ones are 68-180 MB
  (https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html).
  A Zipformer for ECHO would also have to be trained small by us.
- Switching changes the training toolkit (icefall/k2 instead of ESPnet).

The harness is model-agnostic (assumption A1): any ONNX encoder whose inputs are
features `[B, T, F]` plus lengths `[B]` can be registered and measured with the same
commands, so a small Zipformer can be compared on the same gates.

Suggested decision rule: switch only if E-Branchformer fails a gate that the
alternative passes.
