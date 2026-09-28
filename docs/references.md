# External references

## R1: Moonshine Tiny (reference speech recognizer)

Our reviewer asked us to use Moonshine Tiny as a **reference** for ECHO. It is a point of
comparison, not part of the device pipeline.

| | |
|---|---|
| What it is | A 27M-parameter encoder-decoder speech-to-text model for English, by Moonshine AI (formerly Useful Sensors) |
| Link we were sent | https://huggingface.co/litert-community/moonshine-tiny (LiteRT, formerly TFLite, packaging) |
| Base model | https://huggingface.co/moonshine-ai/moonshine-tiny |
| ONNX versions | `moonshine-ai/moonshine` (onnx/merged/tiny) and `onnx-community/moonshine-tiny-ONNX` |
| Paper | Jeffries et al., *Moonshine: Speech Recognition for Live Transcription and Voice Commands*, arXiv:2410.15608 (2024) |
| License | MIT |

### Published numbers (from the model card, read 2026-09-25)

| Measure | f32 file | i8 file |
|---|---|---|
| File size | 109 MB | 52 MB (encoder kept in float32, decoder int8) |
| Raspberry Pi 5, one 5 s window with 11 output tokens, CPU, 4 threads: encode / decode / total | 50.7 / 444.7 / 495.3 ms | 50.3 / 267.4 / 317.7 ms |

Input is raw 16 kHz audio in fixed 5 s windows. Decode time grows with the number of words.

### Encoder-only size (measured 2026-09-27)

ECHO would use only Moonshine's encoder (it turns audio into frames for the prototypes); the
decoder, which writes out words, would be dropped. Measured on a desktop under WSL2
(i7-13620H), transformers 5.17.0. Parameter counts and file sizes don't depend on the machine.

Parameters, from `moonshine-ai/moonshine-tiny`:

| Part | Parameters |
|---|---|
| Whole model | 27.09 M |
| Encoder | 7.68 M |
| Decoder | 19.41 M |

Encoder files, from `onnx-community/moonshine-tiny-ONNX` (onnx/encoder_model*.onnx):

| File | Precision | Size | G1 (INT8 <= 16 MB) |
|---|---|---|---|
| encoder_model.onnx | FP32 | 30.88 MB | n/a (not INT8) |
| encoder_model_int8.onnx | INT8 | 7.92 MB | Pass |
| encoder_model_uint8.onnx | INT8 (unsigned) | 7.92 MB | Pass |
| encoder_model_quantized.onnx | INT8 | 7.94 MB | Pass |
| encoder_model_fp16.onnx | FP16 | 15.52 MB | n/a (not INT8) |
| encoder_model_q4 / bnb4 / q4f16 | 4-bit | 10.73 / 10.36 / 6.94 MB | n/a (not in the blueprint) |

The file sizes agree with the parameter count (7.68 M x 4 bytes = 30.7 MB in FP32, x 1 byte =
7.7 MB in INT8), which confirms the ONNX files hold the same encoder.

Compared with ECHO's E-Branchformer (M1, random weights, same size as trained):

| | Moonshine Tiny encoder | E-Branchformer |
|---|---|---|
| Parameters | 7.68 M | 13.06 M |
| FP32 file | 30.88 MB | 53.47 MB |
| INT8 file | 7.92 MB | 14.97-15.32 MB |

Not yet known: whether the INT8 encoder still produces good embeddings (Moonshine's own
release keeps the encoder in float32 because simple INT8 hurt its early layers), and
accuracy on dysarthric speech. Both need the TORGO check.

### How ECHO uses it

| When | What | Why |
|---|---|---|
| M2 | Run the model card's own script on our Pi 5 and compare with the published Pi 5 numbers | A known reference that checks our Pi setup (cooling, power) and gives a same-hardware speed comparison |
| M5-M6 | Transcribe the same dysarthric test utterances on the desktop, map each transcript to the closest of the 36 phrases, and score on the same folds | Accuracy of a ready-made recognizer versus ECHO's personalized few-shot approach, next to Ablations 1 and 5 (Whisper, MMS) |
| M9 | Report ECHO next to R1: file size, Pi latency, accuracy | Context for the thesis results |

### Caveats to state in the thesis

- Different job: R1 transcribes words; ECHO embeds an utterance and matches it to the user's own prototypes.
- Different engine: LiteRT versus ONNX Runtime (ECHO's only engine, ADR-1). Use an ONNX version if a same-engine comparison is needed.
- The full model is above ECHO's 16 MB budget (G1), but its encoder alone fits (7.92 MB in INT8,
  see "Encoder-only size"). As a full recognizer it is a reference; its encoder could be a
  candidate if the accuracy check supports it.
- English only, and trained and evaluated on typical (non-dysarthric) speech.