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

### How ECHO uses it

| When | What | Why |
|---|---|---|
| M2 | Run the model card's own script on our Pi 5 and compare with the published Pi 5 numbers | A known reference that checks our Pi setup (cooling, power) and gives a same-hardware speed comparison |
| M5-M6 | Transcribe the same dysarthric test utterances on the desktop, map each transcript to the closest of the 36 phrases, and score on the same folds | Accuracy of a ready-made recognizer versus ECHO's personalized few-shot approach, next to Ablations 1 and 5 (Whisper, MMS) |
| M9 | Report ECHO next to R1: file size, Pi latency, accuracy | Context for the thesis results |

### Caveats to state in the thesis

- Different job: R1 transcribes words; ECHO embeds an utterance and matches it to the user's own prototypes.
- Different engine: LiteRT versus ONNX Runtime (ECHO's only engine, ADR-1). Use an ONNX version if a same-engine comparison is needed.
- Above ECHO's 16 MB budget (G1), so it is a reference, not a candidate model.
- English only, and trained and evaluated on typical (non-dysarthric) speech.
