"""The data layer (blueprint M3): ingesting corpora, the CV plan's inputs, and leakage checks.

    manifest.py             JSONL manifest: one line per audio file
    speakers.py             speaker tables (configs/datasets/*_speakers.csv)
    audio.py                sample rate, duration and SHA-256 of a WAV file
    ingest_torgo.py         TORGO   -> manifest records
    ingest_uaspeech.py      UASpeech -> manifest records
    ingest_participants.py  the participant record schema and the >=24 h rule
    loader.py               manifest -> dataset, speaker, label, utterance rows
    leakage.py              rules L1 and L2 (section 6.4) plus data consistency
    coverage.py             feasible K per class and speaker

Making the CV plan needs scikit-learn, which is desktop-only, so it lives in
echo-analysis (`echo-bench cv plan`); everything here also runs on the Pi.
"""
