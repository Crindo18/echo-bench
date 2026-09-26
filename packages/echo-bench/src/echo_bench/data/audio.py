"""Audio file metadata without decoding the whole file: sample rate, duration, SHA-256."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    duration_ms: int
    sha256: str


def audio_info(path: Path) -> AudioInfo | None:
    """None if the file can't be read or holds no audio (the ingester counts these)."""
    try:
        info = sf.info(str(path))
    except (RuntimeError, TypeError, ValueError):
        return None
    if info.frames <= 0 or info.samplerate <= 0:
        return None
    duration_ms = max(1, round(info.frames * 1000 / info.samplerate))
    return AudioInfo(
        int(info.samplerate), duration_ms, hashlib.sha256(path.read_bytes()).hexdigest()
    )
