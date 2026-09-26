"""The JSONL manifest: one line per audio file, written by an ingester (blueprint §4.3).

A manifest is the reviewed, hashable record of exactly which files a dataset
version contains. The database stores its path and SHA-256, so any change to
the files becomes a new dataset version instead of a silent edit.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from echo_core.model_card import sha256_file


class UtteranceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: str  # pseudonymous code, e.g. F02
    label_key: str  # word code or normalized prompt text
    label_text: str
    category: str | None = None  # proxy:<group> for corpora; an intent category for participants
    is_emergency: bool = False
    recording_group_id: str  # the same physical utterance on every mic shares this id
    session: str | None = None  # TORGO session, UASpeech block, or participant session
    channel: str | None = None
    is_primary_channel: bool = True
    audio_path: str  # relative to the dataset root, with forward slashes
    audio_sha256: str
    sample_rate: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    recorded_at: str | None = None  # ISO-8601 UTC; required for participants


def write_manifest(records: Iterable[UtteranceRecord], path: Path) -> str:
    """Write records sorted by path (so the same files always give the same bytes); return SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(json.dumps(r.model_dump(), sort_keys=True) for r in records)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return sha256_file(path)


def read_manifest(path: Path) -> list[UtteranceRecord]:
    with path.open(encoding="utf-8") as handle:
        return [UtteranceRecord.model_validate_json(line) for line in handle if line.strip()]
