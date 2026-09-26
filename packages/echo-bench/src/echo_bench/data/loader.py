"""Manifest -> dataset, speaker, label and utterance rows (blueprint §4.3).

A dataset version is immutable: loading the same manifest again changes nothing,
and a different manifest under the same name and version is refused. Ingest
changed files as a new --version instead, so old CV plans keep pointing at the
exact data they were built from.
"""

import os
from pathlib import Path

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from echo_bench.data.ingest_common import IngestError
from echo_bench.data.manifest import read_manifest
from echo_bench.data.speakers import SpeakerMeta
from echo_bench.db.models import Dataset, Label, Speaker, Utterance, new_id
from echo_core.model_card import sha256_file


def load_dataset(
    db: Session,
    *,
    name: str,
    version: str,
    access_level: str,
    manifest_path: Path,
    speakers: dict[str, SpeakerMeta],
    primary_channel: str | None,
    root_hint: str | None = None,
) -> tuple[Dataset, bool]:
    """Returns (dataset row, newly_loaded)."""
    manifest_sha = sha256_file(manifest_path)
    existing = db.scalar(select(Dataset).where(Dataset.name == name, Dataset.version == version))
    if existing is not None:
        if existing.manifest_sha256 == manifest_sha:
            return existing, False
        raise IngestError(
            f"{name} {version} is already loaded from a different manifest. "
            "Datasets are immutable: ingest the changed files under a new --version."
        )
    records = read_manifest(manifest_path)
    try:
        stored_path = Path(os.path.relpath(manifest_path)).as_posix()
    except ValueError:
        stored_path = manifest_path.resolve().as_posix()
    dataset = Dataset(
        id=new_id(),
        name=name,
        version=version,
        access_level=access_level,
        root_hint=root_hint,
        manifest_path=stored_path,
        manifest_sha256=manifest_sha,
        primary_channel=primary_channel,
    )
    db.add(dataset)

    speaker_ids: dict[str, str] = {}
    for code in sorted({r.speaker for r in records}):
        meta = speakers[code]
        speaker_ids[code] = new_id()
        db.add(
            Speaker(
                id=speaker_ids[code],
                dataset_id=dataset.id,
                code=code,
                cohort=meta.cohort,
                severity_tier=meta.severity_tier,
                intelligibility_pct=meta.intelligibility_pct,
                sex=meta.sex,
            )
        )
    label_ids: dict[str, str] = {}
    for record in sorted(records, key=lambda r: r.label_key):
        if record.label_key not in label_ids:
            label_ids[record.label_key] = new_id()
            db.add(
                Label(
                    id=label_ids[record.label_key],
                    dataset_id=dataset.id,
                    key=record.label_key,
                    text=record.label_text,
                    category=record.category,
                    is_emergency=int(record.is_emergency),
                )
            )
    db.flush()
    rows = [
        {
            "id": new_id(),
            "dataset_id": dataset.id,
            "speaker_id": speaker_ids[r.speaker],
            "label_id": label_ids[r.label_key],
            "recording_group_id": r.recording_group_id,
            "session": r.session,
            "channel": r.channel,
            "is_primary_channel": int(r.is_primary_channel),
            "audio_path": r.audio_path,
            "audio_sha256": r.audio_sha256,
            "sample_rate": r.sample_rate,
            "duration_ms": r.duration_ms,
            "recorded_at": r.recorded_at,
        }
        for r in records
    ]
    for start in range(0, len(rows), 5000):
        db.execute(insert(Utterance), rows[start : start + 5000])
    return dataset, True
