"""Speaker tables: configs/datasets/<corpus>_speakers.csv (blueprint §4.3, speaker table).

One row per speaker code with cohort, severity tier, intelligibility and sex.
Lines starting with # are comments (sources and notes). Severity tiers drive the
stratified CV plan, so every value must come from the corpus documentation or a
cited paper, never from guesswork.
"""

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SpeakerMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    cohort: Literal["dysarthric", "control", "participant", "voice_actor"]
    severity_tier: str | None = None
    intelligibility_pct: float | None = Field(default=None, ge=0, le=100)
    sex: Literal["F", "M"] | None = None


def load_speakers(path: Path) -> dict[str, SpeakerMeta]:
    rows = [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]
    speakers: dict[str, SpeakerMeta] = {}
    for row in csv.DictReader(rows):
        values = {key: (value.strip() or None) for key, value in row.items() if key}
        meta = SpeakerMeta.model_validate(values)
        if meta.code in speakers:
            raise ValueError(f"{path}: speaker {meta.code} appears twice")
        speakers[meta.code] = meta
    return speakers
