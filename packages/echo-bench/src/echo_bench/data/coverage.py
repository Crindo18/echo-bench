"""Which enrollment sizes K each speaker's data can support (blueprint M3 coverage report).

A class supports K-shot evaluation for a speaker when that speaker has at least K + 1
distinct recordings of it (K for the support set, at least one left as a query).
Only primary-channel recordings count, since other mics are the same utterance.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

K_LEVELS = (3, 5, 10)


@dataclass(frozen=True)
class SpeakerCoverage:
    dataset: str
    speaker: str
    cohort: str
    severity_tier: str | None
    classes: int
    median_recordings: float
    feasible: dict[int, int]  # K -> number of classes with >= K + 1 recordings


def coverage(db: Session, dataset: str | None = None) -> list[SpeakerCoverage]:
    rows = db.execute(
        text(
            """
            SELECT d.name, s.code, s.cohort, s.severity_tier, u.label_id,
                   COUNT(DISTINCT u.recording_group_id)
            FROM utterance u JOIN speaker s ON s.id = u.speaker_id JOIN dataset d ON d.id = u.dataset_id
            WHERE u.is_primary_channel = 1 AND (:dataset IS NULL OR d.name = :dataset)
            GROUP BY u.speaker_id, u.label_id
            """
        ),
        {"dataset": dataset},
    ).all()
    per_speaker: dict[tuple[str, str, str, str | None], list[int]] = defaultdict(list)
    for name, code, cohort, tier, _label, n in rows:
        per_speaker[(name, code, cohort, tier)].append(n)
    return [
        SpeakerCoverage(
            dataset=name,
            speaker=code,
            cohort=cohort,
            severity_tier=tier,
            classes=len(counts),
            median_recordings=statistics.median(counts),
            feasible={k: sum(n >= k + 1 for n in counts) for k in K_LEVELS},
        )
        for (name, code, cohort, tier), counts in sorted(per_speaker.items())
    ]
