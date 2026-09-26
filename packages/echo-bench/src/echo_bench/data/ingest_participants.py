"""Participant recordings (defined in M3, used from M7): the record schema and the
>=24 h rule between enrollment and test sessions (blueprint M3, §6.4).

Participant data is restricted (RA 10173): codes only (P01, P02, ...), no names.
Every record must carry recorded_at, and sessions are named enroll_<n> or test_<n>.
"""

import re
from collections import defaultdict
from datetime import datetime, timedelta

from pydantic import field_validator

from echo_bench.data.manifest import UtteranceRecord

SESSION_NAME = re.compile(r"^(enroll|test)_\d+$")


class ParticipantRecord(UtteranceRecord):
    recorded_at: str  # required here: ISO-8601 UTC

    @field_validator("speaker")
    @classmethod
    def _pseudonymous(cls, value: str) -> str:
        if not re.fullmatch(r"P\d{2,3}", value):
            raise ValueError("participant codes look like P01; never store names")
        return value

    @field_validator("session")
    @classmethod
    def _session_name(cls, value: str | None) -> str | None:
        if value is None or not SESSION_NAME.match(value):
            raise ValueError("session must be enroll_<n> or test_<n>")
        return value


def _parse(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def check_test_separation(records: list[ParticipantRecord], min_hours: float = 24) -> list[str]:
    """Problems where a participant's first test recording comes less than min_hours
    after their last enrollment recording (the chronological protocol's separation)."""
    enroll: dict[str, list[datetime]] = defaultdict(list)
    test: dict[str, list[datetime]] = defaultdict(list)
    for record in records:
        target = enroll if (record.session or "").startswith("enroll") else test
        target[record.speaker].append(_parse(record.recorded_at))
    problems = []
    for speaker in sorted(test):
        if not enroll.get(speaker):
            problems.append(f"{speaker}: test recordings but no enrollment session")
            continue
        gap = min(test[speaker]) - max(enroll[speaker])
        if gap < timedelta(hours=min_hours):
            hours = gap.total_seconds() / 3600
            problems.append(
                f"{speaker}: test starts {hours:.1f} h after enrollment (needs >= {min_hours:g} h)"
            )
    return problems
