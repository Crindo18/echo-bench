"""Small helpers for reading and writing rows (blueprint section 5).

M1 adds batched inserts for samples and the campaign/run lifecycle here.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from echo_bench.db.models import HardwareProfile, SchemaMeta, SoftwareEnv


def get_or_create_hardware_profile(session: Session, fields: dict[str, Any]) -> HardwareProfile:
    """Return the stored profile with this fingerprint, creating it the first time."""
    query = select(HardwareProfile).where(HardwareProfile.fingerprint == fields["fingerprint"])
    found = session.scalar(query)
    if found is not None:
        return found
    profile = HardwareProfile(**fields)
    session.add(profile)
    session.flush()
    return profile


def get_or_create_software_env(session: Session, fields: dict[str, Any]) -> SoftwareEnv:
    """Return the stored environment with this fingerprint, creating it the first time."""
    query = select(SoftwareEnv).where(SoftwareEnv.fingerprint == fields["fingerprint"])
    found = session.scalar(query)
    if found is not None:
        return found
    env = SoftwareEnv(**fields)
    session.add(env)
    session.flush()
    return env


def set_meta_if_missing(session: Session, key: str, value: str) -> None:
    if session.get(SchemaMeta, key) is None:
        session.add(SchemaMeta(key=key, value=value))
