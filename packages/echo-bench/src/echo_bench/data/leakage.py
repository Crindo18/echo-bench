"""Leakage and consistency checks (blueprint §6.4 rules L1 and L2; `echo-bench data validate`).

    L1  No speaker is on both the training and the test side within a fold.
    L2  No recording group (one physical utterance on several mics) is split
        across sides, and every group has exactly one speaker, label and primary channel.

L3-L6 concern calibration, fold-matched models, augmentation and prototypes; they
are enforced where those steps happen (M4-M5).
"""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Issue:
    rule: str
    level: Literal["error", "warning"]
    message: str


def _groups(db: Session) -> list[Issue]:
    rows = db.execute(
        text(
            """
            SELECT d.name, u.recording_group_id,
                   COUNT(DISTINCT u.speaker_id), COUNT(DISTINCT u.label_id),
                   COUNT(DISTINCT COALESCE(u.session, '')), SUM(u.is_primary_channel)
            FROM utterance u JOIN dataset d ON d.id = u.dataset_id
            GROUP BY u.dataset_id, u.recording_group_id
            HAVING COUNT(DISTINCT u.speaker_id) > 1 OR COUNT(DISTINCT u.label_id) > 1
                OR COUNT(DISTINCT COALESCE(u.session, '')) > 1 OR SUM(u.is_primary_channel) != 1
            LIMIT 50
            """
        )
    ).all()
    return [
        Issue(
            "L2",
            "error",
            f"{name}: recording group {group} has {spk} speakers, {lab} labels, "
            f"{ses} sessions and {prim} primary channels (needs 1, 1, 1, 1)",
        )
        for name, group, spk, lab, ses, prim in rows
    ]


def _speakers(db: Session) -> list[Issue]:
    rows = db.execute(
        text(
            """
            SELECT d.name, s.code FROM speaker s JOIN dataset d ON d.id = s.dataset_id
            WHERE s.cohort = 'dysarthric' AND s.severity_tier IS NULL
            """
        )
    ).all()
    return [
        Issue(
            "data",
            "warning",
            f"{name}: dysarthric speaker {code} has no severity tier (stratification)",
        )
        for name, code in rows
    ]


def _durations(db: Session) -> list[Issue]:
    rows = db.execute(
        text(
            """
            SELECT d.name, COUNT(*) FROM utterance u JOIN dataset d ON d.id = u.dataset_id
            WHERE u.is_primary_channel = 1 AND u.duration_ms < 100 GROUP BY d.name
            """
        )
    ).all()
    return [
        Issue(
            "data",
            "warning",
            f"{name}: {n} recording(s) shorter than 0.1 s (probably not speech); "
            "encoders can't process them and `echo-train embed` skips them",
        )
        for name, n in rows
    ]


def _plans(db: Session) -> list[Issue]:
    issues: list[Issue] = []
    plans = db.execute(
        text("SELECT id, name, k, dataset_ids_json FROM cv_plan ORDER BY name")
    ).all()
    for plan_id, plan_name, k, dataset_ids in plans:
        expected = db.execute(
            text(
                "SELECT COUNT(DISTINCT s.id) FROM speaker s "
                "WHERE s.dataset_id IN (SELECT value FROM json_each(:ids)) "
                "AND EXISTS (SELECT 1 FROM utterance u WHERE u.speaker_id = s.id)"
            ),
            {"ids": dataset_ids},
        ).scalar_one()
        per_fold = db.execute(
            text(
                "SELECT fold_index, COUNT(*), SUM(role = 'test'), SUM(role = 'train') "
                "FROM fold_assignment WHERE cv_plan_id = :p GROUP BY fold_index ORDER BY fold_index"
            ),
            {"p": plan_id},
        ).all()
        if len(per_fold) != k:
            issues.append(
                Issue("L1", "error", f"{plan_name}: {len(per_fold)} folds stored, plan says {k}")
            )
        for fold, assigned, n_test, n_train in per_fold:
            if assigned != expected:
                issues.append(
                    Issue(
                        "L1",
                        "error",
                        f"{plan_name} fold {fold}: {assigned} of {expected} speakers assigned",
                    )
                )
            if not n_test or not n_train:
                issues.append(
                    Issue("L1", "error", f"{plan_name} fold {fold}: no test or no train speakers")
                )
        tested = db.execute(
            text(
                "SELECT s.code, COUNT(*) FROM fold_assignment f JOIN speaker s ON s.id = f.speaker_id "
                "WHERE f.cv_plan_id = :p AND f.role = 'test' GROUP BY f.speaker_id HAVING COUNT(*) != 1"
            ),
            {"p": plan_id},
        ).all()
        issues += [
            Issue(
                "L1",
                "error",
                f"{plan_name}: speaker {code} is a test speaker in {n} folds (needs 1)",
            )
            for code, n in tested
        ]
        split_groups = db.execute(
            text(
                """
                SELECT f.fold_index, u.recording_group_id FROM utterance u
                JOIN fold_assignment f ON f.speaker_id = u.speaker_id AND f.cv_plan_id = :p
                GROUP BY f.fold_index, u.recording_group_id
                HAVING COUNT(DISTINCT f.role) > 1 LIMIT 20
                """
            ),
            {"p": plan_id},
        ).all()
        issues += [
            Issue(
                "L2",
                "error",
                f"{plan_name} fold {fold}: recording group {group} is split across roles",
            )
            for fold, group in split_groups
        ]
    return issues


def validate(db: Session) -> list[Issue]:
    return _groups(db) + _speakers(db) + _durations(db) + _plans(db)
