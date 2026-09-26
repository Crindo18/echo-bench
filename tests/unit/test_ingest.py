"""Corpus ingesters on tiny fake corpora with the real layouts (M3)."""

from collections import Counter
from pathlib import Path

import pytest

from echo_bench.data.ingest_common import IngestError, normalize_text
from echo_bench.data.ingest_participants import ParticipantRecord, check_test_separation
from echo_bench.data.ingest_torgo import scan_torgo
from echo_bench.data.ingest_uaspeech import scan_uaspeech, word_category
from echo_bench.data.speakers import SpeakerMeta, load_speakers


def test_speaker_tables_are_valid(speaker_tables: dict[str, Path]) -> None:
    torgo, uaspeech = (
        load_speakers(speaker_tables["torgo"]),
        load_speakers(speaker_tables["uaspeech"]),
    )
    assert len(torgo) == 15 and sum(s.cohort == "dysarthric" for s in torgo.values()) == 8
    assert len(uaspeech) == 28 and sum(s.cohort == "dysarthric" for s in uaspeech.values()) == 15
    assert {s.severity_tier for s in uaspeech.values() if s.cohort == "dysarthric"} == {
        "very_low",
        "low",
        "mid",
        "high",
    }


def test_torgo(torgo_root: Path, speaker_tables: dict[str, Path]) -> None:
    records, report = scan_torgo(torgo_root, load_speakers(speaker_tables["torgo"]))
    keys = Counter(r.label_key for r in records)
    assert set(keys) == {
        "yes",
        "no",
        "the quick brown fox",
    }  # picture and instruction prompts skipped
    assert report.counts["skipped: prompt is not a word or sentence"] == 2 * 7
    assert report.counts["skipped: unreadable or empty audio"] == 1
    # M03: prompt 0002 has no head-mic file and 0005's head-mic file is unreadable
    assert report.counts["primary fell back to arrayMic"] == 2
    assert {
        r.channel for r in records if r.speaker == "M03" and r.label_key == "the quick brown fox"
    } == {"arrayMic"}
    assert all(r.is_primary_channel for r in records)  # primary channel only by default
    assert len({r.recording_group_id for r in records}) == len(records)
    f01_yes = [r for r in records if r.speaker == "F01" and r.label_key == "yes"]
    assert len(f01_yes) == 4 and {r.session for r in f01_yes} == {
        f"Session{n}" for n in range(1, 5)
    }


def test_torgo_all_channels_share_recording_groups(
    torgo_root: Path, speaker_tables: dict[str, Path]
) -> None:
    records, _ = scan_torgo(torgo_root, load_speakers(speaker_tables["torgo"]), all_channels=True)
    by_group: dict[str, list[bool]] = {}
    for r in records:
        by_group.setdefault(r.recording_group_id, []).append(r.is_primary_channel)
    assert all(flags.count(True) == 1 for flags in by_group.values())
    assert max(len(flags) for flags in by_group.values()) == 2


def test_unknown_speakers_stop_the_ingest(torgo_root: Path) -> None:
    only_one = {"F01": SpeakerMeta(code="F01", cohort="dysarthric", severity_tier="severe")}
    with pytest.raises(IngestError, match="missing from the speaker table"):
        scan_torgo(torgo_root, only_one)


def test_uaspeech(uaspeech_root: Path, speaker_tables: dict[str, Path]) -> None:
    records, report = scan_uaspeech(uaspeech_root, load_speakers(speaker_tables["uaspeech"]))
    assert report.counts["skipped: mic 1 (synchronization tone)"] == 5
    assert report.counts["primary fell back from M5"] == 3  # M08's LA has no M5 in any block
    assert {r.channel for r in records if r.speaker == "M08" and r.label_key == "LA"} == {"M2"}
    assert {r.label_key for r in records if r.category == "proxy:uncommon_words"} == {
        "B1_UW1",
        "B2_UW101",
        "B3_UW201",
    }
    cw1 = [r for r in records if r.speaker == "F02" and r.label_key == "CW1"]
    assert sorted(r.session for r in cw1) == ["B1", "B2", "B3"]  # common words: once per block


def test_uaspeech_refuses_two_copies_of_the_corpus(
    uaspeech_root: Path, speaker_tables: dict[str, Path]
) -> None:
    copy = uaspeech_root / "noisereduce" / "F02" / "F02_B1_CW1_M5.wav"
    copy.parent.mkdir(parents=True)
    copy.write_bytes((uaspeech_root / "F02" / "F02_B1_CW1_M5.wav").read_bytes())
    with pytest.raises(IngestError, match="two copies"):
        scan_uaspeech(uaspeech_root, load_speakers(speaker_tables["uaspeech"]))


def test_word_categories_and_text() -> None:
    assert (
        word_category("CW12") == "proxy:common_words"
        and word_category("C3") == "proxy:computer_commands"
    )
    assert normalize_text("  Yes. ") == "yes" and normalize_text("don't STOP!") == "don't stop"


def _participant(session: str, when: str) -> ParticipantRecord:
    return ParticipantRecord(
        speaker="P01",
        label_key="pain",
        label_text="I am in pain",
        recording_group_id=f"g-{when}",
        session=session,
        audio_path=f"{when}.wav",
        audio_sha256="0" * 64,
        sample_rate=16000,
        duration_ms=900,
        recorded_at=when,
    )


def test_participant_sessions_need_24_hours_between_enrollment_and_test() -> None:
    ok = [
        _participant("enroll_1", "2026-10-01T09:00:00Z"),
        _participant("test_1", "2026-10-02T09:00:00Z"),
    ]
    early = [
        _participant("enroll_1", "2026-10-01T09:00:00Z"),
        _participant("test_1", "2026-10-01T19:00:00Z"),
    ]
    assert check_test_separation(ok) == []
    assert "10.0 h" in check_test_separation(early)[0]
    with pytest.raises(ValueError, match="never store names"):
        ParticipantRecord.model_validate(
            _participant("enroll_1", "2026-10-01T09:00:00Z").model_dump() | {"speaker": "Juan"}
        )
