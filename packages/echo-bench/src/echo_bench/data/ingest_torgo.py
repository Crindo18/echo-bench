"""TORGO -> manifest records (blueprint M3).

Expected layout (the F/, FC/, M/, MC/ grouping folders are optional):

    <root>/F/F01/Session1/prompts/0001.txt          the prompt text
    <root>/F/F01/Session1/wav_headMic/0001.wav      head-mounted microphone
    <root>/F/F01/Session1/wav_arrayMic/0001.wav     array microphone

One prompt id in one session is one physical recording, so both microphones share
a recording_group_id. Prompts that aren't words or sentences (picture descriptions
given as image paths, bracketed instructions such as "[relax your mouth ...]")
are skipped and counted.
"""

import re
from pathlib import Path

from echo_bench.data.audio import audio_info
from echo_bench.data.ingest_common import IngestError, IngestReport, Progress, normalize_text
from echo_bench.data.manifest import UtteranceRecord
from echo_bench.data.speakers import SpeakerMeta

SPEAKER_DIR = re.compile(r"^[FM]C?\d{2}$")
CHANNEL_DIRS = {"wav_headMic": "headMic", "wav_arrayMic": "arrayMic"}
NON_LEXICAL = re.compile(r"[\[\]/\\]|\.(jpg|jpeg|png|bmp)\b|^x+$", re.IGNORECASE)


def scan_torgo(
    root: Path,
    speakers: dict[str, SpeakerMeta],
    *,
    primary_channel: str = "headMic",
    all_channels: bool = False,
    progress: Progress | None = None,
) -> tuple[list[UtteranceRecord], IngestReport]:
    if primary_channel not in CHANNEL_DIRS.values():
        raise IngestError(f"primary channel must be one of {sorted(CHANNEL_DIRS.values())}")
    root = root.resolve()
    sessions = sorted(
        p for p in root.rglob("Session*") if p.is_dir() and SPEAKER_DIR.match(p.parent.name)
    )
    if not sessions:
        raise IngestError(f"no TORGO speaker/Session folders under {root}")
    unknown = sorted({p.parent.name for p in sessions} - set(speakers))
    if unknown:
        raise IngestError(f"speakers missing from the speaker table: {', '.join(unknown)}")

    preference = [primary_channel] + [c for c in CHANNEL_DIRS.values() if c != primary_channel]
    records: list[UtteranceRecord] = []
    report = IngestReport()
    for session_dir in sessions:
        speaker, session = session_dir.parent.name, session_dir.name
        prompts = session_dir / "prompts"
        if not prompts.is_dir():
            report.counts["skipped: session without a prompts folder"] += 1
            continue
        for prompt_file in sorted(prompts.glob("*.txt")):
            text = " ".join(prompt_file.read_text(encoding="utf-8", errors="replace").split())
            key = normalize_text(text)
            if not key or NON_LEXICAL.search(text):
                report.counts["skipped: prompt is not a word or sentence"] += 1
                continue
            available = {
                channel: session_dir / folder / f"{prompt_file.stem}.wav"
                for folder, channel in CHANNEL_DIRS.items()
                if (session_dir / folder / f"{prompt_file.stem}.wav").is_file()
            }
            if not available:
                report.counts["skipped: prompt without audio"] += 1
                continue
            # The primary is the first READABLE file in preference order, so an
            # unreadable preferred mic never leaves a recording without a primary.
            infos = {}
            for channel in (c for c in preference if c in available):
                info = audio_info(available[channel])
                if info is None:
                    report.counts["skipped: unreadable or empty audio"] += 1
                    continue
                infos[channel] = info
                if not all_channels:
                    break
            if not infos:
                continue
            primary = next(iter(infos))
            if primary != primary_channel:
                report.counts[f"primary fell back to {primary}"] += 1
            group = f"torgo:{speaker}:{session}:{prompt_file.stem}"
            for channel, info in infos.items():
                wav = available[channel]
                records.append(
                    UtteranceRecord(
                        speaker=speaker,
                        label_key=key,
                        label_text=key,
                        category="proxy:word" if " " not in key else "proxy:sentence",
                        recording_group_id=group,
                        session=session,
                        channel=channel,
                        is_primary_channel=channel == primary,
                        audio_path=wav.relative_to(root).as_posix(),
                        audio_sha256=info.sha256,
                        sample_rate=info.sample_rate,
                        duration_ms=info.duration_ms,
                    )
                )
                report.counts["used"] += 1
                if progress and report.counts["used"] % 2000 == 0:
                    progress(f"  {report.counts['used']} files read")
    return records, report
