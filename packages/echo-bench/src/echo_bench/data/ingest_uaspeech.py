"""UASpeech -> manifest records (blueprint M3).

Files are named <speaker>_<block>_<word>_<mic>.wav, for example F02_B1_CW12_M5.wav.
Mic 1 carries a synchronization tone, so it is skipped; mics 2-8 record speech.
The default primary channel is M5, the single microphone used for per-word
prototypes in prior prototype-based UASpeech work. All mics of one word in one
block share a recording_group_id.

Label keys are the word codes. Uncommon-word codes (UW) get their block as a prefix
(B2_UW15), which keeps them unique whether the release numbers them per block or
overall. Word groups become proxy categories (digits, radio alphabet, computer
commands, common and uncommon words), clearly marked as proxies (blueprint §6.5).
The official word list is a spreadsheet; export it to CSV (code,word) and pass it
with --wordlist to get the words as label text; without it the text is the code.
"""

import csv
import re
from pathlib import Path

from echo_bench.data.audio import audio_info
from echo_bench.data.ingest_common import IngestError, IngestReport, Progress
from echo_bench.data.manifest import UtteranceRecord
from echo_bench.data.speakers import SpeakerMeta

FILE_NAME = re.compile(
    r"^(?P<speaker>C?[FM]\d{2})_(?P<block>B[1-3])_(?P<word>[A-Z]+\d*)_(?P<mic>M\d)\.wav$",
    re.IGNORECASE,
)
SYNC_TONE_MIC = "M1"


def word_category(code: str) -> str:
    for prefix, group in (
        ("CW", "common_words"),
        ("UW", "uncommon_words"),
        ("D", "digits"),
        ("L", "radio_alphabet"),
        ("C", "computer_commands"),
    ):
        if code.startswith(prefix):
            return f"proxy:{group}"
    return "proxy:other"


def load_wordlist(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["code"].strip().upper(): row["word"].strip() for row in csv.DictReader(handle)}


def _mic_number(mic: str) -> int:
    return int(mic[1:])


def scan_uaspeech(
    root: Path,
    speakers: dict[str, SpeakerMeta],
    *,
    primary_channel: str = "M5",
    all_channels: bool = False,
    wordlist: dict[str, str] | None = None,
    progress: Progress | None = None,
) -> tuple[list[UtteranceRecord], IngestReport]:
    root = root.resolve()
    report = IngestReport()
    groups: dict[str, dict[str, Path]] = {}
    meta: dict[str, tuple[str, str, str, str]] = {}
    for wav in sorted(root.rglob("*")):
        if not wav.is_file() or wav.suffix.lower() != ".wav":
            continue
        match = FILE_NAME.match(wav.name)
        if match is None:
            report.counts["skipped: file name not in UASpeech format"] += 1
            continue
        speaker, block, word, mic = (match[k].upper() for k in ("speaker", "block", "word", "mic"))
        if mic == SYNC_TONE_MIC:
            report.counts["skipped: mic 1 (synchronization tone)"] += 1
            continue
        key = f"{block}_{word}" if word.startswith("UW") else word
        group = f"uaspeech:{speaker}:{block}:{word}"
        mics = groups.setdefault(group, {})
        if mic in mics:
            raise IngestError(
                f"two copies of the same recording: {mics[mic]} and {wav}. Point --root at "
                "one version of the corpus (for example the original or the noise-reduced folder)."
            )
        mics[mic] = wav
        meta[group] = (speaker, block, word, key)
    if not groups:
        raise IngestError(f"no UASpeech files under {root}")
    unknown = sorted({speaker for speaker, _, _, _ in meta.values()} - set(speakers))
    if unknown:
        raise IngestError(f"speakers missing from the speaker table: {', '.join(unknown)}")

    records: list[UtteranceRecord] = []
    target = _mic_number(primary_channel)
    for group, mics in sorted(groups.items()):
        speaker, block, word, key = meta[group]
        # The primary is the first READABLE mic, nearest to the requested one first.
        infos = {}
        for mic in sorted(mics, key=lambda m: (abs(_mic_number(m) - target), _mic_number(m))):
            info = audio_info(mics[mic])
            if info is None:
                report.counts["skipped: unreadable or empty audio"] += 1
                continue
            infos[mic] = info
            if not all_channels:
                break
        if not infos:
            continue
        primary = next(iter(infos))
        if primary != primary_channel:
            report.counts[f"primary fell back from {primary_channel}"] += 1
        text = key
        if wordlist:
            text = wordlist.get(key) or wordlist.get(word) or key
            if text == key:
                report.counts["word list has no entry (text = code)"] += 1
        for mic, info in sorted(infos.items()):
            wav = mics[mic]
            records.append(
                UtteranceRecord(
                    speaker=speaker,
                    label_key=key,
                    label_text=text,
                    category=word_category(word),
                    recording_group_id=group,
                    session=block,
                    channel=mic,
                    is_primary_channel=mic == primary,
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
