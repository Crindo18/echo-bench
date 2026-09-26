"""Utterance embeddings from a trained ESPnet ASR encoder (early accuracy check, before M5).

Any ESPnet ASR model works: a Hugging Face repo ("hf:pyf98/librispeech_100_e_branchformer")
or a local folder with exp/<run>/config.yaml and a .pth file next to it. The model's own
frontend (log-mel + normalization) runs inside `encode`, so each checkpoint gets the exact
features it was trained on. Each utterance becomes the mean of its encoder frames (the
same masked mean pooling ECHO deploys).

Why the weights are checked here: ESPnet loads checkpoints with strict=False, so tensors
that don't match the installed ESPnet version would be skipped silently, leaving part of
the encoder random. load_model() refuses unless every frontend, normalization and encoder
tensor is present with the right shape.

Output: one .npz per model and dataset (blueprint ADR-4), with the embeddings, the
manifest fields needed for few-shot episodes, and a JSON metadata string.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from espnet2.asr.encoder.abs_encoder import AbsEncoder  # noqa: F401  (import kept lazy)

ENCODER_PREFIXES = ("frontend.", "normalize.", "preencoder.", "encoder.")


class EmbedError(Exception):
    pass


@dataclass(frozen=True)
class Checkpoint:
    root: Path  # config paths (stats file, token list) are relative to this folder
    config: Path
    weights: Path


def fetch(model: str) -> Path:
    """'hf:<user>/<repo>' downloads (and caches) a Hugging Face repo; anything else is a folder."""
    if model.startswith("hf:"):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model[3:]))
    path = Path(model).expanduser()
    if not path.is_dir():
        raise EmbedError(f"{model}: not a folder (use hf:<user>/<repo> for Hugging Face)")
    return path


def find_checkpoint(root: Path) -> Checkpoint:
    configs = sorted(p for p in root.glob("exp/*/config.yaml") if any(p.parent.glob("*.pth")))
    if not configs:
        raise EmbedError(f"no exp/<run>/config.yaml with a .pth file next to it under {root}")
    config = next((p for p in configs if p.parent.name.startswith("asr_train")), configs[0])
    weights = sorted(
        config.parent.glob("*.pth"), key=lambda p: ("ave" not in p.name, -p.stat().st_size)
    )
    return Checkpoint(root=root, config=config, weights=weights[0])


@contextlib.contextmanager
def _inside(folder: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)


def _sample_rate(value: Any) -> int:
    text = str(value or "16k").lower()
    return int(float(text[:-1]) * 1000) if text.endswith("k") else int(text)


def load_model(
    checkpoint: Checkpoint, *, random_weights: bool = False
) -> tuple[Any, dict[str, Any], int]:
    """(model in eval mode on CPU, metadata, sample rate)."""
    from espnet2.tasks.asr import ASRTask

    quiet = io.StringIO()
    with (
        _inside(checkpoint.root),
        contextlib.redirect_stdout(quiet),
        contextlib.redirect_stderr(quiet),
    ):
        model, args = ASRTask.build_model_from_file(checkpoint.config, None, "cpu")
    info: dict[str, Any] = {
        "config": checkpoint.config.relative_to(checkpoint.root).as_posix(),
        "encoder": getattr(args, "encoder", None),
        "encoder_conf": getattr(args, "encoder_conf", None),
        "frontend_conf": getattr(args, "frontend_conf", None),
        "weights_state": "random_init",
        "weights_file": None,
        "weights_sha256": None,
        "encoder_tensors_loaded": 0,
    }
    if not random_weights:
        state = torch.load(checkpoint.weights, map_location="cpu", weights_only=True)
        own = model.state_dict()

        # ESPnet 202209 -> 202610 renamed the input projection: encoder.embed.out.{w,b}
        # became encoder.embed.out.0.{w,b}. Same tensors, same shapes; map the old names
        # to the new ones so the strict check below still guards every encoder weight.
        state = {
            k.replace("encoder.embed.out.0.", "encoder.embed.out."): v for k, v in state.items()
        }

        needed = [k for k in own if k.startswith(ENCODER_PREFIXES)]
        missing = [k for k in needed if k not in state]
        reshaped = [
            k for k in needed if k in state and tuple(state[k].shape) != tuple(own[k].shape)
        ]
        unknown = [k for k in state if k.startswith(ENCODER_PREFIXES) and k not in own]
        if missing or reshaped or unknown:
            example = (missing or reshaped or unknown)[0]
            raise EmbedError(
                f"{checkpoint.weights.name} does not fit this ESPnet version's model: "
                f"{len(missing)} missing, {len(reshaped)} with other shapes, {len(unknown)} unknown "
                f"encoder tensors (for example {example}). Its embeddings would be partly random."
            )
        model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
        info.update(
            weights_state="trained",
            weights_file=checkpoint.weights.relative_to(checkpoint.root).as_posix(),
            weights_sha256=hashlib.sha256(checkpoint.weights.read_bytes()).hexdigest(),
            encoder_tensors_loaded=len(needed),
        )
    fs = _sample_rate((getattr(args, "frontend_conf", None) or {}).get("fs"))
    return model.eval(), info, fs


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != sample_rate:
        import torchaudio.functional as F

        mono = F.resample(torch.from_numpy(mono), rate, sample_rate).numpy()
    return np.ascontiguousarray(mono, dtype=np.float32)


def embed_audio(model: Any, audio: np.ndarray, device: str) -> np.ndarray | None:
    speech = torch.from_numpy(audio).unsqueeze(0).to(device)
    try:
        with torch.inference_mode():
            frames, lengths = model.encode(speech, torch.tensor([audio.shape[0]], device=device))
    except Exception as error:  # noqa: BLE001
        if type(error).__name__ == "TooShortUttError" or "Padding size should be less" in str(
            error
        ):
            return None  # near-silent / truncated clip; skip it
        raise
    valid = int(lengths[0])
    return frames[0, :valid].mean(dim=0).float().cpu().numpy()


def embed_manifest(
    manifest: Path,
    root: Path,
    out: Path,
    *,
    model_source: str,
    random_weights: bool = False,
    device: str = "cpu",
    limit: int | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Embed every primary-channel utterance of a manifest; write the .npz; return a summary."""
    with manifest.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records = [r for r in records if r.get("is_primary_channel", True)][:limit]
    if not records:
        raise EmbedError(f"{manifest} has no primary-channel records")
    missing_files = [r["audio_path"] for r in records if not (root / r["audio_path"]).is_file()]
    if missing_files:
        raise EmbedError(
            f"{len(missing_files)} audio files not found under {root} (first: {missing_files[0]})"
        )

    checkpoint = find_checkpoint(fetch(model_source))
    model, info, sample_rate = load_model(checkpoint, random_weights=random_weights)
    model.to(device)
    progress(
        f"Model: {model_source} ({info['encoder']}, {info['weights_state']}"
        + (
            f", all {info['encoder_tensors_loaded']} encoder tensors loaded)"
            if not random_weights
            else ")"
        )
    )
    started = time.perf_counter()
    vectors = []
    kept_records = []
    skipped = 0
    MIN_SAMPLES = 640  # ~40 ms at 16 kHz; shorter clips can't survive the STFT/subsampling
    for index, record in enumerate(records, start=1):
        audio = read_audio(root / record["audio_path"], sample_rate)
        if audio.shape[0] < MIN_SAMPLES:
            skipped += 1
            continue
        vector = embed_audio(model, audio, device)
        if vector is None:
            skipped += 1
            continue
        vectors.append(vector)
        kept_records.append(record)
        if index % 500 == 0 or index == len(records):
            progress(f"  {index}/{len(records)} utterances embedded ({skipped} skipped: too short)")
    records = kept_records  # only keep the ones we actually embedded
    embeddings = np.stack(vectors).astype(np.float32)
    metadata = {
        "model_source": model_source,
        **info,
        "sample_rate": sample_rate,
        "pooling": "masked_mean",
        "manifest": manifest.name,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        **{
            field: np.array([str(r.get(field) or "") for r in records])
            for field in (
                "audio_path",
                "speaker",
                "label_key",
                "recording_group_id",
                "session",
                "category",
            )
        },
        metadata=np.array(json.dumps(metadata)),
    )
    return {
        "count": len(records),
        "dim": int(embeddings.shape[1]),
        "seconds": time.perf_counter() - started,
        **metadata,
    }
