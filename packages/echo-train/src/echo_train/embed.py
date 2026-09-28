"""Utterance embeddings from a trained speech encoder (early accuracy check, before M5).

Two kinds of model:

  ESPnet ASR models: a Hugging Face repo ("hf:pyf98/librispeech_100_e_branchformer") or a
      local folder with exp/<run>/config.yaml and a .pth file next to it. The model's own
      frontend (log-mel + normalization) runs inside `encode`.
  Moonshine (original, not streaming): "moonshine:UsefulSensors/moonshine-tiny" or
      "moonshine:<local folder>" (Hugging Face transformers format). Only the encoder is
      used; its convolutional frontend reads the raw waveform, so there is no log-mel step.

Either way each checkpoint gets the exact features it was trained on, and each utterance
becomes the mean of its encoder frames (the same masked mean pooling ECHO deploys).

Why the weights are checked here: ESPnet loads checkpoints with strict=False, so tensors
that don't match the installed ESPnet version would be skipped silently, leaving part of
the encoder random. load_model() refuses unless every frontend, normalization and encoder
tensor is present with the right shape. load_moonshine() applies the same rule to the
Moonshine encoder (transformers also fills missing tensors with random values).

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

ENCODER_PREFIXES = ("frontend.", "normalize.", "preencoder.", "encoder.")
MOONSHINE = "moonshine:"
MOONSHINE_ENCODER_PREFIX = "model.encoder."
MOONSHINE_FILES = ["*.json", "*.safetensors"]  # config, feature extractor, weights; no tokenizer


class EmbedError(Exception):
    pass


class TooShortUttError(EmbedError):
    """A clip too short for Moonshine's frontend. Same name as ESPnet's error, so
    embed_manifest skips it the same way."""


@dataclass(frozen=True)
class Checkpoint:
    root: Path  # config paths (stats file, token list) are relative to this folder
    config: Path
    weights: Path


@dataclass(frozen=True)
class Embedder:
    """One loaded model, ready to turn audio at `sample_rate` into one vector."""

    embed: Callable[[np.ndarray], np.ndarray]
    info: dict[str, Any]
    sample_rate: int


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


def legacy_renames(state: dict[str, Any], own: dict[str, Any]) -> dict[str, str]:
    """Tensor names that changed between ESPnet versions without the layer changing.

    Older ESPnet wrapped some layers in nn.Sequential; for example the conv2d input stage
    stored its output layer together with the position encoding, as
    `encoder.embed.out.0.weight`, and current ESPnet stores the same layer as
    `encoder.embed.out.weight`. An unknown checkpoint tensor is renamed only if dropping
    ONE numeric segment from its name gives exactly one missing model tensor with the
    same shape. Anything else is left for the fit check to reject."""
    missing = {k for k in own if k.startswith(ENCODER_PREFIXES) and k not in state}
    renames: dict[str, str] = {}
    for old in state:
        if old in own or not old.startswith(ENCODER_PREFIXES):
            continue
        parts = old.split(".")
        candidates = {
            ".".join(parts[:i] + parts[i + 1 :]) for i, part in enumerate(parts) if part.isdigit()
        } & missing
        if len(candidates) != 1:
            continue
        new = candidates.pop()
        if tuple(state[old].shape) == tuple(own[new].shape) and new not in renames.values():
            renames[old] = new
    return renames


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
        renames = legacy_renames(state, own)
        state = {renames.get(k, k): v for k, v in state.items()}
        needed = [k for k in own if k.startswith(ENCODER_PREFIXES)]
        missing = [k for k in needed if k not in state]
        reshaped = [
            k for k in needed if k in state and tuple(state[k].shape) != tuple(own[k].shape)
        ]
        unknown = [k for k in state if k.startswith(ENCODER_PREFIXES) and k not in own]
        if missing or reshaped or unknown:

            def listed(keys: list[str]) -> str:
                if not keys:
                    return "none"
                return ", ".join(keys[:4]) + (" ..." if len(keys) > 4 else "")

            raise EmbedError(
                f"{checkpoint.weights.name} does not fit this ESPnet version's model, so its "
                f"embeddings would be partly random. Missing: {listed(missing)}. "
                f"Other shapes: {listed(reshaped)}. Unknown: {listed(unknown)}."
            )
        info["renamed_tensors"] = renames
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


def embed_audio(model: Any, audio: np.ndarray, device: str) -> np.ndarray:
    speech = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.inference_mode():
        frames, lengths = model.encode(speech, torch.tensor([audio.shape[0]], device=device))
    valid = int(lengths[0])
    return frames[0, :valid].mean(dim=0).float().cpu().numpy()


def fetch_moonshine(source: str) -> Path:
    """A local folder as is; otherwise a Hugging Face repo id, downloaded (and cached)."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(source, allow_patterns=MOONSHINE_FILES))


def load_moonshine(
    folder: Path, *, random_weights: bool = False
) -> tuple[Any, Any, dict[str, Any], int]:
    """(encoder in eval mode on CPU, feature extractor, metadata, sample rate)."""
    try:
        from transformers import AutoConfig, AutoFeatureExtractor, MoonshineForConditionalGeneration
        from transformers.utils import logging as hf_logging
    except ImportError as error:
        raise EmbedError(
            "Moonshine needs transformers: run "
            '`uv add "transformers>=5.17,<6"` in packages/echo-train'
        ) from error
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    config = AutoConfig.from_pretrained(folder)
    if config.model_type != "moonshine":
        raise EmbedError(
            f"{folder}: model type {config.model_type!r}; only the original Moonshine "
            "('moonshine') is supported so far, not the streaming version"
        )
    weights = folder / "model.safetensors"
    if random_weights:
        torch.manual_seed(0)  # the same untrained floor on every run
        model = MoonshineForConditionalGeneration(config)
    else:
        model, loading = MoonshineForConditionalGeneration.from_pretrained(
            folder, output_loading_info=True
        )
        mismatched = [
            m[0] if isinstance(m, (tuple, list)) else m for m in loading["mismatched_keys"]
        ]
        problems = {
            "missing": [
                k for k in loading["missing_keys"] if k.startswith(MOONSHINE_ENCODER_PREFIX)
            ],
            "with other shapes": [
                k for k in mismatched if str(k).startswith(MOONSHINE_ENCODER_PREFIX)
            ],
            "unknown": [
                k for k in loading["unexpected_keys"] if k.startswith(MOONSHINE_ENCODER_PREFIX)
            ],
        }
        if any(problems.values()):
            example = next(k for keys in problems.values() for k in keys)
            counts = ", ".join(f"{len(keys)} {what}" for what, keys in problems.items())
            raise EmbedError(
                f"{folder} does not fit this transformers version's Moonshine: {counts} "
                f"encoder tensors (for example {example}). Its embeddings would be partly random."
            )
    encoder = model.get_encoder().eval()
    extractor = AutoFeatureExtractor.from_pretrained(folder)
    fs = int(extractor.sampling_rate)
    loaded = sum(1 for k in model.state_dict() if k.startswith(MOONSHINE_ENCODER_PREFIX))
    info: dict[str, Any] = {
        "config": "config.json",
        "encoder": "moonshine",
        "encoder_conf": {
            "output_size": config.hidden_size,
            "num_blocks": config.encoder_num_hidden_layers,
            "attention_heads": config.encoder_num_attention_heads,
            "linear_units": config.intermediate_size,
        },
        "frontend_conf": {"type": "raw waveform, convolutional (inside the encoder)", "fs": fs},
        "encoder_params": sum(p.numel() for p in encoder.parameters()),
        "weights_state": "random_init" if random_weights else "trained",
        "weights_file": None if random_weights else weights.name,
        "weights_sha256": None
        if random_weights or not weights.is_file()
        else hashlib.sha256(weights.read_bytes()).hexdigest(),
        "encoder_tensors_loaded": 0 if random_weights else loaded,
    }
    return encoder, extractor, info, fs


def embed_moonshine(encoder: Any, extractor: Any, audio: np.ndarray, device: str) -> np.ndarray:
    inputs = extractor(audio, sampling_rate=extractor.sampling_rate, return_tensors="pt")
    values = inputs[extractor.model_input_names[0]].to(device)
    with torch.inference_mode():
        frames = encoder(values).last_hidden_state  # (1, frames, hidden)
    if frames.shape[1] == 0:
        raise TooShortUttError(
            f"a {audio.shape[0]}-sample clip is too short for Moonshine's frontend"
        )
    # One unpadded utterance: every frame is valid, so this is the same masked mean as ESPnet's.
    return frames[0].mean(dim=0).float().cpu().numpy()


def open_embedder(model_source: str, *, random_weights: bool, device: str) -> Embedder:
    """Load `model_source` (ESPnet or moonshine:...) and return a ready Embedder."""
    if model_source.startswith(MOONSHINE):
        folder = fetch_moonshine(model_source.removeprefix(MOONSHINE))
        encoder, extractor, info, fs = load_moonshine(folder, random_weights=random_weights)
        encoder.to(device)
        return Embedder(lambda audio: embed_moonshine(encoder, extractor, audio, device), info, fs)
    checkpoint = find_checkpoint(fetch(model_source))
    model, info, fs = load_model(checkpoint, random_weights=random_weights)
    model.to(device)
    return Embedder(lambda audio: embed_audio(model, audio, device), info, fs)


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

    embedder = open_embedder(model_source, random_weights=random_weights, device=device)
    info, sample_rate = embedder.info, embedder.sample_rate
    if "encoder_params" in info:
        progress(f"Encoder parameters: {info['encoder_params'] / 1e6:.2f} M")
    loaded = (
        "" if random_weights else f", all {info['encoder_tensors_loaded']} encoder tensors loaded"
    )
    renamed = info.get("renamed_tensors") or {}
    note = f", {len(renamed)} renamed from an older ESPnet layout" if renamed else ""
    progress(f"Model: {model_source} ({info['encoder']}, {info['weights_state']}{loaded}{note})")
    started = time.perf_counter()
    vectors, kept, skipped = [], [], []
    for index, record in enumerate(records, start=1):
        try:
            audio = read_audio(root / record["audio_path"], sample_rate)
            vectors.append(embedder.embed(audio))
            kept.append(record)
        except Exception as error:
            # Recordings shorter than the input stage's minimum (~0.07 s, not real speech):
            # ESPnet raises TooShortUttError (its module moved between versions, so match
            # by name), and so does embed_moonshine. Skip them; any other error still stops
            # the run.
            if type(error).__name__ != "TooShortUttError":
                raise
            skipped.append(record["audio_path"])
        if index % 500 == 0 or index == len(records):
            progress(f"  {index}/{len(records)} utterances embedded")
    if not vectors:
        raise EmbedError("no recording was long enough for the encoder")
    records = kept
    embeddings = np.stack(vectors).astype(np.float32)
    metadata = {
        "model_source": model_source,
        **info,
        "sample_rate": sample_rate,
        "pooling": "masked_mean",
        "manifest": manifest.name,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "skipped_too_short": skipped,
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
