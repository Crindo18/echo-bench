"""The `echo-train` command (blueprint section 11.2). Desktop only; run inside packages/echo-train.

    uv run echo-train build-random --config ../../configs/models/ebf12m.yaml \
        --out ../../artifacts/models/ebf-12m-rand
    uv run echo-train quantize --fp32 ../../artifacts/models/ebf-12m-rand/<sha8> \
        --method static_qdq --calibration Entropy

Trained checkpoints (`export`, `parity` and `card` for real weights) arrive in M4.
"""

import json
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from rich.console import Console
from rich.table import Table

from echo_core.model_card import load_card
from echo_train.artifacts import base_card_fields, utc_now, write_artifact
from echo_train.build_model import build_model, count_parameters
from echo_train.config import (
    CalibrationConfig,
    EncoderConfig,
    ExportConfig,
    ModelConfig,
    QuantizationConfig,
    load_model_config,
)
from echo_train.export_onnx import check_parity, export_fp32
from echo_train.quantize import VARIANT_OF, int8_sanity, preprocess, quantize

app = typer.Typer(
    help="Build, export and quantize the ECHO encoder (desktop only).", no_args_is_help=True
)
console = Console()
G1_LIMIT_MB = 16.0  # gate G1; configs/gates.yaml is the official copy used by echo-bench
OK, FAIL = "[green]PASS[/green]", "[red]FAIL[/red]"


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def _print_parameters(counts: dict[str, int]) -> None:
    table = Table(title="Parameters", title_justify="left")
    table.add_column("module")
    table.add_column("millions", justify="right")
    blocks = [v for k, v in counts.items() if k.startswith("block")]
    table.add_row("input subsampling", f"{counts['input subsampling'] / 1e6:.2f}")
    if blocks and len(set(blocks)) == 1:
        table.add_row(f"blocks ({len(blocks)} x {blocks[0] / 1e6:.2f})", f"{sum(blocks) / 1e6:.2f}")
    else:
        for key, value in counts.items():
            if key.startswith("block"):
                table.add_row(key, f"{value / 1e6:.2f}")
    table.add_row("everything else", f"{counts['everything else'] / 1e6:.2f}")
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{counts['TOTAL'] / 1e6:.2f}[/bold]")
    console.print(table)


@app.command("build-random")
def build_random(
    config: Annotated[
        Path, typer.Option(help="Model recipe, e.g. ../../configs/models/ebf12m.yaml")
    ],
    out: Annotated[
        Path, typer.Option(help="Folder for this model, e.g. ../../artifacts/models/ebf-12m-rand")
    ],
) -> None:
    """Build the encoder with random weights, export it, check parity, and make INT8 versions."""
    cfg = load_model_config(config)
    name = f"{cfg.name}-rand"
    model = build_model(cfg)
    counts = count_parameters(model)
    _print_parameters(counts)

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp_dir:
        tmp = Path(tmp_dir)
        fp32_tmp = tmp / "model_fp32.onnx"
        export_fp32(model, cfg, fp32_tmp)
        parity = check_parity(model, cfg, fp32_tmp)
        worst = max(parity.values())

        table = Table(title="Export parity, PyTorch vs ONNX (gate G2a)", title_justify="left")
        table.add_column("input length")
        table.add_column("max |difference|", justify="right")
        table.add_column("result")
        for seconds, diff in parity.items():
            ok = diff <= cfg.export.parity_tolerance
            traced = "  (traced)" if seconds == cfg.export.trace_length_s else ""
            table.add_row(f"{seconds:g} s{traced}", f"{diff:.2e}", OK if ok else FAIL)
        console.print(table)
        if worst > cfg.export.parity_tolerance:
            console.print(f"{FAIL} Export parity failed; no artifacts written (section 6.3).")
            raise typer.Exit(1)

        export_info = {
            "torch": torch.__version__.split("+")[0],
            "exporter": "torchscript",
            "opset": cfg.export.opset,
            "dynamic_axes": ["B", "T"],
            "trace_length_s": cfg.export.trace_length_s,
            "parity_lengths_s": list(parity),
            "parity_max_abs": worst,
            "parity_tolerance": cfg.export.parity_tolerance,
        }
        fields = base_card_fields(cfg, name, counts["TOTAL"], export_info)

        prep = tmp / "model_prep.onnx"
        preprocess(fp32_tmp, prep)
        int8_versions = []
        for method in cfg.quantization.methods:
            int8_tmp = tmp / f"model_{method}.onnx"
            info = quantize(prep, int8_tmp, method, cfg)
            int8_versions.append((method, int8_tmp, info, int8_sanity(fp32_tmp, int8_tmp, cfg)))

        fp32_folder, fp32_new = write_artifact(out, fp32_tmp, fields | {"variant": "onnx_fp32"})
        fp32_sha = load_card(fp32_folder).artifact_sha256
        rows = [("onnx_fp32", fp32_folder, fp32_new, None)]
        for method, int8_tmp, info, cosine in int8_versions:
            card = fields | {
                "variant": VARIANT_OF[method],
                "parent_sha256": fp32_sha,
                "quantization": info,
            }
            folder, created = write_artifact(out, int8_tmp, card)
            rows.append((VARIANT_OF[method], folder, created, cosine))

    table = Table(title=f"Artifacts for {name}", title_justify="left")
    for column in ("variant", "folder", "size", "G1", "cosine", "status"):
        table.add_column(column)
    for variant, folder, created, cosine in rows:
        size = _mb(folder / "model.onnx")
        gate = "-" if cosine is None else (OK if size <= G1_LIMIT_MB else FAIL)
        similarity = "-" if cosine is None else f"{cosine:.4f}"
        table.add_row(
            variant,
            folder.name,
            f"{size:.2f} MB",
            gate,
            similarity,
            "new" if created else "unchanged",
        )
    console.print(table)
    console.print(
        "G1: INT8 file <= 16 MB. cosine: lowest INT8-vs-FP32 similarity over all test lengths\n"
        "(with random weights this only shows the INT8 file runs; real fidelity is checked in M4)."
    )
    # Which files this build produced, so scripts/m1_slice.sh benchmarks exactly these
    # (a 'name:variant' reference means "most recently registered", which can be older).
    last_build = {
        "built_at": utc_now(),
        "recipe": config.as_posix(),
        "artifacts": {variant: load_card(folder).artifact_sha256 for variant, folder, _, _ in rows},
    }
    (out / "last_build.json").write_text(json.dumps(last_build, indent=2) + "\n")
    console.print(f"Next: uv run echo-bench model register {out.as_posix().removeprefix('../../')}")


@app.command("quantize")
def quantize_command(
    fp32: Annotated[Path, typer.Option(help="Folder of an onnx_fp32 artifact")],
    method: Annotated[Literal["dynamic", "static_qdq"], typer.Option(help="INT8 method")],
    calibration: Annotated[
        Literal["MinMax", "Entropy", "Percentile"], typer.Option(help="Static only")
    ] = "MinMax",
    n_utts: Annotated[int, typer.Option(help="Calibration utterances (static only)")] = 16,
    per_channel: Annotated[bool, typer.Option(help="Per-channel weight scales")] = True,
) -> None:
    """Make one more INT8 version of an existing FP32 artifact, e.g. with other calibration."""
    card = load_card(fp32)
    if card.variant != "onnx_fp32" or card.export is None:
        console.print(f"{FAIL} {fp32} is a {card.variant} artifact; pass an onnx_fp32 folder.")
        raise typer.Exit(1)
    encoder_fields = {k: v for k, v in card.encoder.items() if k in EncoderConfig.model_fields}
    cfg = ModelConfig(
        name=card.name,
        seed=card.seed or 0,
        input_size=int(card.encoder["input_size"]),
        encoder=EncoderConfig(**encoder_fields),
        export=ExportConfig(parity_lengths_s=card.export.parity_lengths_s),
        quantization=QuantizationConfig(
            methods=[method],
            per_channel=per_channel,
            calibration=CalibrationConfig(n_utts=n_utts, method=calibration),
        ),
    )
    with tempfile.TemporaryDirectory(dir=fp32.parent) as tmp_dir:
        tmp = Path(tmp_dir)
        prep = tmp / "model_prep.onnx"
        preprocess(fp32 / "model.onnx", prep)
        int8_tmp = tmp / "model_int8.onnx"
        info = quantize(prep, int8_tmp, method, cfg)
        cosine = int8_sanity(fp32 / "model.onnx", int8_tmp, cfg)
        fields = card.model_dump(mode="json", exclude={"created_at", "files_sha256"})
        fields |= {"variant": VARIANT_OF[method], "parent_sha256": card.artifact_sha256}
        fields["quantization"] = info
        folder, created = write_artifact(fp32.parent, int8_tmp, fields)
    size = _mb(folder / "model.onnx")
    verdict = OK if size <= G1_LIMIT_MB else FAIL
    status = "new" if created else "unchanged (identical file already existed)"
    console.print(
        f"{VARIANT_OF[method]} -> {folder}  {size:.2f} MB  G1 {verdict}  cosine {cosine:.4f}  ({status})"
    )
