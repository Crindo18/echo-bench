"""The `echo-bench` command line (blueprint ADR-5 and section 11.2).

M0 commands:
    echo-bench doctor       check this machine and print its fingerprints
    echo-bench db init      create the results database, or bring it up to date
    echo-bench db upgrade   apply new schema migrations to an existing database

Later milestones add model, perf, data, fidelity, embed, fewshot, kws and more.
"""

import importlib.util
import json
import platform
import socket
import sqlite3
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import Engine, func, inspect, select

from echo_bench.db.migrate import current_revision, head_revision, upgrade_to_head
from echo_bench.db.models import BenchmarkRun, CvPlan, GateResult, ModelArtifact, RunMetric, utc_now
from echo_bench.db.repository import (
    get_or_create_hardware_profile,
    get_or_create_software_env,
    set_meta_if_missing,
)
from echo_bench.db.session import default_db_path, make_engine, make_session
from echo_bench.env import rpi
from echo_bench.env.fingerprint import collect_hardware, collect_software, is_wsl
from echo_bench.gates import DEFAULT_GATES_FILE

app = typer.Typer(
    help="ECHO-Bench: benchmarking harness for the ECHO encoder.", no_args_is_help=True
)
db_app = typer.Typer(help="Create and upgrade the results database.", no_args_is_help=True)
app.add_typer(db_app, name="db")
console = Console()

DbOption = Annotated[
    Path | None,
    typer.Option("--db", help="SQLite results file.  [default: results/bench_<hostname>.db]"),
]

REQUIRED_PYTHON = (3, 12)
MIN_SQLITE = (3, 38, 0)  # json_valid() is built into SQLite from 3.38
EXPECTED_ORT = "1.30.0"  # the same runtime on both machines (cross-platform parity, gate G2d)
DESKTOP_ONLY = ("torch", "espnet2", "librosa", "pandas", "matplotlib")

OK, FAIL, WARN, INFO = "[green]✓[/green]", "[red]✗[/red]", "[yellow]![/yellow]", "[blue]i[/blue]"


@db_app.command("init")
def db_init(db: DbOption = None) -> None:
    """Create the results database, or bring an existing one up to date."""
    path = db or default_db_path()
    upgrade_to_head(path)
    engine = make_engine(path)
    try:
        with make_session(engine) as session:
            set_meta_if_missing(session, "schema_family", "echo-bench")
            set_meta_if_missing(session, "created_at", utc_now())
            set_meta_if_missing(session, "created_on_host", socket.gethostname())
            session.commit()
        tables = [name for name in inspect(engine).get_table_names() if name != "alembic_version"]
    finally:
        engine.dispose()
    console.print(f"{OK} Database ready: {path}")
    console.print(f"  schema revision {current_revision(path)}, {len(tables)} tables:")
    console.print(f"  {', '.join(tables)}")


@db_app.command("upgrade")
def db_upgrade(db: DbOption = None) -> None:
    """Apply new schema migrations to an existing database."""
    path = db or default_db_path()
    if not path.exists():
        console.print(f"{FAIL} {path} does not exist yet. Run `echo-bench db init` first.")
        raise typer.Exit(1)
    before = current_revision(path)
    upgrade_to_head(path)
    console.print(f"{OK} {path}: schema revision {before} -> {current_revision(path)}")


@app.command()
def doctor(
    db: DbOption = None,
    save: Annotated[
        bool, typer.Option("--save", help="Store the fingerprints in the database.")
    ] = False,
    cooling: Annotated[
        str | None, typer.Option(help="Manual tag: active_cooler, passive or none.")
    ] = None,
    power_source: Annotated[
        str | None, typer.Option(help="Manual tag: official_27w_psu or powerbank_pd_board.")
    ] = None,
    storage: Annotated[
        str | None, typer.Option(help="Manual tag, for example '64GB SanDisk microSD'.")
    ] = None,
) -> None:
    """Check this machine and print its hardware and software fingerprints."""
    hw = collect_hardware(storage=storage, cooling=cooling, power_source=power_source)
    sw = collect_software()

    hw_table = Table(title="Hardware", show_header=False, title_justify="left")
    hw_table.add_column("field", style="bold")
    hw_table.add_column("value")
    for key in (
        "hostname",
        "platform",
        "device_model",
        "cpu_model",
        "cpu_arch",
        "cpu_cores",
        "ram_total_mb",
        "os_name",
        "kernel",
        "storage",
        "cooling",
        "power_source",
    ):
        hw_table.add_row(key, "-" if hw[key] is None else str(hw[key]))
    hw_table.add_row("fingerprint", hw["fingerprint"][:16])
    console.print(hw_table)

    providers = json.loads(sw["ort_providers_json"])
    commit = sw["git_commit"]
    code_version = "not a git repository"
    if commit is not None:
        code_version = commit[:8] + (" (uncommitted changes)" if sw["git_dirty"] else "")
    sw_table = Table(title="Software", show_header=False, title_justify="left")
    sw_table.add_column("field", style="bold")
    sw_table.add_column("value")
    sw_table.add_row("python", sw["python_version"])
    sw_table.add_row("onnxruntime", sw["onnxruntime_version"])
    sw_table.add_row("ort providers", ", ".join(providers))
    sw_table.add_row("numpy", sw["numpy_version"])
    sw_table.add_row("packages", f"{len(json.loads(sw['packages_json']))} installed")
    sw_table.add_row("git commit", code_version)
    sw_table.add_row("fingerprint", sw["fingerprint"][:16])
    console.print(sw_table)

    checks = Table(title="Checks", show_header=False, title_justify="left")
    checks.add_column("mark")
    checks.add_column("check")
    checks.add_column("detail")
    failures = 0

    def check(ok: bool, label: str, detail: str = "", hard: bool = True) -> None:
        nonlocal failures
        if not ok and hard:
            failures += 1
        checks.add_row(OK if ok else (FAIL if hard else WARN), label, detail)

    check(sys.version_info[:2] == REQUIRED_PYTHON, "Python 3.12", platform.python_version())
    check(sqlite3.sqlite_version_info >= MIN_SQLITE, "SQLite 3.38 or newer", sqlite3.sqlite_version)
    check("CPUExecutionProvider" in providers, "ONNX Runtime CPU provider", ", ".join(providers))
    check(
        sw["onnxruntime_version"] == EXPECTED_ORT,
        f"ONNX Runtime {EXPECTED_ORT} (same on desktop and Pi)",
        sw["onnxruntime_version"],
        hard=False,
    )
    check(
        commit is not None, "Inside a git repository", "results record the code version", hard=False
    )
    if commit is not None:
        check(
            not sw["git_dirty"],
            "No uncommitted changes",
            "official runs need a clean tree (section 6.7)",
            hard=False,
        )

    if hw["platform"] == "rpi5":
        leaked = [name for name in DESKTOP_ONLY if importlib.util.find_spec(name) is not None]
        check(
            not leaked,
            "No desktop-only packages on the Pi",
            ", ".join(leaked) if leaked else "none installed",
            hard=False,
        )
        throttled = rpi.read_throttled()
        if throttled is None:
            check(False, "vcgencmd works", "needed for throttle flags (section 6.2)", hard=False)
        else:
            flags = ", ".join(rpi.decode_throttled(throttled)) or "no flags"
            check(
                not rpi.has_any_bit(throttled, rpi.UNDER_VOLTAGE_BITS),
                "No under-voltage since boot",
                f"{hex(throttled)}: {flags}",
                hard=False,
            )
            check(
                not rpi.has_any_bit(throttled, rpi.THROTTLING_BITS),
                "No throttling since boot",
                f"{hex(throttled)}: {flags}",
                hard=False,
            )
        temp, freq, governor = (
            rpi.read_soc_temp_c(),
            rpi.read_cpu_freq_mhz(),
            rpi.read_cpu_governor(),
        )
        checks.add_row(
            INFO, "SoC temperature, CPU frequency, governor", f"{temp} C, {freq} MHz, {governor}"
        )
    elif is_wsl():
        checks.add_row(
            INFO, "Running under WSL2", "Pi temperature and throttle monitors don't apply here"
        )

    path = db or default_db_path()
    revision, head = current_revision(path), head_revision()
    state = "not created yet" if revision is None else f"revision {revision}"
    check(
        revision == head,
        "Results database up to date",
        f"{path}: {state} (latest {head})",
        hard=save,
    )
    console.print(checks)

    if save and revision == head:
        engine = make_engine(path)
        try:
            with make_session(engine) as session:
                hw_row = get_or_create_hardware_profile(session, hw)
                sw_row = get_or_create_software_env(session, sw)
                session.commit()
                console.print(
                    f"{OK} Saved to {path}: hardware_profile {hw_row.id[:8]}, "
                    f"software_env {sw_row.id[:8]}"
                )
        finally:
            engine.dispose()
    elif save:
        console.print(f"{FAIL} Not saved: run `echo-bench db init` first.")

    if failures:
        console.print(f"{FAIL} {failures} check(s) failed.")
        raise typer.Exit(1)
    console.print(f"{OK} doctor finished with no blocking problems.")


# ----------------------------------------------------------------------- M1
# Heavier modules (onnx, onnxruntime, numpy) are imported inside the commands
# that need them, so `echo-bench --help` and `doctor` stay fast.

model_app = typer.Typer(help="Register and inspect model artifacts.", no_args_is_help=True)
app.add_typer(model_app, name="model")

GatesOption = Annotated[Path, typer.Option("--gates", help="Gate thresholds file.")]


def _open_db(db: Path | None) -> Engine:
    """The results database; it must exist and be at the latest schema revision."""
    path = db or default_db_path()
    revision, head = current_revision(path), head_revision()
    if revision != head:
        if revision is None:
            console.print(f"{FAIL} {path} does not exist yet. Run `echo-bench db init` first.")
        else:
            console.print(
                f"{FAIL} {path} is at schema revision {revision}, not {head}. "
                "Run `echo-bench db upgrade` (your results are kept)."
            )
        raise typer.Exit(1)
    return make_engine(path)


def _gate_text(row: GateResult) -> str:
    mark = OK if row.passed else FAIL
    return f"{mark} {row.gate.split('_')[0]}: {row.observed:.4g} {row.comparator} {row.threshold:g}"


@model_app.command("register")
def model_register(
    paths: Annotated[list[Path], typer.Argument(help="Artifact folders, or folders to search.")],
    db: DbOption = None,
    gates_file: GatesOption = DEFAULT_GATES_FILE,
) -> None:
    """Check artifacts against their model cards, record them, and evaluate G1 and G2a."""
    from echo_bench.gates import load_gates
    from echo_bench.registry import RegistryError, register_paths

    engine = _open_db(db)
    try:
        with make_session(engine) as session:
            try:
                results = register_paths(session, paths, load_gates(gates_file))
            except RegistryError as error:
                console.print(f"{FAIL} {error}")
                raise typer.Exit(1) from error
            session.commit()
            table = Table(title="Model registry", title_justify="left")
            for column in ("model", "sha256", "size", "status", "gates"):
                table.add_column(column)
            for result in results:
                artifact = result.artifact
                gates = session.scalars(
                    select(GateResult).where(GateResult.model_artifact_id == artifact.id)
                ).all()
                table.add_row(
                    f"{artifact.name}:{artifact.variant}",
                    artifact.sha256[:8],
                    f"{artifact.size_bytes / 1e6:.2f} MB",
                    "new" if result.created else "already registered",
                    "  ".join(_gate_text(g) for g in gates) or "-",
                )
            console.print(table)
    finally:
        engine.dispose()


@model_app.command("list")
def model_list(db: DbOption = None) -> None:
    """Show every registered model, newest first."""
    engine = _open_db(db)
    try:
        with make_session(engine) as session:
            rows = session.scalars(
                select(ModelArtifact).order_by(ModelArtifact.created_at.desc())
            ).all()
            table = Table(title=f"{len(rows)} registered model(s)", title_justify="left")
            for column in ("model", "sha256", "size", "parameters", "weights", "registered"):
                table.add_column(column)
            for row in rows:
                params = "-" if row.param_count is None else f"{row.param_count / 1e6:.2f} M"
                table.add_row(
                    f"{row.name}:{row.variant}",
                    row.sha256[:8],
                    f"{row.size_bytes / 1e6:.2f} MB",
                    params,
                    row.weights_state,
                    row.created_at[:16].replace("T", " "),
                )
            console.print(table)
    finally:
        engine.dispose()


@model_app.command("inspect")
def model_inspect(
    ref: Annotated[str, typer.Argument(help="'name:variant' or a SHA-256 prefix.")],
    db: DbOption = None,
) -> None:
    """Size, parameters, operators and quantization coverage of one model (section 6.3)."""
    from echo_bench.registry import (
        RegistryError,
        card_of,
        inspect_onnx,
        resolve,
        verify_artifact_file,
    )

    engine = _open_db(db)
    try:
        with make_session(engine) as session:
            try:
                artifact = resolve(session, ref)
                verify_artifact_file(artifact)
            except RegistryError as error:
                console.print(f"{FAIL} {error}")
                raise typer.Exit(1) from error
            card = card_of(artifact)
            gates = session.scalars(
                select(GateResult).where(GateResult.model_artifact_id == artifact.id)
            ).all()
            path = Path(artifact.file_path)
            size_mb = artifact.size_bytes / 1e6
    finally:
        engine.dispose()

    console.print(f"[bold]{card.ref}[/bold]  sha256 {artifact.sha256[:12]}  ({card.weights_state})")
    console.print(f"  file        {path}  {size_mb:.2f} MB")
    if card.param_count is not None:
        console.print(f"  parameters  {card.param_count / 1e6:.2f} M (from the model card)")
    for row in gates:
        console.print(f"  gate        {_gate_text(row)}")
    if path.suffix != ".onnx":
        return
    info = inspect_onnx(path)
    console.print(f"  opset       {info['opset']}")
    console.print(f"  inputs      {info['inputs']}")
    console.print(f"  outputs     {info['outputs']}")
    for family, (quantized, total) in info["coverage"].items():
        if total:
            share = 100 * quantized / total
            console.print(
                f"  {family:<11} {quantized}/{total} weight-bearing nodes quantized ({share:.0f}%)"
            )
    storage = ", ".join(
        f"{dtype} {size / 1e6:.2f} MB" for dtype, size in info["bytes_by_dtype"].items()
    )
    console.print(f"  stored      {storage}")
    table = Table(title="Largest stored tensors", title_justify="left")
    for column in ("tensor", "dtype", "shape", "MB"):
        table.add_column(column)
    for name, dtype, shape, nbytes in info["largest"]:
        table.add_row(name[-60:], dtype, str(shape), f"{nbytes / 1e6:.2f}")
    console.print(table)
    top = ", ".join(f"{op} {count}" for op, count in info["op_histogram"][:12])
    console.print(f"  operators   {top}")

    # Compute per utterance: the same on every machine, so it predicts relative Pi speed.
    from echo_bench.registry import count_macs

    counts = {seconds: count_macs(path, card, seconds) for seconds in (1, 3, 5, 8)}
    line = " | ".join(f"{s} s: {total / 1e9:.2f}" for s, (total, _) in counts.items())
    console.print(f"  compute     billion multiply-accumulates per utterance  {line}")
    total5, groups5 = counts[5]
    share = ", ".join(f"{name} {macs / total5:.0%}" for name, macs in list(groups5.items())[:5])
    console.print(f"  at 5 s      {share}")


@app.command()
def perf(
    config: Annotated[
        Path, typer.Option("--config", help="e.g. configs/benchmarks/perf_smoke.yaml")
    ],
    db: DbOption = None,
    name: Annotated[
        str | None, typer.Option(help="Campaign name.  [default: the config's file name]")
    ] = None,
) -> None:
    """Time each model at each input length, with memory and temperature (section 6.1)."""
    from pydantic import ValidationError

    from echo_bench.config.loader import load_perf_config
    from echo_bench.registry import RegistryError
    from echo_bench.runners.perf import run_perf

    try:
        cfg, cfg_json, cfg_sha = load_perf_config(config)
    except (OSError, ValidationError) as error:
        console.print(f"{FAIL} {config}: {error}")
        raise typer.Exit(1) from error
    engine = _open_db(db)
    try:
        with make_session(engine) as session:
            try:
                campaign = run_perf(
                    session,
                    cfg,
                    name=name or config.stem,
                    config_json=cfg_json,
                    config_sha256=cfg_sha,
                    progress=console.print,
                )
            except RegistryError as error:
                console.print(f"{FAIL} {error}")
                raise typer.Exit(1) from error
            runs = session.scalars(
                select(BenchmarkRun).where(BenchmarkRun.campaign_id == campaign.id)
            ).all()
            table = Table(
                title="Encoder latency in ms (p50 = median, p95 = 95% of runs faster)",
                title_justify="left",
            )
            for column in ("model", "threads", "length", "p50", "p95", "run status"):
                table.add_column(column)
            for run in runs:
                artifact = session.get(ModelArtifact, run.model_artifact_id)
                metrics = session.scalars(select(RunMetric).where(RunMetric.run_id == run.id)).all()
                values = {(m.name, m.scope): m.value for m in metrics}
                for seconds in cfg.input_durations_s:
                    scope = f"encoder@{seconds:g}s"
                    table.add_row(
                        f"{artifact.name}:{artifact.variant}" if artifact else "?",
                        str(run.ort_intra_threads)
                        + ("" if run.ort_allow_spinning else " (no spin)"),
                        f"{seconds:g} s",
                        f"{values.get(('latency_p50_ms', scope), float('nan')):.1f}",
                        f"{values.get(('latency_p95_ms', scope), float('nan')):.1f}",
                        run.status if not run.invalid_reason else f"invalid: {run.invalid_reason}",
                    )
            console.print(table)
            if campaign.notes:
                console.print(f"{WARN} {campaign.notes}")
            console.print(
                f"{OK} Campaign {campaign.id[:8]} saved. Next: uv run echo-bench report perf --latest"
            )
    finally:
        engine.dispose()


# ----------------------------------------------------------------------- M3

data_app = typer.Typer(
    help="Ingest corpora, check them for leakage, report coverage (M3).", no_args_is_help=True
)
ingest_app = typer.Typer(
    help="Scan a corpus folder into a manifest and load it.", no_args_is_help=True
)
app.add_typer(data_app, name="data")
data_app.add_typer(ingest_app, name="ingest")
MANIFEST_DIR = Path("data/manifests")
VersionOption = Annotated[
    str, typer.Option(help="Dataset version; change it when the files change.")
]
AllChannelsOption = Annotated[
    bool, typer.Option("--all-channels", help="Also record the non-primary microphones.")
]


def _ingest(
    db: Path | None,
    name: str,
    version: str,
    root: Path,
    speakers_csv: Path,
    scan: Any,
    primary: str,
) -> None:
    """Shared steps: scan -> manifest (temporary file first) -> database -> summary."""
    from echo_bench.data.ingest_common import IngestError
    from echo_bench.data.loader import load_dataset
    from echo_bench.data.manifest import write_manifest
    from echo_bench.data.speakers import load_speakers
    from echo_bench.db.models import Dataset

    engine = _open_db(db)
    try:
        speakers = load_speakers(speakers_csv)
        console.print(f"Scanning {root} ...")
        records, report = scan(speakers, console.print)
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        final = MANIFEST_DIR / f"{name}-{version}.jsonl"
        temporary = MANIFEST_DIR / f".{name}-{version}.jsonl.tmp"
        manifest_sha = write_manifest(records, temporary)
        with make_session(engine) as session:
            existing = session.scalar(
                select(Dataset).where(Dataset.name == name, Dataset.version == version)
            )
            if existing is not None and existing.manifest_sha256 != manifest_sha:
                temporary.unlink()
                raise IngestError(
                    f"{name} {version} is already loaded from different files. "
                    "Datasets are immutable: use a new --version."
                )
            temporary.replace(final)
            _, created = load_dataset(
                session,
                name=name,
                version=version,
                access_level="licensed_corpus",
                manifest_path=final,
                speakers=speakers,
                primary_channel=primary,
                root_hint=root.name,
            )
            session.commit()
    except (IngestError, OSError, ValueError) as error:
        console.print(f"{FAIL} {error}")
        raise typer.Exit(1) from error
    finally:
        engine.dispose()

    status = "loaded" if created else "already loaded, unchanged"
    table = Table(title=f"{name} {version}: {status}", title_justify="left")
    table.add_column("what")
    table.add_column("count", justify="right")
    for key, value in sorted(report.counts.items(), key=lambda kv: (kv[0] != "used", kv[0])):
        table.add_row("files used" if key == "used" else key, str(value))
    groups: dict[str, int] = {}
    for code in sorted({r.speaker for r in records}):
        group = speakers[code].severity_tier or speakers[code].cohort
        groups[group] = groups.get(group, 0) + 1
    table.add_row("speakers", ", ".join(f"{group} {n}" for group, n in sorted(groups.items())))
    table.add_row("classes (labels)", str(len({r.label_key for r in records})))
    table.add_row("manifest", f"{final} (sha256 {manifest_sha[:12]})")
    console.print(table)
    console.print(
        "Next: uv run echo-bench data validate, then "
        "uv run echo-bench cv plan --config configs/cv/sgkf5-severity-s42.yaml"
    )


@ingest_app.command("torgo")
def ingest_torgo(
    root: Annotated[
        Path, typer.Option(help="The TORGO folder (with F/, M/, ... or the speaker folders).")
    ],
    speakers: Annotated[Path, typer.Option(help="Speaker table.")] = Path(
        "configs/datasets/torgo_speakers.csv"
    ),
    version: VersionOption = "v1",
    primary_channel: Annotated[str, typer.Option(help="headMic or arrayMic.")] = "headMic",
    all_channels: AllChannelsOption = False,
    db: DbOption = None,
) -> None:
    """TORGO: prompts + head/array microphones, one recording group per prompt and session."""
    from echo_bench.data.ingest_torgo import scan_torgo

    def scan(meta: Any, progress: Any) -> Any:
        return scan_torgo(
            root,
            meta,
            primary_channel=primary_channel,
            all_channels=all_channels,
            progress=progress,
        )

    _ingest(db, "torgo", version, root, speakers, scan, primary_channel)


@ingest_app.command("uaspeech")
def ingest_uaspeech(
    root: Annotated[
        Path,
        typer.Option(help="The UASpeech audio folder (one version: original or noise-reduced)."),
    ],
    speakers: Annotated[Path, typer.Option(help="Speaker table.")] = Path(
        "configs/datasets/uaspeech_speakers.csv"
    ),
    wordlist: Annotated[
        Path | None, typer.Option(help="CSV with columns code,word (the official list).")
    ] = None,
    version: VersionOption = "v1",
    primary_channel: Annotated[str, typer.Option(help="Microphone M2-M8.")] = "M5",
    all_channels: AllChannelsOption = False,
    db: DbOption = None,
) -> None:
    """UASpeech: <speaker>_<block>_<word>_<mic>.wav, one recording group per word and block."""
    from echo_bench.data.ingest_uaspeech import load_wordlist, scan_uaspeech

    words = load_wordlist(wordlist) if wordlist else None

    def scan(meta: Any, progress: Any) -> Any:
        return scan_uaspeech(
            root,
            meta,
            primary_channel=primary_channel,
            all_channels=all_channels,
            wordlist=words,
            progress=progress,
        )

    _ingest(db, "uaspeech", version, root, speakers, scan, primary_channel)


@data_app.command("list")
def data_list(db: DbOption = None) -> None:
    """Every loaded dataset version with its speakers, classes and recordings."""
    from sqlalchemy import text as sql

    engine = _open_db(db)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sql(
                    """
                    SELECT d.name, d.version, d.primary_channel, d.created_at,
                           (SELECT COUNT(*) FROM speaker s WHERE s.dataset_id = d.id),
                           (SELECT COUNT(*) FROM label l WHERE l.dataset_id = d.id),
                           (SELECT COUNT(*) FROM utterance u
                              WHERE u.dataset_id = d.id AND u.is_primary_channel = 1),
                           (SELECT COALESCE(SUM(duration_ms), 0) FROM utterance u
                              WHERE u.dataset_id = d.id AND u.is_primary_channel = 1)
                    FROM dataset d ORDER BY d.name, d.created_at
                    """
                )
            ).all()
    finally:
        engine.dispose()
    table = Table(title=f"{len(rows)} dataset version(s)", title_justify="left")
    for column in (
        "dataset",
        "version",
        "primary mic",
        "speakers",
        "classes",
        "recordings",
        "hours",
        "loaded",
    ):
        table.add_column(column)
    for name, version, mic, created, n_spk, n_lab, n_utt, ms in rows:
        table.add_row(
            name,
            version,
            mic or "-",
            str(n_spk),
            str(n_lab),
            str(n_utt),
            f"{ms / 3.6e6:.2f}",
            created[:16].replace("T", " "),
        )
    console.print(table)


@data_app.command("validate")
def data_validate(db: DbOption = None) -> None:
    """Check leakage rules L1-L2 for every CV plan, plus recording-group consistency."""
    from echo_bench.data.leakage import validate

    engine = _open_db(db)
    try:
        with make_session(engine) as session:
            issues = validate(session)
            n_plans = session.scalar(select(func.count()).select_from(CvPlan)) or 0
    finally:
        engine.dispose()
    errors = [i for i in issues if i.level == "error"]
    if issues:
        table = Table(title="Data validation", title_justify="left")
        for column in ("rule", "level", "problem"):
            table.add_column(column)
        for issue in issues:
            table.add_row(
                issue.rule,
                f"{FAIL if issue.level == 'error' else WARN} {issue.level}",
                issue.message,
            )
        console.print(table)
    if errors:
        console.print(f"{FAIL} {len(errors)} error(s): fix them before training or evaluating.")
        raise typer.Exit(1)
    console.print(
        f"{OK} No leakage found ({n_plans} CV plan(s) checked, {len(issues)} warning(s))."
    )


@data_app.command("coverage")
def data_coverage(
    dataset: Annotated[str | None, typer.Option(help="Only this dataset, e.g. torgo.")] = None,
    db: DbOption = None,
) -> None:
    """Classes per speaker with enough recordings for K = 3, 5 and 10 (K support + 1 query)."""
    from echo_bench.data.coverage import K_LEVELS, coverage

    engine = _open_db(db)
    try:
        with make_session(engine) as session:
            rows = coverage(session, dataset)
    finally:
        engine.dispose()
    table = Table(
        title="Feasible K per speaker: classes with at least K + 1 recordings", title_justify="left"
    )
    for column in (
        "dataset",
        "speaker",
        "group",
        "classes",
        "median recordings",
        *(f"K={k}" for k in K_LEVELS),
    ):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.dataset,
            row.speaker,
            row.severity_tier or row.cohort,
            str(row.classes),
            f"{row.median_recordings:g}",
            *(str(row.feasible[k]) for k in K_LEVELS),
        )
    console.print(table)


# ----------------------------------------------------------- early accuracy check

fewshot_app = typer.Typer(
    help="Few-shot accuracy with Prototypical Networks.", no_args_is_help=True
)
app.add_typer(fewshot_app, name="fewshot")


def _speaker_groups(path: Path | None, dataset_hint: str) -> dict[str, str]:
    from echo_bench.data.speakers import load_speakers

    if path is None:
        guess = Path("configs/datasets") / f"{dataset_hint}_speakers.csv"
        path = guess if guess.is_file() else None
    if path is None:
        return {}
    return {code: meta.severity_tier or meta.cohort for code, meta in load_speakers(path).items()}


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


@fewshot_app.command("quick")
def fewshot_quick(
    embeddings: Annotated[
        list[Path],
        typer.Option("--embeddings", help=".npz from `echo-train embed`; repeat to compare."),
    ],
    speakers: Annotated[
        Path | None, typer.Option(help="Speaker table [default: from the manifest name].")
    ] = None,
    k: Annotated[
        list[int] | None, typer.Option("--k", help="Enrollment sizes to test [default: 1 2 3].")
    ] = None,
    ways: Annotated[
        int, typer.Option(help="Classes per episode (ECHO after H-KWS: under 10).")
    ] = 10,
    episodes: Annotated[int, typer.Option(help="Random episodes per speaker and condition.")] = 20,
    seed: int = 42,
    normalize: Annotated[bool, typer.Option(help="Scale embeddings to length 1 first.")] = False,
) -> None:
    """Early accuracy check: speaker-independent (SI) and personal K-shot prototypes per speaker."""
    from echo_bench.fewshot.embedding_cache import load_embeddings
    from echo_bench.fewshot.quick import evaluate, summarize

    ks = k or [1, 2, 3]
    conditions = ["SI", *(f"K={n}" for n in ks)]
    overall: list[tuple[str, dict[str, float | None]]] = []
    sets = [load_embeddings(path) for path in embeddings]
    if len(sets) > 1:
        common = set.intersection(*(set(es.group) for es in sets))
        if any(len(es.group) != len(common) for es in sets):
            console.print(
                f"{WARN} Comparing on the {len(common)} recordings that every file contains "
                "(some encoders skipped recordings the others kept)."
            )
            sets = [es.restricted(common) for es in sets]
    for es in sets:
        hint = str(es.metadata.get("manifest", "")).split("-")[0]
        groups = _speaker_groups(speakers, hint)
        results = evaluate(
            es, groups, ks=ks, ways=ways, episodes=episodes, seed=seed, normalize=normalize
        )
        rows = summarize(results, conditions)
        table = Table(
            title=f"{es.name}: {len(results)} speakers, {len(set(es.label))} classes, {ways}-way episodes",
            title_justify="left",
        )
        for column in ("group", "speakers", *conditions):
            table.add_column(column, justify="left" if column == "group" else "right")
        for group, n, values in rows:
            table.add_row(group, str(n), *(_pct(values[c]) for c in conditions))
        console.print(table)
        overall.append((es.name, rows[-1][2] if rows and rows[-1][0] == "all dysarthric" else {}))
    if len(overall) > 1:
        table = Table(title="All dysarthric speakers, side by side", title_justify="left")
        for column in ("encoder", *conditions):
            table.add_column(column, justify="left" if column == "encoder" else "right")
        for name, values in overall:
            table.add_row(name, *(_pct(values.get(c)) for c in conditions))
        console.print(table)
    console.print(
        f"Chance is about {100 / ways:.0f}% ({ways}-way). SI = prototypes from other speakers only; "
        "K=k = k of the speaker's own recordings per class. A quick check, not the thesis protocol (M5)."
    )


def _load_plugins() -> None:
    """Desktop-only command groups, such as `report` from echo-analysis, add themselves
    through the 'echo_bench.plugins' entry point, so echo_bench never imports them."""
    for entry in entry_points(group="echo_bench.plugins"):
        try:
            app.add_typer(entry.load(), name=entry.name)
        except Exception as error:  # a broken plugin must not break the harness
            console.print(f"{WARN} Could not load the {entry.name!r} command: {error}")


_load_plugins()
