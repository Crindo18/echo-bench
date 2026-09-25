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
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import Engine, inspect, select

from echo_bench.db.migrate import current_revision, head_revision, upgrade_to_head
from echo_bench.db.models import BenchmarkRun, GateResult, ModelArtifact, RunMetric, utc_now
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
        state = (
            "does not exist yet" if revision is None else f"is at revision {revision}, not {head}"
        )
        console.print(f"{FAIL} {path} {state}. Run `echo-bench db init` first.")
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


def _load_plugins() -> None:
    """Desktop-only command groups, such as `report` from echo-analysis, add themselves
    through the 'echo_bench.plugins' entry point, so echo_bench never imports them."""
    for entry in entry_points(group="echo_bench.plugins"):
        try:
            app.add_typer(entry.load(), name=entry.name)
        except Exception as error:  # a broken plugin must not break the harness
            console.print(f"{WARN} Could not load the {entry.name!r} command: {error}")


_load_plugins()
