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
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import inspect

from echo_bench.db.migrate import current_revision, head_revision, upgrade_to_head
from echo_bench.db.models import utc_now
from echo_bench.db.repository import (
    get_or_create_hardware_profile,
    get_or_create_software_env,
    set_meta_if_missing,
)
from echo_bench.db.session import default_db_path, make_engine, make_session
from echo_bench.env import rpi
from echo_bench.env.fingerprint import collect_hardware, collect_software, is_wsl

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
