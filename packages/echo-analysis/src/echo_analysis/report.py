"""`echo-bench report ...`: tables and figures from the results database (desktop only).

This Typer app is added to `echo-bench` under the name `report` through the
'echo_bench.plugins' entry point (see pyproject.toml). The Pi doesn't install
echo-analysis, so it simply has no report command. pandas and matplotlib are
imported only when a report runs, so other echo-bench commands stay fast.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from echo_bench.db.session import default_db_path, make_engine

app = typer.Typer(
    help="Tables and figures from the results database (desktop only).", no_args_is_help=True
)
console = Console()

CAMPAIGNS = text(
    """
    SELECT c.id, c.name, c.status, c.started_at, c.notes, h.hostname, h.platform, h.cpu_model
    FROM campaign c JOIN hardware_profile h ON h.id = c.hardware_profile_id
    WHERE c.kind = 'perf' ORDER BY c.started_at DESC
    """
)
METRICS = text(
    """
    SELECT r.id AS run_id, r.status, r.invalid_reason, r.ort_intra_threads AS threads,
           r.ort_allow_spinning AS spinning, a.name || ':' || a.variant AS model,
           a.variant, substr(a.sha256, 1, 8) AS sha8, a.size_bytes,
           m.name AS metric, m.scope, m.value, m.ci_low, m.ci_high, m.n
    FROM benchmark_run r
    JOIN model_artifact a ON a.id = r.model_artifact_id
    JOIN run_metric m ON m.run_id = r.id
    WHERE r.campaign_id = :campaign_id
    """
)


@app.command("perf")
def perf_report(
    latest: Annotated[
        bool, typer.Option("--latest", help="Newest perf campaign (the default).")
    ] = False,
    campaign: Annotated[
        str | None, typer.Option(help="Campaign id or its first characters.")
    ] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite results file.")] = None,
    out_dir: Annotated[Path, typer.Option(help="Folder for the figure.")] = Path("reports"),
) -> None:
    """Latency table and a latency-vs-input-length figure for one perf campaign."""
    import matplotlib

    matplotlib.use("Agg")  # draw straight to a file; no window needed
    import matplotlib.pyplot as plt
    import pandas as pd

    path = db or default_db_path()
    if not path.exists():
        console.print(f"[red]✗[/red] {path} does not exist. Run a perf campaign first.")
        raise typer.Exit(1)
    engine = make_engine(path)
    try:
        with engine.connect() as connection:
            campaigns = pd.read_sql(CAMPAIGNS, connection)
            if campaign:
                campaigns = campaigns[campaigns["id"].str.startswith(campaign)]
            if campaigns.empty:
                console.print("[red]✗[/red] No matching perf campaign in this database.")
                raise typer.Exit(1)
            chosen = campaigns.iloc[0]
            data = pd.read_sql(METRICS, connection, params={"campaign_id": chosen["id"]})
    finally:
        engine.dispose()

    console.print(
        f"[bold]Perf campaign {chosen['id'][:8]}[/bold] '{chosen['name']}' on {chosen['hostname']} "
        f"({chosen['platform']}, {chosen['cpu_model']}), {chosen['started_at'][:16].replace('T', ' ')} UTC,"
        f" status {chosen['status']}"
    )
    if chosen["notes"]:
        console.print(f"[yellow]![/yellow] {chosen['notes']}")
    flagged = data[data["status"] != "completed"].drop_duplicates("run_id")
    for row in flagged.itertuples():
        console.print(
            f"[yellow]![/yellow] Excluded {row.model} ({row.status}): {row.invalid_reason or 'no reason'}"
        )
    data = data[data["status"] == "completed"]
    if data.empty:
        console.print("[red]✗[/red] No valid runs to report.")
        raise typer.Exit(1)

    several_settings = data[["threads", "spinning"]].drop_duplicates().shape[0] > 1

    def label(model: str, threads: int, spinning: int) -> str:
        if not several_settings:
            return model
        return f"{model}, {threads} threads{'' if spinning else ', no spin'}"

    latency = data[data["scope"].str.startswith("encoder@")].copy()
    latency["length_s"] = (
        latency["scope"].str.removeprefix("encoder@").str.removesuffix("s").astype(float)
    )
    keys = ["model", "sha8", "threads", "spinning", "length_s"]

    table = Table(
        title="Encoder latency in ms: p50 (median) and p95 (95% of runs faster), [95% CI]",
        title_justify="left",
    )
    for column in ("model", "length", "p50 [CI]", "p95 [CI]", "p99", "n"):
        table.add_column(column)
    for (model, sha8, threads, spinning, length), group in latency.groupby(keys, sort=True):
        stats = {row.metric: row for row in group.itertuples()}
        p50, p95, p99 = stats["latency_p50_ms"], stats["latency_p95_ms"], stats["latency_p99_ms"]
        table.add_row(
            f"{label(model, threads, spinning)} ({sha8})",
            f"{length:g} s",
            f"{p50.value:.1f} [{p50.ci_low:.1f}-{p50.ci_high:.1f}]",
            f"{p95.value:.1f} [{p95.ci_low:.1f}-{p95.ci_high:.1f}]",
            f"{p99.value:.1f}",
            f"{int(p50.n)}",
        )
    console.print(table)

    overall = data[data["metric"].isin(["session_load_ms", "cold_inference_ms"])]
    table = Table(title="Start-up cost per model", title_justify="left")
    for column in ("model", "file size", "session load", "first inference"):
        table.add_column(column)
    for (model, sha8, threads, spinning, size), group in overall.groupby(
        ["model", "sha8", "threads", "spinning", "size_bytes"], sort=True
    ):
        values = dict(zip(group["metric"], group["value"], strict=True))
        table.add_row(
            f"{label(model, threads, spinning)} ({sha8})",
            f"{size / 1e6:.2f} MB",
            f"{values.get('session_load_ms', float('nan')):.0f} ms",
            f"{values.get('cold_inference_ms', float('nan')):.0f} ms",
        )
    console.print(table)
    memory = data[data["metric"].isin(["peak_rss_mb", "peak_sys_used_mb", "max_soc_temp_c"])]
    summary = memory.groupby("metric")["value"].max()
    parts = []
    if "peak_rss_mb" in summary:
        parts.append(
            f"peak process memory {summary['peak_rss_mb']:.0f} MiB (all models loaded together)"
        )
    if "peak_sys_used_mb" in summary:
        parts.append(f"peak system memory in use {summary['peak_sys_used_mb']:.0f} MiB")
    if "max_soc_temp_c" in summary:
        parts.append(f"hottest SoC reading {summary['max_soc_temp_c']:.1f} C")
    if parts:
        console.print("; ".join(parts))

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for (model, threads, spinning), group in latency.groupby(
        ["model", "threads", "spinning"], sort=True
    ):
        curves = group.pivot_table(index="length_s", columns="metric", values="value").sort_index()
        name = label(model.split(":", 1)[1], threads, spinning)
        (line,) = axis.plot(curves.index, curves["latency_p50_ms"], marker="o", label=f"{name} p50")
        axis.plot(
            curves.index,
            curves["latency_p95_ms"],
            marker="o",
            linestyle="--",
            color=line.get_color(),
            label=f"{name} p95",
        )
    axis.set_xlabel("input length (s)")
    axis.set_ylabel("encoder latency (ms)")
    axis.set_title(f"{chosen['name']} on {chosen['hostname']} ({chosen['platform']})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"perf_{chosen['id'][:8]}_latency_vs_length.png"
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    console.print(f"[green]✓[/green] Figure saved: {output}")
