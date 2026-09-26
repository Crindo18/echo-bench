"""`echo-bench cv ...`: speaker-independent cross-validation plans (desktop only, blueprint M3).

Registered as a plugin like `report` (scikit-learn stays off the Pi). The plan is
made once on the desktop and stored in the results database, and the model card of
every trained model records which plan and fold it came from (leakage rule L4).

How folds are made (thesis §1.7.4.4: stratified, speaker-independent k-fold):
- StratifiedGroupKFold over primary-channel recordings, groups = speakers, so a
  speaker's recordings never straddle folds; strata = dataset + severity tier
  (controls form their own stratum), so every fold gets a mix of severities.
- Fold i: its group is the test set, the next group is the validation set
  ("validation: next_fold"), everything else is training.
"""

import json
import warnings
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from echo_bench.db.migrate import current_revision, head_revision
from echo_bench.db.models import CvPlan, Dataset, FoldAssignment, Speaker, Utterance
from echo_bench.db.session import default_db_path, make_engine, make_session

app = typer.Typer(
    help="Cross-validation plans (desktop only; uses scikit-learn).", no_args_is_help=True
)
console = Console()
OK, FAIL, WARN = "[green]✓[/green]", "[red]✗[/red]", "[yellow]![/yellow]"


class CvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    strategy: Literal["stratified_group_kfold", "leave_one_speaker_out"] = "stratified_group_kfold"
    k: int = Field(5, ge=2)  # ignored for leave_one_speaker_out (k = number of speakers)
    stratify_by: Literal["severity_tier"] = "severity_tier"
    seed: int = 42
    datasets: list[str] = Field(min_length=1)  # "torgo" (newest version) or "torgo:v1"
    validation: Literal["next_fold", "none"] = "next_fold"

    @model_validator(mode="after")
    def _training_left(self) -> "CvConfig":
        if (
            self.strategy == "stratified_group_kfold"
            and self.validation == "next_fold"
            and self.k < 3
        ):
            raise ValueError(
                "validation: next_fold needs k >= 3 (with k = 2 no speakers are left for training)"
            )
        return self


class PlanError(Exception):
    pass


def _datasets(db: Session, refs: list[str]) -> list[Dataset]:
    found = []
    for ref in refs:
        name, _, version = ref.partition(":")
        query = select(Dataset).where(Dataset.name == name)
        if version:
            query = query.where(Dataset.version == version)
        dataset = db.scalar(query.order_by(Dataset.created_at.desc()))
        if dataset is None:
            raise PlanError(
                f"dataset {ref!r} is not loaded; run `echo-bench data ingest ...` first"
            )
        found.append(dataset)
    return found


def make_plan(db: Session, config: CvConfig) -> tuple[CvPlan, list[str]]:
    """Create and store the plan. Returns the plan and any warnings from scikit-learn."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    if db.scalar(select(CvPlan).where(CvPlan.name == config.name)) is not None:
        raise PlanError(
            f"a plan named {config.name!r} already exists; plans are immutable, pick a new name"
        )
    datasets = _datasets(db, config.datasets)
    names = {d.id: d.name for d in datasets}
    rows = db.execute(
        select(
            Utterance.speaker_id,
            Speaker.dataset_id,
            Speaker.severity_tier,
            Speaker.cohort,
            Speaker.code,
        )
        .join(Speaker, Speaker.id == Utterance.speaker_id)
        .where(Utterance.is_primary_channel == 1, Speaker.dataset_id.in_(list(names)))
    ).all()
    if not rows:
        raise PlanError("the selected datasets have no recordings")
    groups = np.array([r[0] for r in rows])
    strata = np.array([f"{names[r[1]]}:{r[2] or r[3]}" for r in rows])
    speakers = sorted({(names[r[1]], r[4], r[0]) for r in rows})
    caught: list[str] = []
    test_fold: dict[str, int] = {}
    if config.strategy == "stratified_group_kfold":
        k = config.k
        if len(speakers) < k:
            raise PlanError(f"{len(speakers)} speakers cannot fill {k} folds")
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.seed)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for fold, (_, test_idx) in enumerate(
                splitter.split(np.zeros(len(rows)), strata, groups)
            ):
                for speaker_id in set(groups[test_idx]):
                    test_fold[str(speaker_id)] = fold
        caught = sorted({str(w.message) for w in recorded})
    else:
        k = len(speakers)
        for fold, (_, _, speaker_id) in enumerate(speakers):
            test_fold[speaker_id] = fold
    plan = CvPlan(
        name=config.name,
        strategy=config.strategy,
        k=k,
        stratify_by=config.stratify_by,
        seed=config.seed,
        dataset_ids_json=json.dumps(sorted(names)),
    )
    db.add(plan)
    db.flush()
    for fold in range(k):
        for speaker_id, own_fold in test_fold.items():
            if own_fold == fold:
                role = "test"
            elif config.validation == "next_fold" and own_fold == (fold + 1) % k:
                role = "val"
            else:
                role = "train"
            db.add(
                FoldAssignment(
                    cv_plan_id=plan.id, fold_index=fold, speaker_id=speaker_id, role=role
                )
            )
    return plan, caught


def _db(db: Path | None) -> Path:
    path = db or default_db_path()
    if current_revision(path) != head_revision():
        console.print(f"{FAIL} {path} is missing or outdated. Run `echo-bench db upgrade` first.")
        raise typer.Exit(1)
    return path


@app.command("plan")
def plan_command(
    config: Annotated[
        Path, typer.Option("--config", help="e.g. configs/cv/sgkf5-severity-s42.yaml")
    ],
    db: Annotated[Path | None, typer.Option("--db", help="SQLite results file.")] = None,
) -> None:
    """Assign every speaker to folds (train/val/test) and store the plan."""
    cfg = CvConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    engine = make_engine(_db(db))
    try:
        with make_session(engine) as session:
            try:
                plan, caught = make_plan(session, cfg)
            except PlanError as error:
                console.print(f"{FAIL} {error}")
                raise typer.Exit(1) from error
            session.commit()
            name = plan.name
    finally:
        engine.dispose()
    for message in caught:
        console.print(f"{WARN} scikit-learn: {message}")
    console.print(
        f"{OK} CV plan {name!r} stored. Next: uv run echo-bench data validate, then echo-bench cv summary"
    )


@app.command("summary")
def summary_command(
    plan: Annotated[str | None, typer.Option(help="Plan name [default: the newest].")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="SQLite results file.")] = None,
    out_dir: Annotated[Path, typer.Option(help="Folder for the CSV and Markdown tables.")] = Path(
        "reports"
    ),
) -> None:
    """Fold summary for Chapters 3 and 5: test speakers and recordings per fold and severity."""
    import pandas as pd
    from sqlalchemy import text

    engine = make_engine(_db(db))
    try:
        with engine.connect() as connection:
            plans = pd.read_sql(
                text("SELECT id, name, k, strategy FROM cv_plan ORDER BY created_at DESC"),
                connection,
            )
            if plan:
                plans = plans[plans["name"] == plan]
            if plans.empty:
                console.print(f"{FAIL} No CV plan found.")
                raise typer.Exit(1)
            chosen = plans.iloc[0]
            data = pd.read_sql(
                text(
                    """
                    SELECT f.fold_index AS fold, f.role, d.name AS dataset,
                           COALESCE(s.severity_tier, s.cohort) AS grp, s.code,
                           (SELECT COUNT(*) FROM utterance u
                              WHERE u.speaker_id = s.id AND u.is_primary_channel = 1) AS recordings
                    FROM fold_assignment f JOIN speaker s ON s.id = f.speaker_id
                    JOIN dataset d ON d.id = s.dataset_id WHERE f.cv_plan_id = :p
                    """
                ),
                connection,
                params={"p": chosen["id"]},
            )
    finally:
        engine.dispose()

    data["stratum"] = data["dataset"] + ":" + data["grp"]
    long = (
        data.groupby(["fold", "role", "stratum"])
        .agg(speakers=("code", "nunique"), recordings=("recordings", "sum"))
        .reset_index()
    )
    test = long[long["role"] == "test"]
    folds = sorted(data["fold"].unique())
    table = Table(
        title=f"CV plan {chosen['name']}: test set per fold (speakers / recordings)",
        title_justify="left",
    )
    table.add_column("dataset:group")
    for fold in folds:
        table.add_column(f"fold {fold}", justify="right")
    markdown = [
        "| dataset:group | " + " | ".join(f"fold {f}" for f in folds) + " |",
        "|---" * (len(folds) + 1) + "|",
    ]
    for stratum in sorted(data["stratum"].unique()):
        cells = []
        for fold in folds:
            row = test[(test["fold"] == fold) & (test["stratum"] == stratum)]
            cells.append(
                f"{int(row['speakers'].sum())} / {int(row['recordings'].sum()):,}"
                if not row.empty
                else "-"
            )
        table.add_row(stratum, *cells)
        markdown.append(f"| {stratum} | " + " | ".join(cells) + " |")
    totals = [
        f"{int(test[test['fold'] == f]['speakers'].sum())} / {int(test[test['fold'] == f]['recordings'].sum()):,}"
        for f in folds
    ]
    table.add_row("[bold]total[/bold]", *totals)
    markdown.append("| **total** | " + " | ".join(totals) + " |")
    console.print(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"cv_{chosen['name']}_summary"
    long.to_csv(f"{stem}.csv", index=False)
    Path(f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    console.print(
        f"{OK} Saved {stem}.csv (every fold, role and group) and {stem}.md (the table above)"
    )
