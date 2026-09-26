"""Tables of the ECHO-Bench results database (blueprint section 4.3), as SQLAlchemy 2.0 models.

Revision 0001 (M0) created the tables that M1 needs:
    provenance  schema_meta, hardware_profile, software_env
    models      model_artifact
    execution   campaign, benchmark_run, latency_sample, resource_sample
    results     run_metric, gate_result
Revision 0002 (M3) adds the data layer:
    data        dataset, speaker, label, utterance, cv_plan, fold_assignment
    plus model_artifact.cv_plan_id/fold_index and benchmark_run.cv_plan_id/fold_index.

Still to come as new migrations: embedding_set, fewshot_condition, episode,
episode_support and prediction in M5 (with benchmark_run.embedding_set_id and
latency_sample.utterance_id); manual_measurement in M8.

Conventions (section 4.1): text UUID keys, ISO-8601 UTC text timestamps, 0/1
integers for booleans, JSON stored as text and checked with json_valid(), and
latencies as integer microseconds.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    REAL,
    CheckConstraint,
    ForeignKey,
    Index,
    MetaData,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Predictable constraint names make future migrations (Alembic batch mode) reliable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

MODEL_VARIANTS = (
    "torch_fp32",
    "onnx_fp32",
    "onnx_int8_dynamic",
    "onnx_int8_static",
    "onnx_int8_mixed",
)
CAMPAIGN_KINDS = ("perf", "fidelity", "fewshot", "kws", "enrollment", "e2e", "soak")
LATENCY_STAGES = (
    "session_load",
    "frontend",
    "kws",
    "encoder",
    "classify",
    "routing",
    "tts_synth",
    "tts_first_audio",
    "enroll_augment",
    "enroll_embed",
    "enroll_prototype",
    "e2e",
)


def new_id() -> str:
    """A fresh random UUID as text. Random keys let databases from different machines merge (ADR-2)."""
    return str(uuid.uuid4())


def utc_now() -> str:
    """The current time as ISO-8601 UTC text, e.g. 2026-09-24T15:04:05.123Z."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def is_bool(column: str) -> CheckConstraint:
    return CheckConstraint(f"{column} IN (0, 1)", name=f"{column}_is_bool")


def is_json(column: str, *, nullable: bool = False) -> CheckConstraint:
    rule = f"json_valid({column})"
    if nullable:
        rule = f"{column} IS NULL OR {rule}"
    return CheckConstraint(rule, name=f"{column}_is_json")


def one_of(column: str, *values: str) -> CheckConstraint:
    allowed = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=f"{column}_allowed")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {str: Text(), float: REAL()}


# -- Provenance -------------------------------------------------------------


class SchemaMeta(Base):
    __tablename__ = "schema_meta"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str]


class HardwareProfile(Base):
    __tablename__ = "hardware_profile"
    __table_args__ = (
        one_of("platform", "desktop", "rpi5", "other"),
        CheckConstraint("cpu_cores > 0", name="cpu_cores_positive"),
        CheckConstraint("ram_total_mb > 0", name="ram_total_mb_positive"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    fingerprint: Mapped[str] = mapped_column(unique=True)  # sha256 of the fields below
    hostname: Mapped[str]
    platform: Mapped[str]
    device_model: Mapped[str | None]  # /proc/device-tree/model on the Pi
    cpu_model: Mapped[str]
    cpu_arch: Mapped[str]  # x86_64 | aarch64
    cpu_cores: Mapped[int]
    ram_total_mb: Mapped[int]
    os_name: Mapped[str]
    kernel: Mapped[str]
    storage: Mapped[str | None]  # manual tag, e.g. '64GB SanDisk microSD'
    cooling: Mapped[str | None]  # manual tag: active_cooler | passive | none
    power_source: Mapped[str | None]  # manual tag: official_27w_psu | powerbank_pd_board
    created_at: Mapped[str] = mapped_column(default=utc_now)


class SoftwareEnv(Base):
    __tablename__ = "software_env"
    __table_args__ = (
        is_json("ort_providers_json"),
        is_json("packages_json"),
        is_bool("git_dirty"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    fingerprint: Mapped[str] = mapped_column(unique=True)
    python_version: Mapped[str]
    onnxruntime_version: Mapped[str]
    ort_providers_json: Mapped[str]
    numpy_version: Mapped[str]
    packages_json: Mapped[str]  # every installed package and version
    git_commit: Mapped[str | None]
    git_dirty: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    created_at: Mapped[str] = mapped_column(default=utc_now)


# -- Models -------------------------------------------------------------------


class ModelArtifact(Base):
    __tablename__ = "model_artifact"
    __table_args__ = (
        one_of("variant", *MODEL_VARIANTS),
        one_of("weights_state", "random_init", "trained"),
        CheckConstraint("size_bytes > 0", name="size_bytes_positive"),
        is_json("model_card_json"),
        is_bool("promoted"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    name: Mapped[str]  # e.g. ebf-12m
    variant: Mapped[str]
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_artifact.id")
    )  # INT8 -> FP32 source
    weights_state: Mapped[str]
    file_path: Mapped[str]
    sha256: Mapped[str] = mapped_column(unique=True)
    size_bytes: Mapped[int]
    param_count: Mapped[int | None]
    embedding_dim: Mapped[int]
    opset: Mapped[int | None]
    model_card_json: Mapped[str]
    promoted: Mapped[int] = mapped_column(
        default=0, server_default=text("0")
    )  # passed desktop gates
    created_at: Mapped[str] = mapped_column(default=utc_now)
    cv_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("cv_plan.id")
    )  # fold-matched encoder (leakage rule L4); added in M3
    fold_index: Mapped[int | None]


# -- Execution ------------------------------------------------------------------


class Campaign(Base):
    """One CLI invocation, e.g. 'perf sweep on the Pi' (section 4.2)."""

    __tablename__ = "campaign"
    __table_args__ = (
        one_of("kind", *CAMPAIGN_KINDS),
        is_json("config_json"),
        one_of("status", "running", "completed", "failed", "aborted"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    name: Mapped[str]
    kind: Mapped[str]
    config_json: Mapped[str]
    config_sha256: Mapped[str]
    hardware_profile_id: Mapped[str] = mapped_column(ForeignKey("hardware_profile.id"))
    software_env_id: Mapped[str] = mapped_column(ForeignKey("software_env.id"))
    status: Mapped[str]
    started_at: Mapped[str] = mapped_column(default=utc_now)
    finished_at: Mapped[str | None]
    notes: Mapped[str | None]


class BenchmarkRun(Base):
    """One cell of a campaign: one model variant x one setting (section 4.2)."""

    __tablename__ = "benchmark_run"
    __table_args__ = (
        is_json("params_json"),
        is_bool("ort_allow_spinning"),
        one_of("status", "running", "completed", "failed", "invalid"),
        Index("ix_run_campaign", "campaign_id"),
        Index("ix_run_model", "model_artifact_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaign.id", ondelete="CASCADE"))
    model_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("model_artifact.id"))
    reference_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("model_artifact.id"))
    params_json: Mapped[str]  # this cell's settings
    seed: Mapped[int]
    ort_intra_threads: Mapped[int | None]
    ort_allow_spinning: Mapped[int | None]
    execution_provider: Mapped[str | None]
    cpu_governor: Mapped[str | None]
    temp_start_c: Mapped[float | None]
    temp_end_c: Mapped[float | None]
    throttled_start: Mapped[int | None]  # vcgencmd get_throttled bitmask
    throttled_end: Mapped[int | None]
    status: Mapped[str]
    invalid_reason: Mapped[str | None]  # e.g. 'under-voltage occurred (bit 16)'
    started_at: Mapped[str] = mapped_column(default=utc_now)
    finished_at: Mapped[str | None]
    cv_plan_id: Mapped[str | None] = mapped_column(ForeignKey("cv_plan.id"))  # added in M3
    fold_index: Mapped[int | None]


class LatencySample(Base):
    __tablename__ = "latency_sample"
    __table_args__ = (
        one_of("stage", *LATENCY_STAGES),
        is_bool("is_warmup"),
        is_bool("is_cold"),
        CheckConstraint("latency_us >= 0", name="latency_us_non_negative"),
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("benchmark_run.id", ondelete="CASCADE"), primary_key=True
    )
    iteration: Mapped[int] = mapped_column(primary_key=True)
    stage: Mapped[str] = mapped_column(primary_key=True)
    input_duration_ms: Mapped[int | None]
    is_warmup: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    is_cold: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    latency_us: Mapped[int]
    t_offset_ms: Mapped[int]  # lines up with resource_sample


class ResourceSample(Base):
    __tablename__ = "resource_sample"
    __table_args__ = (is_json("cpu_percent_per_core", nullable=True),)

    run_id: Mapped[str] = mapped_column(
        ForeignKey("benchmark_run.id", ondelete="CASCADE"), primary_key=True
    )
    t_offset_ms: Mapped[int] = mapped_column(primary_key=True)
    proc_rss_mb: Mapped[float]
    sys_mem_available_mb: Mapped[float]
    cpu_percent: Mapped[float]
    cpu_percent_per_core: Mapped[str | None]
    cpu_freq_mhz: Mapped[float | None]
    soc_temp_c: Mapped[float | None]
    pmic_power_w: Mapped[float | None]  # Pi 5 PMIC estimate; excludes USB peripherals


# -- Results --------------------------------------------------------------------


class RunMetric(Base):
    __tablename__ = "run_metric"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("benchmark_run.id", ondelete="CASCADE"), primary_key=True
    )
    name: Mapped[str] = mapped_column(primary_key=True)  # accuracy | latency_p95_ms | ...
    scope: Mapped[str] = mapped_column(
        primary_key=True, default="overall", server_default="overall"
    )  # overall | speaker:F02 | stage:encoder | ...
    value: Mapped[float]
    ci_low: Mapped[float | None]
    ci_high: Mapped[float | None]
    n: Mapped[int | None]
    unit: Mapped[str | None]


class GateResult(Base):
    __tablename__ = "gate_result"
    __table_args__ = (
        one_of("comparator", "<=", "<", ">=", ">"),
        is_bool("passed"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    gate: Mapped[str]  # G1_footprint, G5a_e2e_latency_p95, ...
    source: Mapped[str]  # 'SO1 D1' | 'Sec 1.7.4.6.2' | 'internal'
    run_id: Mapped[str | None] = mapped_column(ForeignKey("benchmark_run.id", ondelete="CASCADE"))
    model_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("model_artifact.id"))
    comparator: Mapped[str]
    threshold: Mapped[float]
    observed: Mapped[float]
    passed: Mapped[int]
    evaluated_at: Mapped[str] = mapped_column(default=utc_now)


# -- Data (M3, revision 0002) ------------------------------------------------------
# Privacy by design: speakers are pseudonymous codes only. No names, contact details
# or diagnoses; add a column only if the analysis plan needs it AND consent covers it.

DATASET_ACCESS = ("licensed_corpus", "restricted_participant", "synthetic")
SPEAKER_COHORTS = ("dysarthric", "control", "participant", "voice_actor")
FOLD_ROLES = ("train", "val", "test")
CV_STRATEGIES = ("stratified_group_kfold", "leave_one_speaker_out")


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    name: Mapped[str]  # torgo | uaspeech | echo_participants | negatives | synthetic
    version: Mapped[str]
    access_level: Mapped[str]
    root_hint: Mapped[str | None]  # documentation only; the real root comes from local config
    manifest_path: Mapped[str]
    manifest_sha256: Mapped[str]
    primary_channel: Mapped[str | None]  # the single mic channel used for evaluation
    created_at: Mapped[str] = mapped_column(default=utc_now)

    __table_args__ = (
        one_of("access_level", *DATASET_ACCESS),
        UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )


class Speaker(Base):
    __tablename__ = "speaker"
    __table_args__ = (
        one_of("cohort", *SPEAKER_COHORTS),
        CheckConstraint(
            "intelligibility_pct IS NULL OR intelligibility_pct BETWEEN 0 AND 100",
            name="intelligibility_pct_range",
        ),
        CheckConstraint("sex IS NULL OR sex IN ('F', 'M')", name="sex_allowed"),
        UniqueConstraint("dataset_id", "code", name="uq_speaker_dataset_code"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset.id"))
    code: Mapped[str]  # pseudonymous only: F02, M05, P07
    cohort: Mapped[str]
    severity_tier: Mapped[str | None]  # very_low|low|mid|high (UASpeech) or mild..severe (TORGO)
    intelligibility_pct: Mapped[float | None]
    sex: Mapped[str | None]


class Label(Base):
    __tablename__ = "label"
    __table_args__ = (
        is_bool("is_emergency"),
        UniqueConstraint("dataset_id", "key", name="uq_label_dataset_key"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset.id"))
    key: Mapped[str]  # 'pain' | UASpeech word code
    text: Mapped[str]  # sentence template or word
    category: Mapped[str | None]  # Emergency | Basic Needs | ... | proxy:<group> for corpora
    is_emergency: Mapped[int] = mapped_column(default=0, server_default=text("0"))


class Utterance(Base):
    __tablename__ = "utterance"
    __table_args__ = (
        is_bool("is_primary_channel"),
        CheckConstraint("duration_ms > 0", name="duration_ms_positive"),
        UniqueConstraint("dataset_id", "audio_path", name="uq_utterance_dataset_audio_path"),
        Index("ix_utterance_speaker_label", "speaker_id", "label_id"),
        Index("ix_utterance_group", "recording_group_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset.id"))
    speaker_id: Mapped[str] = mapped_column(ForeignKey("speaker.id"))
    label_id: Mapped[str] = mapped_column(ForeignKey("label.id"))
    recording_group_id: Mapped[str]  # one physical utterance across all mics/channels
    session: Mapped[str | None]  # TORGO session | UASpeech block B1-B3 | ECHO session
    channel: Mapped[str | None]  # mic/channel id
    is_primary_channel: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    audio_path: Mapped[str]  # relative to the dataset root
    audio_sha256: Mapped[str]
    sample_rate: Mapped[int]
    duration_ms: Mapped[int]
    recorded_at: Mapped[str | None]  # participant sessions: enforces the >=24 h test separation


class CvPlan(Base):
    __tablename__ = "cv_plan"
    __table_args__ = (
        one_of("strategy", *CV_STRATEGIES),
        CheckConstraint("k >= 2", name="k_at_least_2"),
        is_json("dataset_ids_json"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(unique=True)  # e.g. sgkf5-severity-s42
    strategy: Mapped[str]
    k: Mapped[int]
    stratify_by: Mapped[str]  # severity_tier
    seed: Mapped[int]
    dataset_ids_json: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=utc_now)


class FoldAssignment(Base):
    __tablename__ = "fold_assignment"
    __table_args__ = (
        one_of("role", *FOLD_ROLES),
        CheckConstraint("fold_index >= 0", name="fold_index_nonnegative"),
    )

    cv_plan_id: Mapped[str] = mapped_column(ForeignKey("cv_plan.id"), primary_key=True)
    fold_index: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[str] = mapped_column(ForeignKey("speaker.id"), primary_key=True)
    role: Mapped[str]
