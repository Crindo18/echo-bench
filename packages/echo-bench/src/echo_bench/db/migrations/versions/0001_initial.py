"""Initial schema: the tables M1 needs (blueprint section 4.3, Alembic revision 0001_initial).

Generated with `alembic revision --autogenerate` from echo_bench/db/models.py,
then reviewed. Later milestones add tables in new revisions (0002, 0003, ...);
never edit this file once a database you care about has been created with it.

Revision ID: 0001
Revises:
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "hardware_profile",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("hostname", sa.Text(), nullable=False),
        sa.Column("platform", sa.Text(), nullable=False),
        sa.Column("device_model", sa.Text(), nullable=True),
        sa.Column("cpu_model", sa.Text(), nullable=False),
        sa.Column("cpu_arch", sa.Text(), nullable=False),
        sa.Column("cpu_cores", sa.Integer(), nullable=False),
        sa.Column("ram_total_mb", sa.Integer(), nullable=False),
        sa.Column("os_name", sa.Text(), nullable=False),
        sa.Column("kernel", sa.Text(), nullable=False),
        sa.Column("storage", sa.Text(), nullable=True),
        sa.Column("cooling", sa.Text(), nullable=True),
        sa.Column("power_source", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "platform IN ('desktop', 'rpi5', 'other')",
            name=op.f("ck_hardware_profile_platform_allowed"),
        ),
        sa.CheckConstraint("cpu_cores > 0", name=op.f("ck_hardware_profile_cpu_cores_positive")),
        sa.CheckConstraint(
            "ram_total_mb > 0", name=op.f("ck_hardware_profile_ram_total_mb_positive")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_hardware_profile")),
        sa.UniqueConstraint("fingerprint", name=op.f("uq_hardware_profile_fingerprint")),
    )
    op.create_table(
        "model_artifact",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("variant", sa.Text(), nullable=False),
        sa.Column("parent_id", sa.Text(), nullable=True),
        sa.Column("weights_state", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("param_count", sa.Integer(), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("opset", sa.Integer(), nullable=True),
        sa.Column("model_card_json", sa.Text(), nullable=False),
        sa.Column("promoted", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "variant IN ('torch_fp32', 'onnx_fp32', 'onnx_int8_dynamic', 'onnx_int8_static', 'onnx_int8_mixed')",
            name=op.f("ck_model_artifact_variant_allowed"),
        ),
        sa.CheckConstraint(
            "weights_state IN ('random_init', 'trained')",
            name=op.f("ck_model_artifact_weights_state_allowed"),
        ),
        sa.CheckConstraint(
            "json_valid(model_card_json)", name=op.f("ck_model_artifact_model_card_json_is_json")
        ),
        sa.CheckConstraint("promoted IN (0, 1)", name=op.f("ck_model_artifact_promoted_is_bool")),
        sa.CheckConstraint("size_bytes > 0", name=op.f("ck_model_artifact_size_bytes_positive")),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["model_artifact.id"],
            name=op.f("fk_model_artifact_parent_id_model_artifact"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_artifact")),
        sa.UniqueConstraint("sha256", name=op.f("uq_model_artifact_sha256")),
    )
    op.create_table(
        "schema_meta",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_schema_meta")),
    )
    op.create_table(
        "software_env",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("python_version", sa.Text(), nullable=False),
        sa.Column("onnxruntime_version", sa.Text(), nullable=False),
        sa.Column("ort_providers_json", sa.Text(), nullable=False),
        sa.Column("numpy_version", sa.Text(), nullable=False),
        sa.Column("packages_json", sa.Text(), nullable=False),
        sa.Column("git_commit", sa.Text(), nullable=True),
        sa.Column("git_dirty", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.CheckConstraint("git_dirty IN (0, 1)", name=op.f("ck_software_env_git_dirty_is_bool")),
        sa.CheckConstraint(
            "json_valid(ort_providers_json)",
            name=op.f("ck_software_env_ort_providers_json_is_json"),
        ),
        sa.CheckConstraint(
            "json_valid(packages_json)", name=op.f("ck_software_env_packages_json_is_json")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_software_env")),
        sa.UniqueConstraint("fingerprint", name=op.f("uq_software_env_fingerprint")),
    )
    op.create_table(
        "campaign",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("config_sha256", sa.Text(), nullable=False),
        sa.Column("hardware_profile_id", sa.Text(), nullable=False),
        sa.Column("software_env_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("finished_at", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('perf', 'fidelity', 'fewshot', 'kws', 'enrollment', 'e2e', 'soak')",
            name=op.f("ck_campaign_kind_allowed"),
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'aborted')",
            name=op.f("ck_campaign_status_allowed"),
        ),
        sa.CheckConstraint("json_valid(config_json)", name=op.f("ck_campaign_config_json_is_json")),
        sa.ForeignKeyConstraint(
            ["hardware_profile_id"],
            ["hardware_profile.id"],
            name=op.f("fk_campaign_hardware_profile_id_hardware_profile"),
        ),
        sa.ForeignKeyConstraint(
            ["software_env_id"],
            ["software_env.id"],
            name=op.f("fk_campaign_software_env_id_software_env"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_campaign")),
    )
    op.create_table(
        "benchmark_run",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("campaign_id", sa.Text(), nullable=False),
        sa.Column("model_artifact_id", sa.Text(), nullable=True),
        sa.Column("reference_artifact_id", sa.Text(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("ort_intra_threads", sa.Integer(), nullable=True),
        sa.Column("ort_allow_spinning", sa.Integer(), nullable=True),
        sa.Column("execution_provider", sa.Text(), nullable=True),
        sa.Column("cpu_governor", sa.Text(), nullable=True),
        sa.Column("temp_start_c", sa.REAL(), nullable=True),
        sa.Column("temp_end_c", sa.REAL(), nullable=True),
        sa.Column("throttled_start", sa.Integer(), nullable=True),
        sa.Column("throttled_end", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("invalid_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("finished_at", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'invalid')",
            name=op.f("ck_benchmark_run_status_allowed"),
        ),
        sa.CheckConstraint(
            "json_valid(params_json)", name=op.f("ck_benchmark_run_params_json_is_json")
        ),
        sa.CheckConstraint(
            "ort_allow_spinning IN (0, 1)", name=op.f("ck_benchmark_run_ort_allow_spinning_is_bool")
        ),
        sa.ForeignKeyConstraint(
            ["campaign_id"],
            ["campaign.id"],
            name=op.f("fk_benchmark_run_campaign_id_campaign"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_artifact_id"],
            ["model_artifact.id"],
            name=op.f("fk_benchmark_run_model_artifact_id_model_artifact"),
        ),
        sa.ForeignKeyConstraint(
            ["reference_artifact_id"],
            ["model_artifact.id"],
            name=op.f("fk_benchmark_run_reference_artifact_id_model_artifact"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_run")),
    )
    with op.batch_alter_table("benchmark_run", schema=None) as batch_op:
        batch_op.create_index("ix_run_campaign", ["campaign_id"], unique=False)
        batch_op.create_index("ix_run_model", ["model_artifact_id"], unique=False)

    op.create_table(
        "gate_result",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("gate", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("model_artifact_id", sa.Text(), nullable=True),
        sa.Column("comparator", sa.Text(), nullable=False),
        sa.Column("threshold", sa.REAL(), nullable=False),
        sa.Column("observed", sa.REAL(), nullable=False),
        sa.Column("passed", sa.Integer(), nullable=False),
        sa.Column("evaluated_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "comparator IN ('<=', '<', '>=', '>')", name=op.f("ck_gate_result_comparator_allowed")
        ),
        sa.CheckConstraint("passed IN (0, 1)", name=op.f("ck_gate_result_passed_is_bool")),
        sa.ForeignKeyConstraint(
            ["model_artifact_id"],
            ["model_artifact.id"],
            name=op.f("fk_gate_result_model_artifact_id_model_artifact"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_run.id"],
            name=op.f("fk_gate_result_run_id_benchmark_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_gate_result")),
    )
    op.create_table(
        "latency_sample",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("input_duration_ms", sa.Integer(), nullable=True),
        sa.Column("is_warmup", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_cold", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("latency_us", sa.Integer(), nullable=False),
        sa.Column("t_offset_ms", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "stage IN ('session_load', 'frontend', 'kws', 'encoder', 'classify', 'routing', 'tts_synth', 'tts_first_audio', 'enroll_augment', 'enroll_embed', 'enroll_prototype', 'e2e')",
            name=op.f("ck_latency_sample_stage_allowed"),
        ),
        sa.CheckConstraint("is_cold IN (0, 1)", name=op.f("ck_latency_sample_is_cold_is_bool")),
        sa.CheckConstraint("is_warmup IN (0, 1)", name=op.f("ck_latency_sample_is_warmup_is_bool")),
        sa.CheckConstraint(
            "latency_us >= 0", name=op.f("ck_latency_sample_latency_us_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_run.id"],
            name=op.f("fk_latency_sample_run_id_benchmark_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "iteration", "stage", name=op.f("pk_latency_sample")),
    )
    op.create_table(
        "resource_sample",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("t_offset_ms", sa.Integer(), nullable=False),
        sa.Column("proc_rss_mb", sa.REAL(), nullable=False),
        sa.Column("sys_mem_available_mb", sa.REAL(), nullable=False),
        sa.Column("cpu_percent", sa.REAL(), nullable=False),
        sa.Column("cpu_percent_per_core", sa.Text(), nullable=True),
        sa.Column("cpu_freq_mhz", sa.REAL(), nullable=True),
        sa.Column("soc_temp_c", sa.REAL(), nullable=True),
        sa.Column("pmic_power_w", sa.REAL(), nullable=True),
        sa.CheckConstraint(
            "cpu_percent_per_core IS NULL OR json_valid(cpu_percent_per_core)",
            name=op.f("ck_resource_sample_cpu_percent_per_core_is_json"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_run.id"],
            name=op.f("fk_resource_sample_run_id_benchmark_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "t_offset_ms", name=op.f("pk_resource_sample")),
    )
    op.create_table(
        "run_metric",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), server_default="overall", nullable=False),
        sa.Column("value", sa.REAL(), nullable=False),
        sa.Column("ci_low", sa.REAL(), nullable=True),
        sa.Column("ci_high", sa.REAL(), nullable=True),
        sa.Column("n", sa.Integer(), nullable=True),
        sa.Column("unit", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_run.id"],
            name=op.f("fk_run_metric_run_id_benchmark_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "name", "scope", name=op.f("pk_run_metric")),
    )


def downgrade() -> None:
    op.drop_table("run_metric")
    op.drop_table("resource_sample")
    op.drop_table("latency_sample")
    op.drop_table("gate_result")
    with op.batch_alter_table("benchmark_run", schema=None) as batch_op:
        batch_op.drop_index("ix_run_model")
        batch_op.drop_index("ix_run_campaign")

    op.drop_table("benchmark_run")
    op.drop_table("campaign")
    op.drop_table("software_env")
    op.drop_table("schema_meta")
    op.drop_table("model_artifact")
    op.drop_table("hardware_profile")
