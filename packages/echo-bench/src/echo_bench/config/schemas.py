"""Benchmark configuration files (blueprint section 4.6), validated before anything runs.

FidelityConfig, FewShotConfig and the others join PerfConfig as their milestones arrive.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )  # a typo in a config is an error, not a silent default


class OrtSettings(_Strict):
    intra_op_threads: list[int] = Field(
        default_factory=lambda: [4], min_length=1
    )  # a list = a sweep
    allow_spinning: list[bool] = Field(default_factory=lambda: [True], min_length=1)
    graph_optimization: Literal["all", "extended", "basic", "disabled"] = "all"
    enable_cpu_mem_arena: bool = True
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    enable_profiling: bool = False


class MonitorSettings(_Strict):
    sample_hz: float = Field(10.0, gt=0, le=100)
    pmic_power: bool = False  # Pi 5 power readings arrive in M2


class PerfConfig(_Strict):
    kind: Literal["perf"] = "perf"
    models: list[str] = Field(min_length=1)  # 'name:variant' or a SHA-256 prefix
    input_durations_s: list[float] = Field(default_factory=lambda: [1, 2, 3, 5, 8], min_length=1)
    warmup_iters: int = Field(20, ge=1)
    measure_iters: int = Field(300, ge=1)
    block_size: int = Field(50, ge=1)  # interleaving granularity
    cooldown_below_c: float | None = 60.0  # burst regime; None = sustained regime
    cooldown_max_wait_s: int = 300
    disable_gc_in_timed_blocks: bool = True
    pause_between_blocks_s: float = Field(0.02, ge=0)  # lets the previous session's threads go idle
    seed: int = 42
    bootstrap_resamples: int = Field(2000, ge=100)
    ort: OrtSettings = Field(default_factory=OrtSettings)
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)

    @property
    def thermal_regime(self) -> str:
        return "sustained" if self.cooldown_below_c is None else "burst"
