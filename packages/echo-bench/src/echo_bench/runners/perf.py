"""The perf runner (blueprint sections 6.1, 6.2 and 6.7).

How a perf campaign is timed:
- Every (model x ONNX Runtime setting) pair is one run with its own session.
  Creating the session is timed as 'session_load' (the cold-start cost).
- The first inference of every run is flagged is_cold.
- Warm-up iterations are stored with is_warmup=1 and left out of the statistics.
- Measured iterations run in blocks, in a seeded random order across models and
  input lengths, so a machine that slowly heats up can't favour one model.
- Nothing is written to disk or printed inside a timed block, and Python's
  garbage collector is paused there.
- A background thread records memory, CPU, frequency and temperature.
- Everything is written to the database after the timing is over.
"""

import gc
import itertools
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
from sqlalchemy.orm import Session

from echo_bench.config.schemas import PerfConfig
from echo_bench.db.models import (
    BenchmarkRun,
    Campaign,
    LatencySample,
    ModelArtifact,
    ResourceSample,
    RunMetric,
    utc_now,
)
from echo_bench.db.repository import get_or_create_hardware_profile, get_or_create_software_env
from echo_bench.env import rpi
from echo_bench.env.fingerprint import collect_hardware, collect_software
from echo_bench.inputs import synthetic_inputs
from echo_bench.metrics.latency import summarize
from echo_bench.monitor.cooldown import wait_until_cool
from echo_bench.monitor.sampler import MIB, ResourceSampler
from echo_bench.registry import card_of, resolve, verify_artifact_file
from echo_core.model_card import ModelCard
from echo_core.runtime.session import OrtOptions, make_session

try:
    import resource
except ImportError:  # Windows has no `resource` module; the sampler's maximum is used instead
    resource = None  # type: ignore[assignment]


@dataclass
class _RunState:
    row: BenchmarkRun
    card: ModelCard
    session: Any
    latencies: list[LatencySample] = field(default_factory=list)
    next_iteration: int = 0
    cold_done: bool = False


@dataclass
class _Cell:
    state: _RunState
    duration_s: float
    inputs: dict[str, np.ndarray]


def _record(cell: _Cell, starts: list[int], ends: list[int], t0: int, *, warmup: bool) -> None:
    state, duration_ms = cell.state, int(round(cell.duration_s * 1000))
    for start, end in zip(starts, ends, strict=True):
        state.latencies.append(
            LatencySample(
                run_id=state.row.id,
                iteration=state.next_iteration,
                stage="encoder",
                input_duration_ms=duration_ms,
                is_warmup=int(warmup),
                is_cold=int(not state.cold_done),
                latency_us=(end - start) // 1000,
                t_offset_ms=(start - t0) // 1_000_000,
            )
        )
        state.next_iteration += 1
        state.cold_done = True


def _time_block(cell: _Cell, size: int, t0: int, *, warmup: bool, pause_gc: bool) -> None:
    run, inputs = cell.state.session.run, cell.inputs
    starts, ends = [0] * size, [0] * size
    if pause_gc:
        gc.collect()
        gc.disable()
    try:
        for i in range(size):
            starts[i] = time.perf_counter_ns()
            run(None, inputs)
            ends[i] = time.perf_counter_ns()
    finally:
        if pause_gc:
            gc.enable()
    _record(cell, starts, ends, t0, warmup=warmup)  # bookkeeping happens after the timed loop


def _invalid_reason(config: PerfConfig, start: int | None, end: int | None) -> str | None:
    """Validity rules 1 and 2 (section 6.7), from `vcgencmd get_throttled` at start and end."""
    reasons = []
    for moment, value in (("start", start), ("end", end)):
        if value is None:
            continue
        for bit in rpi.UNDER_VOLTAGE_BITS:
            if value & (1 << bit):
                reasons.append(f"under-voltage flag, bit {bit}, at {moment}")
        if config.thermal_regime == "burst":
            for bit in rpi.THROTTLING_BITS:
                if value & (1 << bit):
                    reasons.append(f"throttling flag, bit {bit}, at {moment}")
    return "; ".join(reasons) or None


def _run_setting(
    db: Session,
    config: PerfConfig,
    campaign: Campaign,
    models: list[tuple[ModelArtifact, ModelCard]],
    options: OrtOptions,
    sampler: ResourceSampler,
    t0: int,
    progress: Callable[[str], None],
) -> list[_RunState]:
    throttled_start, temp_start = rpi.read_throttled(), rpi.read_soc_temp_c()
    params = {
        "input_durations_s": config.input_durations_s,
        "warmup_iters": config.warmup_iters,
        "measure_iters": config.measure_iters,
        "block_size": config.block_size,
        "thermal_regime": config.thermal_regime,
        "cooldown_below_c": config.cooldown_below_c,
        "pause_between_blocks_s": config.pause_between_blocks_s,
        "gc_paused_in_blocks": config.disable_gc_in_timed_blocks,
        "graph_optimization": options.graph_optimization,
        "enable_cpu_mem_arena": options.enable_cpu_mem_arena,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    states = []
    for artifact, card in models:
        run = BenchmarkRun(
            campaign_id=campaign.id,
            model_artifact_id=artifact.id,
            params_json=json.dumps(params | {"model": card.ref}),
            seed=config.seed,
            ort_intra_threads=options.intra_op_threads,
            ort_allow_spinning=int(options.allow_spinning),
            execution_provider=options.providers[0],
            cpu_governor=rpi.read_cpu_governor(),
            temp_start_c=temp_start,
            throttled_start=throttled_start,
            status="running",
        )
        db.add(run)
        db.flush()
        sampler.current_run_id = run.id
        started = time.perf_counter_ns()
        session = make_session(artifact.file_path, options)
        loaded = time.perf_counter_ns()
        state = _RunState(row=run, card=card, session=session)
        state.latencies.append(
            LatencySample(
                run_id=run.id,
                iteration=0,
                stage="session_load",
                is_warmup=0,
                is_cold=1,
                latency_us=(loaded - started) // 1000,
                t_offset_ms=(started - t0) // 1_000_000,
            )
        )
        states.append(state)
    db.commit()  # the runs are visible in the database even if timing crashes

    cells = [
        _Cell(
            state, seconds, synthetic_inputs(state.card, seconds, config.seed + int(seconds * 1000))
        )
        for state in states
        for seconds in config.input_durations_s
    ]
    for cell in cells:  # warm-up; each run's very first inference is its cold one
        sampler.current_run_id = cell.state.row.id
        _time_block(cell, config.warmup_iters, t0, warmup=True, pause_gc=False)

    sizes = [config.block_size] * (config.measure_iters // config.block_size)
    if config.measure_iters % config.block_size:
        sizes.append(config.measure_iters % config.block_size)
    schedule = [(cell, size) for cell in cells for size in sizes]
    order = np.random.default_rng(config.seed).permutation(len(schedule))
    step = max(1, len(order) // 5)
    for done, index in enumerate(order, start=1):
        cell, size = schedule[index]
        if config.cooldown_below_c is not None:
            wait_until_cool(config.cooldown_below_c, config.cooldown_max_wait_s)
        if config.pause_between_blocks_s:
            time.sleep(config.pause_between_blocks_s)
        sampler.current_run_id = cell.state.row.id
        _time_block(cell, size, t0, warmup=False, pause_gc=config.disable_gc_in_timed_blocks)
        if done % step == 0 or done == len(order):
            progress(f"  {done}/{len(order)} timed blocks done")

    throttled_end, temp_end = rpi.read_throttled(), rpi.read_soc_temp_c()
    for state in states:
        reason = _invalid_reason(config, throttled_start, throttled_end)
        state.row.throttled_end = throttled_end
        state.row.temp_end_c = temp_end
        state.row.status = "invalid" if reason else "completed"
        state.row.invalid_reason = reason
        state.row.finished_at = utc_now()
        state.session = None  # release the ONNX Runtime session
    return states


def _metrics(
    config: PerfConfig, state: _RunState, samples: list[Any], peak_rss_mb: float
) -> list[RunMetric]:
    run_id = state.row.id
    rows: list[RunMetric] = []
    load = next(s for s in state.latencies if s.stage == "session_load")
    rows.append(
        RunMetric(
            run_id=run_id,
            name="session_load_ms",
            scope="overall",
            value=load.latency_us / 1000,
            n=1,
            unit="ms",
        )
    )
    cold = next((s for s in state.latencies if s.stage == "encoder" and s.is_cold), None)
    if cold is not None:
        scope = f"encoder@{cold.input_duration_ms / 1000:g}s"
        rows.append(
            RunMetric(
                run_id=run_id,
                name="cold_inference_ms",
                scope=scope,
                value=cold.latency_us / 1000,
                n=1,
                unit="ms",
            )
        )

    measured: dict[int, list[int]] = defaultdict(list)
    for sample in state.latencies:
        if (
            sample.stage == "encoder"
            and not sample.is_warmup
            and sample.input_duration_ms is not None
        ):
            measured[sample.input_duration_ms].append(sample.latency_us)
    for duration_ms, values in sorted(measured.items()):
        stats = summarize(
            np.array(values) / 1000, resamples=config.bootstrap_resamples, seed=config.seed
        )
        for name, (value, low, high) in stats.items():
            rows.append(
                RunMetric(
                    run_id=run_id,
                    name=name,
                    scope=f"encoder@{duration_ms / 1000:g}s",
                    value=value,
                    ci_low=low,
                    ci_high=high,
                    n=len(values),
                    unit="ms",
                )
            )

    # Memory is process-wide: every run of the campaign has its session loaded at once.
    rows.append(
        RunMetric(run_id=run_id, name="peak_rss_mb", scope="process", value=peak_rss_mb, unit="MiB")
    )
    if samples:
        total_mb = psutil.virtual_memory().total / MIB
        min_available = min(s.sys_mem_available_mb for s in samples)
        rows.append(
            RunMetric(
                run_id=run_id,
                name="peak_sys_used_mb",
                scope="system",
                value=total_mb - min_available,
                unit="MiB",
            )
        )
        temps = [s.soc_temp_c for s in samples if s.run_id == run_id and s.soc_temp_c is not None]
        if temps:
            rows.append(
                RunMetric(
                    run_id=run_id,
                    name="max_soc_temp_c",
                    scope="overall",
                    value=max(temps),
                    n=len(temps),
                    unit="C",
                )
            )
    return rows


def run_perf(
    db: Session,
    config: PerfConfig,
    *,
    name: str,
    config_json: str,
    config_sha256: str,
    progress: Callable[[str], None] = print,
) -> Campaign:
    """Run one perf campaign and write everything to the database. Returns the campaign row."""
    hw = get_or_create_hardware_profile(db, collect_hardware())
    sw = get_or_create_software_env(db, collect_software())
    campaign = Campaign(
        name=name,
        kind="perf",
        config_json=config_json,
        config_sha256=config_sha256,
        hardware_profile_id=hw.id,
        software_env_id=sw.id,
        status="running",
        notes="exploratory: uncommitted code changes (section 6.7)" if sw.git_dirty else None,
    )
    db.add(campaign)
    db.commit()

    try:
        models = []
        for ref in config.models:
            artifact = resolve(db, ref)
            verify_artifact_file(artifact)  # validity rule 3
            models.append((artifact, card_of(artifact)))

        t0 = time.perf_counter_ns()
        sampler = ResourceSampler(config.monitor.sample_hz, t0)
        sampler.start()
        states: list[_RunState] = []
        try:
            for threads, spinning in itertools.product(
                config.ort.intra_op_threads, config.ort.allow_spinning
            ):
                progress(f"ONNX Runtime: {threads} threads, spinning {'on' if spinning else 'off'}")
                options = OrtOptions(
                    intra_op_threads=threads,
                    allow_spinning=spinning,
                    graph_optimization=config.ort.graph_optimization,
                    enable_cpu_mem_arena=config.ort.enable_cpu_mem_arena,
                    providers=tuple(config.ort.providers),
                    enable_profiling=config.ort.enable_profiling,
                )
                states += _run_setting(db, config, campaign, models, options, sampler, t0, progress)
        finally:
            sampler.stop()

        if resource is not None:
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
        else:
            peak_rss_mb = max((s.proc_rss_mb for s in sampler.samples), default=0.0)
        seen: set[tuple[str, int]] = set()
        for sample in sampler.samples:
            if (sample.run_id, sample.t_offset_ms) not in seen:
                seen.add((sample.run_id, sample.t_offset_ms))
                db.add(ResourceSample(**vars(sample)))
        for state in states:
            db.add_all(state.latencies)
            db.add_all(_metrics(config, state, sampler.samples, peak_rss_mb))
        campaign.status = "completed"
        campaign.finished_at = utc_now()
        db.commit()
        return campaign
    except BaseException as error:
        db.rollback()
        campaign.status = "aborted" if isinstance(error, KeyboardInterrupt) else "failed"
        campaign.finished_at = utc_now()
        db.commit()
        raise
