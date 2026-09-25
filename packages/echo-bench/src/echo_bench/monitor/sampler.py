"""Background resource sampler (blueprint section 6.2).

A daemon thread records process memory, available system memory, CPU use,
CPU frequency and SoC temperature at N Hz. It only appends to an in-memory
list, so it does no disk I/O while benchmarks are being timed.
"""

import json
import threading
import time
from dataclasses import dataclass

import psutil

from echo_bench.env import rpi

MIB = 1024 * 1024


@dataclass
class Sample:
    run_id: str
    t_offset_ms: int
    proc_rss_mb: float
    sys_mem_available_mb: float
    cpu_percent: float
    cpu_percent_per_core: str
    cpu_freq_mhz: float | None
    soc_temp_c: float | None


class ResourceSampler:
    def __init__(self, sample_hz: float, t0_ns: int):
        self.interval_s = 1.0 / sample_hz
        self.t0_ns = t0_ns  # the same clock origin as the latency samples
        self.current_run_id: str | None = None  # set by the runner before each block
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._process = psutil.Process()
        self._thread = threading.Thread(target=self._loop, name="echo-bench-sampler", daemon=True)

    def start(self) -> None:
        psutil.cpu_percent(percpu=True)  # the first reading only sets a baseline
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            run_id = self.current_run_id
            if run_id is None:
                continue
            per_core = psutil.cpu_percent(percpu=True)
            self.samples.append(
                Sample(
                    run_id=run_id,
                    t_offset_ms=(time.perf_counter_ns() - self.t0_ns) // 1_000_000,
                    proc_rss_mb=self._process.memory_info().rss / MIB,
                    sys_mem_available_mb=psutil.virtual_memory().available / MIB,
                    cpu_percent=sum(per_core) / len(per_core),
                    cpu_percent_per_core=json.dumps(per_core),
                    cpu_freq_mhz=rpi.read_cpu_freq_mhz(),
                    soc_temp_c=rpi.read_soc_temp_c(),
                )
            )
