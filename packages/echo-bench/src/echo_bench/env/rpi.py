"""Raspberry Pi 5 readers (blueprint section 6.2): throttle flags, temperature, CPU frequency.

Temperature and frequency come straight from sysfs files, which is cheap.
Throttle flags need the `vcgencmd` program, so they are read only at the start
and end of a run. On a normal PC these functions return None.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

THROTTLE_BITS: dict[int, str] = {
    0: "under-voltage now",
    1: "ARM frequency capped now",
    2: "throttled now",
    3: "soft temperature limit now",
    16: "under-voltage has occurred",
    17: "ARM frequency capping has occurred",
    18: "throttling has occurred",
    19: "soft temperature limit has occurred",
}
UNDER_VOLTAGE_BITS = (0, 16)  # a run with either bit set is invalid (section 6.7, rule 1)
THROTTLING_BITS = (2, 18)  # invalid during a burst-regime run (section 6.7, rule 2)


def parse_throttled(output: str) -> int:
    """'throttled=0x50000' -> 327680."""
    return int(output.strip().split("=", 1)[1], 16)


def decode_throttled(value: int) -> list[str]:
    return [label for bit, label in THROTTLE_BITS.items() if value & (1 << bit)]


def has_any_bit(value: int, bits: tuple[int, ...]) -> bool:
    return any(value & (1 << bit) for bit in bits)


def read_throttled() -> int | None:
    if shutil.which("vcgencmd") is None:
        return None
    result = subprocess.run(
        ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5, check=False
    )
    if result.returncode != 0 or "=" not in result.stdout:
        return None
    return parse_throttled(result.stdout)


def _read_number(path: str) -> float | None:
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def read_soc_temp_c() -> float | None:
    millidegrees = _read_number("/sys/class/thermal/thermal_zone0/temp")
    return None if millidegrees is None else millidegrees / 1000.0


def read_cpu_freq_mhz() -> float | None:
    khz = _read_number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return None if khz is None else khz / 1000.0


def read_cpu_governor() -> str | None:
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return None
