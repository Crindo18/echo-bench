"""Burst-regime cooldown (blueprint section 6.1): before each block, wait until the SoC is cool."""

import time

from echo_bench.env import rpi


def wait_until_cool(
    below_c: float, max_wait_s: float, poll_s: float = 1.0
) -> tuple[float | None, float]:
    """Return (last temperature, seconds waited). Without a sensor (e.g. WSL) it returns at once."""
    start = time.monotonic()
    temperature = rpi.read_soc_temp_c()
    while temperature is not None and temperature >= below_c:
        if time.monotonic() - start >= max_wait_s:
            break
        time.sleep(poll_s)
        temperature = rpi.read_soc_temp_c()
    return temperature, time.monotonic() - start
