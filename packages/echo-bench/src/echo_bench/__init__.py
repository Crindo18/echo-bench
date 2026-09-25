"""echo_bench: the benchmarking harness that drives echo_core (blueprint section 2.2).

Runs on the desktop and on the Raspberry Pi. Like echo_core, it must never
import torch, espnet, librosa or pandas.
"""

__version__ = "0.1.0"
