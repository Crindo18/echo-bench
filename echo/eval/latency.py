"""Latency + peak-RAM profiling through ONNX Runtime (SO5 metrics).

Runs the exported INT8 ONNX encoder through onnxruntime and reports per-stage
latency, real-time factor, and peak RAM. Written to run unchanged on the
Raspberry Pi 5 target (spec Section 4): install onnxruntime on the Pi, copy the
INT8 .onnx over, and run this module. On the Pi, ORT uses the ARM CPU provider;
the numbers are the deployed encoder-stage latency in the end-to-end budget.

Reports mean ± SD over repeated runs after warmup.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np


def _peak_rss_mb() -> float:
    """Peak resident-set size in MB, cross-platform.

    Reports the OS's *peak* memory on every platform so the number stays
    comparable across dev machines and the Pi target (a current-RSS reading on
    one OS and a peak on another would be apples-to-oranges in the scorecard):
      - Linux/BSD/macOS: resource.getrusage(...).ru_maxrss  (Unix-only module)
      - Windows:         PeakWorkingSetSize via GetProcessMemoryInfo (psapi),
                         trying psutil first if it happens to be installed.
    Returns NaN only if every method fails.
    """
    import sys

    # --- Unix / macOS: `resource` is stdlib but absent on Windows ---------- #
    if sys.platform != "win32":
        try:
            import resource
            val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss units differ: bytes on macOS, KiB on Linux/BSD.
            if sys.platform == "darwin":
                return val / (1024 * 1024)
            return val / 1024
        except Exception:
            return float("nan")

    # --- Windows: prefer psutil's peak working set if available ----------- #
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
    except Exception:
        pass

    # --- Windows fallback: GetProcessMemoryInfo via ctypes (stdlib only) --- #
    # PeakWorkingSetSize is the Windows analogue of ru_maxrss, so this keeps
    # the "peak" semantics with no third-party dependency.
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass

    return float("nan")


def profile_onnx(onnx_path: str, seconds: float = 4.0, sr: int = 16000,
                 accepts_waveform: bool = True, n_mels: int = 80,
                 runs: int = 50, warmup: int = 5, threads: int = 4) -> Dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    in_names = [i.name for i in inputs]
    n = int(seconds * sr)
    # Decide input shape from the exported graph: rank-3 => log-Mel features
    # (supervised encoders; front-end runs separately on-device), rank-2 => raw
    # waveform (DPWavLM). Matches echo.quantize.EmbeddingExport.
    input_rank = len(inputs[0].shape)
    if input_rank == 3:
        hop, n_mels = 160, 80
        n_frames = n // hop + 1
        x = np.random.randn(1, n_frames, n_mels).astype(np.float32)
        lengths = np.array([n_frames], dtype=np.int64)
    else:
        x = np.random.randn(1, n).astype(np.float32)
        lengths = np.array([n], dtype=np.int64)
    feeds = {in_names[0]: x, in_names[1]: lengths}

    for _ in range(warmup):
        sess.run(None, feeds)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    times = np.array(times)
    audio_ms = seconds * 1000.0
    report = dict(
        latency_ms_mean=float(times.mean()),
        latency_ms_sd=float(times.std(ddof=1)) if runs > 1 else 0.0,
        latency_ms_p90=float(np.percentile(times, 90)),
        rtf=float(times.mean() / audio_ms),           # <1 is faster than real time
        peak_ram_mb=_peak_rss_mb(),
        audio_seconds=seconds, threads=threads, runs=runs,
    )
    return report


def print_report(name: str, r: Dict) -> None:
    print(f"\n===== latency: {name} =====")
    print(f"  audio {r['audio_seconds']}s | {r['threads']} threads | {r['runs']} runs")
    print(f"  latency  : {r['latency_ms_mean']:7.1f} ± {r['latency_ms_sd']:.1f} ms "
          f"(p90 {r['latency_ms_p90']:.1f} ms)")
    print(f"  RTF      : {r['rtf']:.3f}  ({'real-time OK' if r['rtf'] < 1 else 'SLOWER than real-time'})")
    ram = r['peak_ram_mb']
    ram_str = f"{ram:.0f} MB" if ram == ram else "n/a (probe unavailable)"
    print(f"  peak RAM : {ram_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX Runtime latency/RAM profiler")
    ap.add_argument("--onnx", required=True, help="INT8 .onnx encoder")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    r = profile_onnx(args.onnx, seconds=args.seconds, runs=args.runs,
                     warmup=args.warmup, threads=args.threads)
    print_report(args.onnx, r)


if __name__ == "__main__":
    main()
