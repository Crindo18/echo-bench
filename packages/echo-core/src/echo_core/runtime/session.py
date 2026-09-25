"""Creating ONNX Runtime sessions the same way on every machine (blueprint section 6.1).

Nothing is left to ONNX Runtime's defaults: thread count, spinning, graph
optimization and memory arena are always set explicitly, so a benchmark and
the device use exactly the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import onnxruntime as ort

GraphOptimization = Literal["all", "extended", "basic", "disabled"]
_LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}


@dataclass(frozen=True)
class OrtOptions:
    intra_op_threads: int = 4
    allow_spinning: bool = True  # lower latency, but burns CPU and heat while idle-waiting
    graph_optimization: GraphOptimization = "all"
    enable_cpu_mem_arena: bool = True
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    enable_profiling: bool = False
    profile_prefix: str = "ort_profile"


def make_session(model_path: str | Path, options: OrtOptions | None = None) -> Any:
    """Return an onnxruntime.InferenceSession with every setting made explicit."""
    opts = options or OrtOptions()
    so = ort.SessionOptions()
    so.intra_op_num_threads = opts.intra_op_threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry(
        "session.intra_op.allow_spinning", "1" if opts.allow_spinning else "0"
    )
    so.graph_optimization_level = _LEVELS[opts.graph_optimization]
    so.enable_cpu_mem_arena = opts.enable_cpu_mem_arena
    if opts.enable_profiling:
        so.enable_profiling = True
        so.profile_file_prefix = opts.profile_prefix
    return ort.InferenceSession(str(model_path), sess_options=so, providers=list(opts.providers))


def runtime_info() -> dict[str, Any]:
    return {"onnxruntime": ort.__version__, "providers": list(ort.get_available_providers())}
