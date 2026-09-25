"""Hardware and software fingerprints (blueprint principle P5).

A fingerprint is the SHA-256 of a canonical JSON description. Two results with
the same hardware fingerprint came from the same machine setup, and two with the
same software fingerprint came from the same code and package versions, so every
number can be traced back to where and how it was produced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy
import onnxruntime
import psutil


def canonical_json(data: Any) -> str:
    """JSON with sorted keys and no spaces, so equal data always gives identical text."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_of(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _read_text(path: str) -> str | None:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return content.replace("\x00", "").strip() or None


def is_wsl() -> bool:
    return "microsoft" in platform.release().lower()


def device_model() -> str | None:
    """For example 'Raspberry Pi 5 Model B Rev 1.0' on a Pi, None on a normal PC."""
    return _read_text("/proc/device-tree/model")


def detect_platform(model: str | None) -> str:
    if model and "Raspberry Pi 5" in model:
        return "rpi5"
    if model and "Raspberry Pi" in model:
        return "other"
    return "desktop"


def cpu_model() -> str:
    for line in (_read_text("/proc/cpuinfo") or "").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    if shutil.which("lscpu"):  # ARM boards such as the Pi report the model here instead
        output = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
        for line in output.splitlines():
            if line.strip().lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def os_name() -> str:
    for line in (_read_text("/etc/os-release") or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            name = line.split("=", 1)[1].strip().strip('"')
            return f"{name} (WSL2)" if is_wsl() else name
    return platform.platform()


def collect_hardware(
    *, storage: str | None = None, cooling: str | None = None, power_source: str | None = None
) -> dict[str, Any]:
    """Describe this machine. The last three fields are manual tags you pass in."""
    model = device_model()
    fields: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": detect_platform(model),
        "device_model": model,
        "cpu_model": cpu_model(),
        "cpu_arch": platform.machine(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_total_mb": int(psutil.virtual_memory().total // (1024 * 1024)),
        "os_name": os_name(),
        "kernel": platform.release(),
        "storage": storage,
        "cooling": cooling,
        "power_source": power_source,
    }
    fields["fingerprint"] = fingerprint_of(fields)
    return fields


def git_state(repo_dir: str | Path = ".") -> tuple[str | None, bool]:
    """(commit hash, has uncommitted changes), or (None, False) outside a git repository."""
    if shutil.which("git") is None:
        return None, False
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        return None, False
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=False
    )
    return head.stdout.strip(), bool(status.stdout.strip())


def installed_packages() -> list[str]:
    names = {
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    return sorted(names, key=str.lower)


def collect_software(repo_dir: str | Path = ".") -> dict[str, Any]:
    """Describe the Python environment and the code version."""
    commit, dirty = git_state(repo_dir)
    fields: dict[str, Any] = {
        "python_version": platform.python_version(),
        "onnxruntime_version": onnxruntime.__version__,
        "ort_providers_json": canonical_json(onnxruntime.get_available_providers()),
        "numpy_version": numpy.__version__,
        "packages_json": canonical_json(installed_packages()),
        "git_commit": commit,
        "git_dirty": int(dirty),
    }
    fields["fingerprint"] = fingerprint_of(fields)
    return fields
