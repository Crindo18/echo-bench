"""Reading a benchmark config: YAML -> validated config, plus its canonical JSON and SHA-256."""

from pathlib import Path

import yaml

from echo_bench.config.schemas import PerfConfig
from echo_bench.env.fingerprint import canonical_json, fingerprint_of


def load_perf_config(path: str | Path) -> tuple[PerfConfig, str, str]:
    config = PerfConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    data = config.model_dump(mode="json")
    return config, canonical_json(data), fingerprint_of(data)
