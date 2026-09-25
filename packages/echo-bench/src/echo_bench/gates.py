"""Gate thresholds (configs/gates.yaml) and gate_result rows (blueprint section 6.8)."""

import operator
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from echo_bench.db.models import GateResult

DEFAULT_GATES_FILE = Path("configs/gates.yaml")
_COMPARE = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt}


@dataclass(frozen=True)
class Gate:
    key: str
    metric: str
    comparator: str
    threshold: float
    source: str

    def passes(self, observed: float) -> bool:
        return bool(_COMPARE[self.comparator](observed, self.threshold))


def load_gates(path: Path = DEFAULT_GATES_FILE) -> dict[str, Gate]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["gates"]
    gates = {}
    for key, spec in raw.items():
        if spec["comparator"] not in _COMPARE:
            raise ValueError(f"{key}: unknown comparator {spec['comparator']!r}")
        gates[key] = Gate(
            key, spec["metric"], spec["comparator"], float(spec["threshold"]), str(spec["source"])
        )
    return gates


def record_gate(
    session: Session,
    gate: Gate,
    observed: float,
    *,
    run_id: str | None = None,
    model_artifact_id: str | None = None,
) -> GateResult:
    row = GateResult(
        gate=gate.key,
        source=gate.source,
        run_id=run_id,
        model_artifact_id=model_artifact_id,
        comparator=gate.comparator,
        threshold=gate.threshold,
        observed=float(observed),
        passed=int(gate.passes(observed)),
    )
    session.add(row)
    return row
