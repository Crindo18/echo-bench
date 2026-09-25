"""configs/gates.yaml holds every section 6.8 gate, and comparisons work as written."""

from pathlib import Path

import pytest

from echo_bench.gates import Gate, load_gates

GATES_FILE = Path(__file__).resolve().parents[2] / "configs" / "gates.yaml"


def test_every_gate_from_section_6_8_is_present() -> None:
    gates = load_gates(GATES_FILE)
    prefixes = {key.split("_")[0] for key in gates}
    expected = {"G1", "G2a", "G2b", "G2c", "G2d", "G3a", "G3b", "G3c", "G3d"}
    expected |= {"G4a", "G4b", "G4c", "G4d", "G5a", "G5b", "G6", "G7"}
    assert prefixes == expected
    assert gates["G1_footprint"].threshold == 16.0
    assert gates["G5a_e2e_latency_p95"].threshold == 1500


@pytest.mark.parametrize(
    ("comparator", "threshold", "observed", "expected"),
    [
        ("<=", 16.0, 16.0, True),
        ("<=", 16.0, 16.01, False),
        ("<", 10.0, 10.0, False),
        (">=", 80.0, 80.0, True),
        (">", 60.0, 60.0, False),
        (">", 60.0, 60.5, True),
    ],
)
def test_comparisons(comparator: str, threshold: float, observed: float, expected: bool) -> None:
    assert Gate("G", "metric", comparator, threshold, "source").passes(observed) is expected


def test_unknown_comparator_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "gates.yaml"
    bad.write_text("gates:\n  G1_x: {metric: m, comparator: '=<', threshold: 1, source: s}\n")
    with pytest.raises(ValueError, match="comparator"):
        load_gates(bad)
