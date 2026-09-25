"""The device-side packages must never import heavy desktop libraries (blueprint sections 2.2 and 8).

The Raspberry Pi never installs PyTorch, ESPnet, librosa or pandas, so a single
stray import in echo_core or echo_bench would crash the harness on the Pi. The
layering rule also keeps echo_core independent of the harness and of analysis.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEAVY = {"torch", "torchaudio", "espnet", "espnet2", "librosa", "pandas"}
RULES = {
    "packages/echo-core/src": HEAVY | {"echo_bench", "echo_analysis", "echo_train"},
    "packages/echo-bench/src": HEAVY | {"echo_analysis", "echo_train"},
}


def imported_top_level_modules(source: str) -> set[str]:
    """Every top-level module a piece of code imports, including imports inside functions."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("src_dir", sorted(RULES))
def test_no_forbidden_imports(src_dir: str) -> None:
    violations = []
    for path in sorted((ROOT / src_dir).rglob("*.py")):
        bad = imported_top_level_modules(path.read_text(encoding="utf-8")) & RULES[src_dir]
        if bad:
            violations.append(f"{path.relative_to(ROOT)} imports {sorted(bad)}")
    assert not violations, "Forbidden imports:\n" + "\n".join(violations)


def test_scanner_finds_imports_hidden_inside_functions() -> None:
    source = "def f():\n    import torch.nn\n    from pandas import DataFrame\n"
    assert {"torch", "pandas"} <= imported_top_level_modules(source)
