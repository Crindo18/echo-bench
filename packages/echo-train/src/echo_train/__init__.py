"""echo_train: build, export and quantize the encoder (desktop only, own lockfile, ADR-6).

In M1, scripts/echo_practice_run.py is split into build_model.py, export_onnx.py,
parity.py, quantize.py and make_model_card.py (blueprint section 5), and the
`echo-train` command gets its real subcommands.
"""

__version__ = "0.1.0"


def main() -> None:
    print("echo-train subcommands (build-random, export, parity, quantize, card) arrive in M1.")
    print("For now: uv run python scripts/echo_practice_run.py")
