"""Fingerprints must be stable, and must change when anything they describe changes (principle P5)."""

import json

from echo_bench.env.fingerprint import (
    canonical_json,
    collect_hardware,
    collect_software,
    detect_platform,
    fingerprint_of,
)


def test_key_order_does_not_change_the_fingerprint() -> None:
    assert fingerprint_of({"a": 1, "b": [1, 2]}) == fingerprint_of({"b": [1, 2], "a": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_any_change_changes_the_fingerprint() -> None:
    assert fingerprint_of({"cooling": "active_cooler"}) != fingerprint_of({"cooling": "none"})


def test_platform_detection() -> None:
    assert detect_platform("Raspberry Pi 5 Model B Rev 1.0") == "rpi5"
    assert detect_platform("Raspberry Pi 4 Model B Rev 1.5") == "other"
    assert detect_platform(None) == "desktop"


def test_hardware_description_is_complete() -> None:
    hw = collect_hardware(cooling="active_cooler")
    assert hw["platform"] in {"desktop", "rpi5", "other"}
    assert hw["cpu_cores"] > 0 and hw["ram_total_mb"] > 0
    assert hw["cooling"] == "active_cooler"
    assert len(hw["fingerprint"]) == 64
    assert collect_hardware(cooling="none")["fingerprint"] != hw["fingerprint"]


def test_software_description_is_valid_json_where_needed() -> None:
    sw = collect_software()
    assert "CPUExecutionProvider" in json.loads(sw["ort_providers_json"])
    assert any(p.lower().startswith("onnxruntime==") for p in json.loads(sw["packages_json"]))
    assert sw["git_dirty"] in (0, 1)
