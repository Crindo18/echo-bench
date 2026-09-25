"""Decoding `vcgencmd get_throttled` (blueprint sections 6.2 and 6.7)."""

from echo_bench.env import rpi


def test_parse_throttled() -> None:
    assert rpi.parse_throttled("throttled=0x0\n") == 0
    assert rpi.parse_throttled("throttled=0x50000") == 0x50000


def test_no_flags_when_zero() -> None:
    assert rpi.decode_throttled(0) == []


def test_classic_under_voltage_history() -> None:
    # 0x50000 = bits 16 and 18: under-voltage and throttling happened since boot, not right now
    assert rpi.decode_throttled(0x50000) == [
        "under-voltage has occurred",
        "throttling has occurred",
    ]
    assert rpi.has_any_bit(0x50000, rpi.UNDER_VOLTAGE_BITS)
    assert rpi.has_any_bit(0x50000, rpi.THROTTLING_BITS)


def test_current_flags() -> None:
    assert rpi.decode_throttled(0x5) == ["under-voltage now", "throttled now"]
    assert not rpi.has_any_bit(0x8, rpi.UNDER_VOLTAGE_BITS)  # soft temperature limit only
