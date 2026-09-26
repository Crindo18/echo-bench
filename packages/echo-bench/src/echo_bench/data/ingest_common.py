"""Shared pieces for the corpus ingesters."""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

Progress = Callable[[str], None]


class IngestError(Exception):
    """The corpus on disk doesn't match what the ingester expects; nothing was written."""


@dataclass
class IngestReport:
    counts: Counter[str] = field(default_factory=Counter)  # e.g. 'used', 'skipped: ...'
    notes: list[str] = field(default_factory=list)


_PUNCTUATION = re.compile(r"[^\w\s']")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation except apostrophes, collapse spaces: 'Yes.' -> 'yes'."""
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())
