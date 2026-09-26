"""Embedding files from `echo-train embed` (blueprint ADR-4: one .npz per model and dataset)."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class EmbeddingSet:
    name: str
    path: Path
    embeddings: npt.NDArray[np.float32]
    speaker: npt.NDArray[np.str_]
    label: npt.NDArray[np.str_]
    group: npt.NDArray[np.str_]
    session: npt.NDArray[np.str_]
    metadata: dict[str, Any]

    def restricted(self, groups: set[str]) -> "EmbeddingSet":
        """Only the given recordings (to compare encoders on identical data)."""
        keep = np.isin(self.group, sorted(groups))
        return replace(
            self,
            embeddings=self.embeddings[keep],
            speaker=self.speaker[keep],
            label=self.label[keep],
            group=self.group[keep],
            session=self.session[keep],
        )


def load_embeddings(path: Path) -> EmbeddingSet:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        group = data["recording_group_id"]
        _, first = np.unique(group, return_index=True)  # one vector per physical recording
        keep = np.sort(first)
        source = metadata.get("model_source", path.stem).removeprefix("hf:")
        name = source + (
            " (random weights)" if metadata.get("weights_state") == "random_init" else ""
        )
        return EmbeddingSet(
            name=name,
            path=path,
            embeddings=data["embeddings"][keep].astype(np.float32),
            speaker=data["speaker"][keep],
            label=data["label_key"][keep],
            group=group[keep],
            session=data["session"][keep],
            metadata=metadata,
        )
