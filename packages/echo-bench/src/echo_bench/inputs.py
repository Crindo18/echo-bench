"""Synthetic model inputs built from a model card's `io` block (blueprint section 6.1).

Encoder cost depends only on input shape, so random features of the right shape are
valid for size, speed and compute measurements. Any model whose inputs are
features [B, T, F] plus lengths [B] works: ECHO's (feats, feats_lens) and, for
example, an offline Zipformer's (x, x_lens).
"""

import numpy as np

from echo_core.model_card import ModelCard

FRAMES_PER_SECOND = 100  # 10 ms hop


def synthetic_inputs(card: ModelCard, seconds: float, seed: int = 0) -> dict[str, np.ndarray]:
    frames = int(round(seconds * FRAMES_PER_SECOND))
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for name, shape in card.io.inputs.items():
        if len(shape) == 3 and isinstance(shape[-1], int):
            inputs[name] = rng.standard_normal((1, frames, shape[-1]), dtype=np.float32)
        elif len(shape) == 1:
            inputs[name] = np.array([frames], dtype=np.int64)
        else:
            raise ValueError(
                f"{card.ref}: input {name} {shape} is neither features [B, T, F] nor lengths [B]"
            )
    return inputs
