"""The embedding network: E-Branchformer encoder + masked mean pooling (blueprint section 5)."""

import contextlib
import io

import torch

from echo_train.config import ModelConfig

# ESPnet prints "Failed to import Flash Attention" on import. Flash Attention is a
# GPU-only speed-up that ECHO doesn't use, so the message is hidden.
with contextlib.redirect_stdout(io.StringIO()):
    from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder

ENCODER_IMPL = "espnet2.asr.encoder.e_branchformer_encoder.EBranchformerEncoder"


class EmbeddingNet(torch.nn.Module):
    """Encoder + masked mean pooling: one embedding (a list of numbers) per utterance."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats: torch.Tensor, feats_lens: torch.Tensor) -> torch.Tensor:
        hs, hlens, _ = self.encoder(feats, feats_lens)  # hs: [batch, frames, width]
        frame_idx = torch.arange(hs.size(1), device=hs.device)
        mask = (frame_idx[None, :] < hlens[:, None]).to(hs.dtype)  # 1 = real frame, 0 = padding
        summed = (hs * mask.unsqueeze(-1)).sum(dim=1)
        return summed / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


def build_model(config: ModelConfig) -> EmbeddingNet:
    """Build the network with random weights from config.seed, in inference mode."""
    torch.manual_seed(config.seed)
    encoder = EBranchformerEncoder(
        input_size=config.input_size, **config.encoder.model_dump(), use_flash_attn=False
    )
    return EmbeddingNet(encoder).eval()  # eval() switches off training-only behaviour (dropout)


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_parameters(model: EmbeddingNet) -> dict[str, int]:
    """Parameters per module, in model order, plus the total."""
    encoder = model.encoder
    counts = {"input subsampling": _count(encoder.embed)}
    for index, block in enumerate(encoder.encoders):
        counts[f"block {index}"] = _count(block)
    counts["everything else"] = _count(model) - sum(counts.values())
    counts["TOTAL"] = _count(model)
    return counts
