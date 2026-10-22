"""Original one-step AFNO architecture, with strict checkpoint reconstruction."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from torch import nn


class AFNOBlock(nn.Module):
    def __init__(self, dim: int, h: int, w: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.complex_weight = nn.Parameter(
            torch.view_as_complex(torch.randn(h, w // 2 + 1, dim, 2) * 0.02)
        )
        self.freq_mixer = nn.Parameter(torch.randn(h, w // 2 + 1, 1) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        spectrum = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        spectrum = spectrum * self.complex_weight * torch.sigmoid(self.freq_mixer)
        x = torch.fft.irfft2(spectrum, s=x.shape[1:3], dim=(1, 2), norm="ortho")
        x = x + residual
        return x + self.mlp(self.norm2(x))


class AFNO2D_OneStep(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim, depth, patch_size):
        super().__init__()
        if min(*img_size, in_chans, embed_dim, depth, patch_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        if any(size % patch_size for size in img_size):
            raise ValueError("Image dimensions must be divisible by patch_size")
        self.img_size = tuple(img_size)
        self.input_steps = in_chans
        self.ph, self.pw = (size // patch_size for size in img_size)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.ph, self.pw, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([AFNOBlock(embed_dim, self.ph, self.pw) for _ in range(depth)])
        self.head = nn.ConvTranspose2d(embed_dim, 1, patch_size, stride=patch_size)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_steps, *self.img_size):
            raise ValueError(f"Expected [batch, {self.input_steps}, {self.img_size}]")
        x = self.patch_embed(x).permute(0, 2, 3, 1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(x.permute(0, 3, 1, 2))


def load_model(path: Path, device: torch.device, expected_shape: tuple[int, int]):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("model_state_dict must be a mapping")
    for prefix in ("_orig_mod.", "module."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items()}
    try:
        embed_dim, input_steps, patch_h, patch_w = state["patch_embed.weight"].shape
        batch, ph, pw, pos_dim = state["pos_embed"].shape
        indices = {int(m[1]) for k in state if (m := re.match(r"blocks\.(\d+)\.", k))}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Checkpoint is not a supported ARAFNO model") from exc
    if patch_h != patch_w or batch != 1 or pos_dim != embed_dim:
        raise ValueError("Inconsistent checkpoint architecture")
    if not indices or indices != set(range(len(indices))):
        raise ValueError("Checkpoint block indices must be contiguous")
    shape = (ph * patch_h, pw * patch_w)
    if shape != expected_shape:
        raise ValueError(f"Checkpoint grid {shape} differs from domain {expected_shape}")
    model = AFNO2D_OneStep(shape, input_steps, embed_dim, len(indices), patch_h)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def forecast_steps(model: nn.Module, inputs: torch.Tensor, steps: int) -> Iterator[torch.Tensor]:
    """Yield normalized one-hour leads without retaining the entire rollout on the GPU."""
    if steps < 1:
        raise ValueError("Forecast horizon must be positive")
    model.eval()
    current = inputs
    for _ in range(steps):
        with torch.inference_mode():
            prediction = model(current)
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced non-finite predictions")
            current = torch.cat((current[:, 1:], prediction), dim=1)
        yield prediction
