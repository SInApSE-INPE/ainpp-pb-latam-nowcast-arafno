# ============================================================
# AUTOREGRESSIVE INFERENCE + PLOT (ALL LEADS)  |  OBS vs PRED
# Assumptions:
# - You trained the autoregressive AFNO one-step (out=1)
# - Dataset is the Zarr with groups train/validation/test
# - You want to pass an index and plot all target_len lead times
# ============================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from typing import Literal, Union, Optional, Tuple



SplitName = Literal["train", "validation", "test"]
InputSource = Literal["mvk", "nrt"]

# ----------------------------
# Dataset (same as training)
# ----------------------------
class GSMaPZarrSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path: Union[str, Path],
        split: SplitName,
        input_len: int,
        target_len: int,
        stride: int = 1,
        input_source: InputSource = "nrt",
        target_var: str = "mvk",
        consolidated: bool = True,
    ):
        super().__init__()
        self.zarr_path = str(Path(zarr_path))
        self.split = split
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.stride = int(stride)
        self.input_source = input_source
        self.target_var = target_var

        self.ds = xr.open_zarr(self.zarr_path, group=self.split, consolidated=consolidated)

        for dim in ("time", "lat", "lon"):
            if dim not in self.ds.dims:
                raise ValueError(f"Dimensão '{dim}' não encontrada no split '{self.split}'.")

        self.time_len = int(self.ds.sizes["time"])
        self.h = int(self.ds.sizes["lat"])
        self.w = int(self.ds.sizes["lon"])

        total = self.input_len + self.target_len
        max_start = self.time_len - total
        if max_start < 0:
            raise ValueError(f"Split '{self.split}' tem {self.time_len} timesteps; precisa de {total}.")

        self.starts = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        x = self.ds[f"gsmap_{self.input_source}"].isel(time=slice(start, start + self.input_len)).values
        y = self.ds[f"gsmap_{self.target_var}"].isel(
            time=slice(start + self.input_len, start + self.input_len + self.target_len)
        ).values

        x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0)
        y = np.nan_to_num(y.astype(np.float32, copy=False), nan=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)

# ----------------------------
# Model (autoregressive one-step AFNO)
# ----------------------------
import torch.nn as nn
import torch.nn.functional as F

class AFNOBlock(nn.Module):
    def __init__(self, dim: int, h: int, w: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.complex_weight = nn.Parameter(
            torch.view_as_complex(torch.randn(h, w // 2 + 1, dim, 2) * 0.02)
        )
        self.freq_mixer = nn.Parameter(torch.randn(h, w // 2 + 1, 1) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_fft = x_fft * self.complex_weight * torch.sigmoid(self.freq_mixer)
        x = torch.fft.irfft2(x_fft, s=x.shape[1:3], dim=(1, 2), norm="ortho")
        x = x + res
        x = x + self.mlp(self.norm2(x))
        return x

class AFNO2D_OneStep(nn.Module):
    """
    input:  [B, Tin, H, W]
    output: [B, 1,  H, W]
    """
    def __init__(
        self,
        img_size: Tuple[int, int],
        in_chans: int,
        embed_dim: int = 256,
        depth: int = 8,
        patch_size: int = 10,
    ):
        super().__init__()
        H, W = img_size
        self.ph = H // patch_size
        self.pw = W // patch_size

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.ph, self.pw, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([AFNOBlock(embed_dim, self.ph, self.pw) for _ in range(depth)])
        self.head = nn.ConvTranspose2d(
            embed_dim, 1,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.patch_embed(x)          # [B,E,ph,pw]
        x = x.permute(0, 2, 3, 1)        # [B,ph,pw,E]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)        # [B,E,ph,pw]
        x = self.head(x)                 # [B,1,H,W]
        return x

# ----------------------------
# Optional: denormalization (log1p + zscore inverse)
# ----------------------------
def inverse_transform(tensor_norm: np.ndarray, mean_log: float, std_log: float) -> np.ndarray:
    x_log = tensor_norm * std_log + mean_log
    x_mmh = np.expm1(x_log)
    return np.maximum(x_mmh, 0.0)

# ----------------------------
# Autoregressive rollout (inference)
# ----------------------------
@torch.no_grad()
def autoregressive_rollout(model: nn.Module, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    x0: [B, Tin, H, W]
    returns: [B, steps, H, W]
    """
    model.eval()
    preds = []
    cur = x0
    for _ in range(steps):
        y_hat = model(cur)           # [B,1,H,W]
        preds.append(y_hat)
        cur = torch.cat([cur[:, 1:], y_hat], dim=1)  # shift window
    return torch.cat(preds, dim=1)

# ----------------------------
# Main: infer by index and plot all lead times
# ----------------------------
@torch.no_grad()
def infer_and_plot_all_leads_autoreg(
    zarr_path: str,
    checkpoint_path: str,
    index: int,
    split: SplitName = "test",
    input_source: InputSource = "nrt",
    target_var: str = "mvk",
    input_len: int = 12,
    target_len: int = 6,
    stride: int = 1,
    device: Optional[str] = None,
    # model hyperparams MUST match training
    embed_dim: int = 256,
    depth: int = 8,
    patch_size: int = 10,
    # denorm
    denorm: bool = False,
    mean_path: Optional[str] = None,
    std_path: Optional[str] = None,
    # plotting
    vmax_clip: Optional[float] = None,   # e.g. 10 for mm/h
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ds = GSMaPZarrSequenceDataset(
        zarr_path=zarr_path,
        split=split,
        input_len=input_len,
        target_len=target_len,
        stride=stride,
        input_source=input_source,
        target_var=target_var,
        consolidated=True,
    )

    if index < 0 or index >= len(ds):
        raise IndexError(f"index={index} fora do range [0, {len(ds)-1}]")

    # model
    model = AFNO2D_OneStep(
        img_size=(ds.h, ds.w),
        in_chans=input_len,
        embed_dim=embed_dim,
        depth=depth,
        patch_size=patch_size,
    ).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # accept either state_dict directly or wrapped
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # data
    x, y_true = ds[index]                         # x: [Tin,H,W], y: [Tout,H,W]
    x_b = x.unsqueeze(0).to(device)               # [1,Tin,H,W]

    y_pred = autoregressive_rollout(model, x_b, steps=target_len)  # [1,Tout,H,W]
    y_pred = y_pred.squeeze(0).float().cpu().numpy()               # [Tout,H,W]
    y_true = y_true.float().cpu().numpy()                           # [Tout,H,W]

    # timestamps for titles
    start = int(ds.starts[index])
    times = ds.ds["time"].values
    y_times = times[start + input_len : start + input_len + target_len]

    # denorm
    unit = "norm"
    if denorm:
        if mean_path is None or std_path is None:
            raise ValueError("Para denorm=True, forneça mean_path e std_path")
        mean_log = float(np.load(mean_path).reshape(()))
        std_log = float(np.load(std_path).reshape(()))

        y_pred = np.stack([inverse_transform(y_pred[i], mean_log, std_log) for i in range(target_len)], axis=0)
        y_true = np.stack([inverse_transform(y_true[i], mean_log, std_log) for i in range(target_len)], axis=0)
        unit = "mm/h"

    # common color scale
    vmin = float(np.nanmin([y_true.min(), y_pred.min()]))
    vmax = float(np.nanmax([y_true.max(), y_pred.max()]))

    if vmax_clip is not None:
        vmax = float(vmax_clip)
        vmin = 0.0 if denorm else vmin

    # plot grid: 2 rows (OBS/PRED) x target_len cols
    fig, axes = plt.subplots(2, target_len, figsize=(4 * target_len, 7), constrained_layout=True)

    last_im = None
    for i in range(target_len):
        im0 = axes[0, i].imshow(y_true[i], vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"OBS | +{i+1}h\n{str(y_times[i])}")
        axes[0, i].axis("off")

        im1 = axes[1, i].imshow(y_pred[i], vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"PRED | +{i+1}h")
        axes[1, i].axis("off")

        last_im = im1

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(unit)
    plt.show()

    return {
        "index": index,
        "split": split,
        "input_source": input_source,
        "target_var": target_var,
        "input_len": input_len,
        "target_len": target_len,
        "y_times": [str(t) for t in y_times],
        "unit": unit,
        "vmin": vmin,
        "vmax": vmax,
        "y_true": y_true,
        "y_pred": y_pred,
    }