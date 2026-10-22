"""
Optimized inference + Zarr v2 export for downstream verification (CT metrics / FSS).

Output layout:
  OUTPUT_ROOT/<MODEL_NAME>/LEAD_XXXh.zarr
  with variables:
    precip_fct(time, lat, lon)  [mm/h]
    precip_obs(time, lat, lon)  [mm/h]

Notes:
- Writes one Zarr per lead-time to keep downstream I/O efficient.
- Uses Zarr v2 + numcodecs.Blosc (stable with xarray).
- For xarray: encoding is only provided on creation (mode="w"), not on append (mode="a").
"""

from __future__ import annotations

import os
import shutil
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from numcodecs import Blosc
from tqdm.auto import tqdm

from model import AFNO2D_OneStep, autoregressive_rollout


# ==============================================================================
# Types
# ==============================================================================
SplitName = Literal["train", "validation", "test"]
InputSource = Literal["mvk", "nrt"]
VarName = Literal["mvk", "nrt"]


# ==============================================================================
# Config
# ==============================================================================
@dataclass(frozen=True)
class DomainConfig:
    nlat: int = 880
    nlon: int = 970
    lat_min: float = -55.0
    lat_max: float = 33.0
    lon_min: float = -120.0
    lon_max: float = -23.0

    @property
    def lats(self) -> np.ndarray:
        return np.linspace(self.lat_min, self.lat_max, self.nlat, dtype=np.float32)

    @property
    def lons(self) -> np.ndarray:
        return np.linspace(self.lon_min, self.lon_max, self.nlon, dtype=np.float32)


@dataclass(frozen=True)
class SequenceConfig:
    input_steps: int = 6
    output_steps: int = 6


@dataclass(frozen=True)
class IOConfig:
    model_name: str = "AFNO"
    model_ckpt: Path = Path(
        "/prj/ideeps/adriano.almeida/laboratory/ConvLSTM/REFACT_AFNO/checkpoints_ar/best_model.pth"
    )
    input_zarr: Path = Path(
        "/prj/ideeps/adriano.almeida/data/ainpp/legacy/gsmap_nrt+mvk_log_zscore_ainpp-latin-america.zarr"
    )
    params_dir: Path = Path("/prj/ideeps/adriano.almeida/data/ainpp/legacy/model_params")
    output_root: Path = Path(
        "/prj/ideeps/adriano.almeida/laboratory/ConvLSTM/REFACT_AFNO/results/forecasts/"
        f"{model_name.lower()}_gsmap_mvk_latin_america"
    )


@dataclass(frozen=True)
class RuntimeConfig:
    split: SplitName = "test"
    input_source: InputSource = "nrt"
    target_var: VarName = "mvk"
    # Typical: non-overlapping windows for export
    # Set stride=1 if you want overlapped export (more samples, much more I/O)
    stride: int = 18
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True

    # Zarr write tuning
    write_chunks: Dict[str, int] = None  # set in __post_init__-like below

    def __post_init__(self):
        # dataclass(frozen=True) doesn't allow mutation; we set default via a helper below
        pass


def default_write_chunks() -> Dict[str, int]:
    # Coherent for LATAM 880x970; tune if needed
    return {"time": 48, "lat": 220, "lon": 242}


# ==============================================================================
# Small utilities
# ==============================================================================
def utc_now_string() -> str:
    # timezone-aware UTC (Python 3.12+)
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_zarr_v2_compressor() -> Blosc:
    # Zarr v2 compressor via numcodecs (stable with xarray)
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)


def inverse_log_zscore_to_mmh(x_norm: np.ndarray, mean_log: float, std_log: float) -> np.ndarray:
    """
    Inverse transform from Log-Zscore to mm/h:
      x_log = x_norm * std + mean
      x_mmh = expm1(x_log)
    """
    x_log = x_norm * std_log + mean_log
    x_mmh = np.expm1(x_log)
    return np.maximum(x_mmh, 0.0).astype(np.float32)


# ==============================================================================
# Dataset
# ==============================================================================
class GSMaPSequenceDataset(torch.utils.data.Dataset):
    """
    Windowed dataset from a Zarr group (train/validation/test):
      x: (T_in, lat, lon)
      y: (T_out, lat, lon)
    """

    def __init__(
        self,
        zarr_path: Union[str, Path],
        split: SplitName,
        input_steps: int,
        output_steps: int,
        stride: int,
        input_source: InputSource,
        target_var: VarName,
        consolidated: bool = True,
    ) -> None:
        super().__init__()
        self.zarr_path = str(Path(zarr_path))
        self.split = split
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.stride = int(stride)
        self.input_source = input_source
        self.target_var = target_var
        self.consolidated = consolidated

        self.ds = xr.open_zarr(self.zarr_path, group=self.split, consolidated=self.consolidated)

        for dim in ("time", "lat", "lon"):
            if dim not in self.ds.dims:
                raise ValueError(f"Missing dimension '{dim}' in split '{self.split}'.")

        required_vars = [f"gsmap_{self.input_source}", f"gsmap_{self.target_var}"]
        missing = [v for v in required_vars if v not in self.ds.data_vars]
        if missing:
            raise ValueError(f"Missing variables: {missing}. Available: {list(self.ds.data_vars)}")

        time_len = int(self.ds.sizes["time"])
        total = self.input_steps + self.output_steps
        max_start = time_len - total
        if max_start < 0:
            raise ValueError(f"Split '{self.split}' has {time_len} timesteps; needs at least {total}.")

        self.starts = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])

        x = (
            self.ds[f"gsmap_{self.input_source}"]
            .isel(time=slice(start, start + self.input_steps))
            .values.astype(np.float32, copy=False)
        )
        y = (
            self.ds[f"gsmap_{self.target_var}"]
            .isel(time=slice(start + self.input_steps, start + self.input_steps + self.output_steps))
            .values.astype(np.float32, copy=False)
        )

        return torch.from_numpy(x), torch.from_numpy(y)


# ==============================================================================
# Zarr v2 writer
# ==============================================================================
def store_is_xarray_zarr_with_time(zarr_path: Path) -> bool:
    """
    True if `xr.open_zarr` works and the dataset has a 'time' dimension.
    This helps detect broken stores or Zarr v3 remnants (e.g., zarr.json).
    """
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        ok = ("time" in ds.dims)
        ds.close()
        return ok
    except Exception:
        return False


def append_batch_to_lead_zarr_v2(
    zarr_path: Path,
    valid_times: np.ndarray,           # (B,) datetime64
    precip_fct_mmh: np.ndarray,        # (B, lat, lon) float32
    precip_obs_mmh: np.ndarray,        # (B, lat, lon) float32
    lat: np.ndarray,
    lon: np.ndarray,
    dataset_attrs: Dict[str, str],
    chunks: Dict[str, int],
    compressor: Blosc,
) -> None:
    """
    Append to a per-lead Zarr v2 store.

    xarray rule:
      - on CREATE (mode="w"): provide encoding (compressor/dtype)
      - on APPEND (mode="a"): do not provide encoding for existing variables
    """
    ds_batch = xr.Dataset(
        data_vars={
            "precip_fct": (("time", "lat", "lon"), precip_fct_mmh),
            "precip_obs": (("time", "lat", "lon"), precip_obs_mmh),
        },
        coords={
            "time": valid_times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs=dict(dataset_attrs),
    ).chunk(chunks)

    encoding = {
        "precip_fct": {"compressor": compressor, "dtype": "float32"},
        "precip_obs": {"compressor": compressor, "dtype": "float32"},
    }

    ensure_dir(zarr_path.parent)

    # If the path exists but is not a valid xarray-openable Zarr store with time, recreate it.
    if zarr_path.exists() and (not store_is_xarray_zarr_with_time(zarr_path)):
        print(f"[WARN] Invalid/incompatible store detected. Recreating: {zarr_path}")
        remove_tree(zarr_path)

    if not zarr_path.exists():
        ds_batch.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
            encoding=encoding,
            zarr_format=2,
        )
    else:
        ds_batch.to_zarr(
            zarr_path,
            mode="a",
            append_dim="time",
            consolidated=True,
            zarr_format=2,
        )

    ds_batch.close()


def finalize_lead_store(zarr_path: Path) -> None:
    """
    Optional post-pass:
    - sort by time
    - drop duplicate times
    - rewrite as a clean Zarr v2 store (single pass)
    """
    if not zarr_path.exists():
        return

    ds = xr.open_zarr(zarr_path, consolidated=True)

    if "time" in ds.coords:
        ds = ds.sortby("time")
        _, idx = np.unique(ds["time"].values, return_index=True)
        if idx.size != ds.sizes["time"]:
            ds = ds.isel(time=np.sort(idx))

    tmp = zarr_path.with_suffix(".zarr.tmp")
    if tmp.exists():
        remove_tree(tmp)

    ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=2)
    ds.close()

    remove_tree(zarr_path)
    shutil.move(tmp, zarr_path)


# ==============================================================================
# Model loading
# ==============================================================================
def load_forecaster(checkpoint_path: Path, device: torch.device, seq: SequenceConfig) -> torch.nn.Module:
    model = AFNO2D_OneStep(
        img_size=(880, 970),
        in_chans=6,
        embed_dim=256,
        depth=8,
        patch_size=10,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    try:
        model = torch.compile(model)
    except Exception:
        pass

    return model


# ==============================================================================
# Time helpers
# ==============================================================================
def compute_run_time(ds: xr.Dataset, start_idx: int, input_steps: int) -> np.datetime64:
    # run_time = last timestep of the input window
    return np.datetime64(ds["time"].isel(time=start_idx + input_steps - 1).values)


def compute_valid_time(ds: xr.Dataset, start_idx: int, input_steps: int, lead_step: int) -> np.datetime64:
    # valid_time = timestep at target lead
    return np.datetime64(ds["time"].isel(time=start_idx + input_steps + lead_step).values)


def lead_hours(run_time: np.datetime64, valid_time: np.datetime64) -> float:
    return float((valid_time - run_time) / np.timedelta64(1, "h"))


def lead_tag_from_hours(hours: float) -> str:
    return f"{int(round(hours)):03d}h"


# ==============================================================================
# Main pipeline
# ==============================================================================
def run_inference_and_export_to_zarr() -> None:
    domain = DomainConfig()
    seq = SequenceConfig()
    io = IOConfig()

    # You can override this if needed; (non-overlapping)
    runtime = RuntimeConfig(stride=seq.input_steps + seq.output_steps)
    write_chunks = default_write_chunks()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Normalization params
    mean_log = np.float32(np.load(io.params_dir / "gsmap_nrt+mvk_log_mean_ainpp-latin-america.npy"))
    std_log = np.float32(np.load(io.params_dir / "gsmap_nrt+mvk_log_std_ainpp-latin-america.npy"))

    # Model
    print(f"[INFO] Loading {io.model_name} forecaster...")
    model = load_forecaster(io.model_ckpt, device=device, seq=seq)

    # Dataset
    print("[INFO] Opening base dataset...")
    dataset = GSMaPSequenceDataset(
        zarr_path=io.input_zarr,
        split=runtime.split,
        input_steps=seq.input_steps,
        output_steps=seq.output_steps,
        stride=runtime.stride,
        input_source=runtime.input_source,
        target_var=runtime.target_var,
        consolidated=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=runtime.batch_size,
        shuffle=False,
        num_workers=runtime.num_workers,
        pin_memory=runtime.pin_memory,
        drop_last=False,
    )

    print(f"[OK] Dataset ready: {len(dataset)} samples.")

    compressor = build_zarr_v2_compressor()

    base_attrs = {
        "title": f"{io.model_name} precipitation forecast with colocated observation",
        "Conventions": "CF-1.8",
        "institution": "INPE / AINPP",
        "source": io.model_name,
        "split": runtime.split,
        "domain": "LATAM 880x970 (0.1 degree)",
        "units": "mm hr-1",
        "history": f"generated on {utc_now_string()}",
    }

    pbar = tqdm(dataloader, total=len(dataloader), desc="Batches", unit="batch", dynamic_ncols=True)

    for batch_idx, (x_in, y_true) in enumerate(pbar):
        x_in = x_in.to(device, non_blocking=True)

        y_pred = autoregressive_rollout(model, x_in, steps=seq.output_steps)
        print("--->>>>", y_pred.shape)

        # Expected: [B, T_out, 1, H, W] or [B, T_out, H, W]
        if y_pred.ndim == 5:
            y_pred = y_pred.squeeze(2)

        y_pred_np = y_pred.detach().cpu().numpy().astype(np.float32, copy=False)
        y_true_np = y_true.numpy().astype(np.float32, copy=False)

        batch_size = y_pred_np.shape[0]

        # Append per lead
        for lead_step in range(seq.output_steps):
            valid_times: List[np.datetime64] = []
            run_times: List[np.datetime64] = []

            # Determine effective batch (last batch may be smaller)
            for i in range(batch_size):
                global_idx = batch_idx * dataloader.batch_size + i
                if global_idx >= len(dataset):
                    break

                start_idx = int(dataset.starts[global_idx])
                rt = compute_run_time(dataset.ds, start_idx, seq.input_steps)
                vt = compute_valid_time(dataset.ds, start_idx, seq.input_steps, lead_step)

                run_times.append(rt)
                valid_times.append(vt)

            b_eff = len(valid_times)
            if b_eff == 0:
                continue

            valid_times_arr = np.asarray(valid_times, dtype="datetime64[ns]")

            fct_mmh = inverse_log_zscore_to_mmh(y_pred_np[:b_eff, lead_step], mean_log, std_log)
            obs_mmh = inverse_log_zscore_to_mmh(y_true_np[:b_eff, lead_step], mean_log, std_log)

            # Lead hours should be constant for the lead store
            lh = lead_hours(run_times[0], valid_times_arr[0])
            lead_tag = lead_tag_from_hours(lh)

            lead_store = io.output_root / io.model_name / f"LEAD_{lead_tag}.zarr"

            attrs = dict(base_attrs)
            attrs["lead_time_hours"] = str(int(round(lh)))

            append_batch_to_lead_zarr_v2(
                zarr_path=lead_store,
                valid_times=valid_times_arr,
                precip_fct_mmh=fct_mmh,
                precip_obs_mmh=obs_mmh,
                lat=domain.lats,
                lon=domain.lons,
                dataset_attrs=attrs,
                chunks=write_chunks,
                compressor=compressor,
            )

    # Finalize all lead stores (optional but recommended)
    print("[INFO] Finalizing lead stores (sort + dedup)...")
    lead_root = io.output_root / io.model_name
    for zarr_path in sorted(lead_root.glob("LEAD_*.zarr")):
        finalize_lead_store(zarr_path)
        print(f"[OK] Finalized: {zarr_path}")

    print("[DONE] All finished.")


# ==============================================================================
# Entrypoint
# ==============================================================================
if __name__ == "__main__":
    # Thread caps to reduce CPU contention on multi-GPU nodes (tune as needed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    run_inference_and_export_to_zarr()
