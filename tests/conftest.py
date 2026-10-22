import gzip
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from arafno.artifacts import sha256
from arafno.model import AFNO2D_OneStep


@pytest.fixture
def case(tmp_path):
    """A tiny asymmetric region inside full-size, real-format GSMaP binaries."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = AFNO2D_OneStep((4, 6), 2, embed_dim=4, depth=1, patch_size=2)
    torch.save(model.state_dict(), artifacts / "model.pth")
    np.save(artifacts / "mean.npy", np.float32(0.2))
    np.save(artifacts / "std.npy", np.float32(0.5))
    files = {
        role: {"name": name, "sha256": sha256(artifacts / name)}
        for role, name in (("checkpoint", "model.pth"), ("mean", "mean.npy"), ("std", "std.npy"))
    }
    (artifacts / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": "test",
                "files": files,
                "training_latitude_order": None,
                "reference_input_product": "gsmap_nrt",
            }
        )
    )
    config = {
        "artifacts": "artifacts",
        "domain": {"west": -120, "south": 32.6, "east": -119.4, "north": 33},
        "model_latitude_order": "south_to_north",
        "product": "gsmap_now",
        "remote_directory": "/now/latest",
        "missing_policy": "error",
        "max_missing_fraction": 0.05,
        "ftp": {
            "timeout_seconds": 1,
            "attempts": 3,
            "retry_delay_seconds": 0.01,
            "max_retry_delay_seconds": 0.02,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    run_time = datetime(2026, 1, 2, 0, tzinfo=UTC)
    paths = []
    arrays = []
    for step in range(2):
        timestamp = run_time - timedelta(hours=1 - step)
        # Marker is not symmetric under latitude flip or longitude shift.
        rain = np.arange(24, dtype=np.float32).reshape(4, 6) / 10 + step
        global_data = np.zeros((1200, 3600), dtype="<f4")
        global_data[270:274, 2400:2406] = rain
        path = inputs / f"gsmap_now.{timestamp:%Y%m%d.%H00}.dat.gz"
        with gzip.open(path, "wb", compresslevel=1) as stream:
            stream.write(global_data.tobytes())
        paths.append(path)
        arrays.append(rain)
    return {
        "root": tmp_path,
        "config": config_path,
        "payload": config,
        "model": model,
        "paths": paths,
        "rain": np.stack(arrays),
        "run_time": run_time,
    }
