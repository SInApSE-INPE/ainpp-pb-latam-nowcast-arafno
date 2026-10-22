import gzip
import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import rasterio
import torch

from arafno.acquisition import FTPSource, catalog, parse_time, select_window
from arafno.artifacts import verify_bundle
from arafno.config import Config
from arafno.entrypoint import main, parser, run
from arafno.export import NODATA, write_geotiff
from arafno.model import forecast_steps, load_model
from arafno.processing import Normalization, prepare_inputs, read_gsmap


@pytest.mark.parametrize("field", ["product", "north"])
def test_incomplete_config_has_no_silent_operational_defaults(case, field):
    payload = case["payload"]
    container = payload["domain"] if field == "north" else payload
    del container[field]
    case["config"].write_text(json.dumps(payload))
    with pytest.raises(TypeError, match=field):
        Config.load(case["config"])


def test_time_window_crosses_midnight_and_rejects_gaps(case):
    records = catalog([str(p) for p in reversed(case["paths"])], "gsmap_now")
    records = select_window(records, 2)
    assert records[-1].time == case["run_time"]
    assert records[0].time == case["run_time"] - timedelta(hours=1)
    with pytest.raises(ValueError, match="Incomplete"):
        select_window(records, 3)
    with pytest.raises(ValueError, match="duplicate"):
        select_window(records + records, 2)
    assert (
        catalog(["gsmap_now.20260101.2330.dat.gz", "gsmap_nrt.20260101.2300.dat.gz"], "gsmap_now")
        == []
    )


def test_utc_parsing():
    assert parse_time("2026-01-01T21:00:00-03:00").isoformat() == "2026-01-02T00:00:00+00:00"
    for value in ("2026-01-01T21:00:00", "2026-01-01T21:30:00Z"):
        with pytest.raises(ValueError):
            parse_time(value)


def test_now_interval_filenames():
    observations = catalog(["gsmap_now.20260101.2300_0000.dat.gz"], "gsmap_now")
    assert observations[0].time == parse_time("2026-01-01T23:00:00Z")
    assert catalog(["gsmap_now.20260101.2330_0030.dat.gz"], "gsmap_now") == []
    with pytest.raises(ValueError, match="one-hour"):
        catalog(["gsmap_now.20260101.2300_0100.dat.gz"], "gsmap_now")


def test_crop_normalization_and_latitude_orientation(case):
    config = Config.load(case["config"])
    norm = Normalization(0.2, 0.5)
    north = read_gsmap(case["paths"][0], config.domain)
    np.testing.assert_allclose(north, case["rain"][0])
    inputs, mask, fractions = prepare_inputs(case["paths"], config, norm)
    np.testing.assert_allclose(norm.inverse(inputs)[:, ::-1, :], case["rain"], atol=1e-6)
    assert inputs.flags.c_contiguous
    assert not mask.any()
    assert fractions == [0, 0]


@pytest.mark.parametrize("missing_value", [np.nan, np.inf, -np.inf])
def test_missing_policy_and_mask(case, missing_value):
    config = Config.load(case["config"])
    path = case["paths"][0]
    with gzip.open(path, "rb") as stream:
        data = np.frombuffer(stream.read(), dtype="<f4").copy().reshape(1200, 3600)
    data[270, 2400] = missing_value
    with gzip.open(path, "wb", compresslevel=1) as stream:
        stream.write(data.tobytes())
    norm = Normalization(0.2, 0.5)
    with pytest.raises(ValueError, match="Missing precipitation"):
        prepare_inputs(case["paths"], config, norm)
    filled, mask, _ = prepare_inputs(
        case["paths"], replace(config, missing_policy="normalized_zero"), norm
    )
    assert filled[0, -1, 0] == 0
    assert mask[0, 0] and mask.sum() == 1
    filled, _, _ = prepare_inputs(case["paths"], replace(config, missing_policy="zero_rain"), norm)
    assert filled[0, -1, 0] == pytest.approx(-0.4)
    with pytest.raises(ValueError, match="exceeds limit"):
        prepare_inputs(case["paths"], replace(config, max_missing_fraction=0.01), norm)


@pytest.mark.parametrize("policy", ["error", "normalized_zero", "zero_rain"])
def test_negative_rain_becomes_zero_before_normalization(case, policy):
    config = replace(Config.load(case["config"]), missing_policy=policy, max_missing_fraction=0)
    path = case["paths"][0]
    with gzip.open(path, "rb") as stream:
        data = np.frombuffer(stream.read(), dtype="<f4").copy().reshape(1200, 3600)
    data[270, 2400:2405] = [-99, -8, -4, -0.1, 0]
    with gzip.open(path, "wb", compresslevel=1) as stream:
        stream.write(data.tobytes())
    norm = Normalization(0.2, 0.5)
    inputs, mask, fractions = prepare_inputs(case["paths"], config, norm)
    # Zero rainfall is -mean/std in normalized space, not normalized zero.
    np.testing.assert_allclose(inputs[0, -1, :5], -0.4)
    expected = case["rain"].copy()
    expected[0, 0, :5] = 0
    np.testing.assert_allclose(norm.inverse(inputs)[:, ::-1, :], expected, atol=1e-6)
    assert not mask.any()
    assert fractions == [0, 0]


def test_invalid_binary_and_normalization(case):
    path = case["root"] / "bad.dat.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="size"):
        read_gsmap(path, Config.load(case["config"]).domain)
    for mean, std in [(0, 0), (float("nan"), 1), (0, -1)]:
        with pytest.raises(ValueError):
            Normalization(mean, std)
    with pytest.raises(ValueError, match="Non-finite"):
        Normalization(0, 1).inverse(np.array([1000], dtype=np.float32))
    with pytest.raises(ValueError):
        replace(Config.load(case["config"]).domain, west=-120.01)


def test_model_loading_and_autoregressive_feedback(case):
    path = case["root"] / "artifacts/model.pth"
    model = load_model(path, torch.device("cpu"), (4, 6))
    x = torch.randn(1, 2, 4, 6)
    with torch.inference_mode():
        first = case["model"](x)
        second = case["model"](torch.cat((x[:, 1:], first), dim=1))
    forecasts = list(forecast_steps(model, x, 2))
    torch.testing.assert_close(forecasts[0], first)
    torch.testing.assert_close(forecasts[1], second)
    with pytest.raises(ValueError, match="grid"):
        load_model(path, torch.device("cpu"), (6, 6))
    state = model.state_dict()
    del state["head.bias"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path, torch.device("cpu"), (4, 6))


def test_bundle_integrity(case):
    folder = case["root"] / "artifacts"
    verify_bundle(folder)
    (folder / "std.npy").write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(folder)


def test_geotiff_coordinates_and_nodata(case):
    domain = Config.load(case["config"]).domain
    path = case["root"] / "output.tif"
    mask = np.zeros(domain.shape, dtype=bool)
    mask[0, 0] = True
    write_geotiff(path, case["rain"][0], mask, domain, {"lead_hours": 1})
    with rasterio.open(path) as raster:
        assert raster.crs.to_epsg() == 4326
        assert tuple(raster.bounds) == pytest.approx((-120, 32.6, -119.4, 33))
        assert raster.xy(0, 0) == pytest.approx((-119.95, 32.95))
        assert raster.xy(3, 5) == pytest.approx((-119.45, 32.65))
        assert raster.nodata == NODATA
        assert raster.read(1, masked=True).mask[0, 0]
        np.testing.assert_allclose(raster.read(1)[1:], case["rain"][0, 1:])
        assert raster.units == ("mm/h",)
        assert raster.tags()["lead_hours"] == "1"


def test_end_to_end_and_no_overwrite(case):
    args = parser().parse_args(
        [
            "--config",
            str(case["config"]),
            "--input-dir",
            str(case["root"] / "inputs"),
            "--output-dir",
            str(case["root"] / "outputs"),
            "--device",
            "cpu",
            "--steps",
            "2",
        ]
    )
    target = run(args)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["run_time"] == case["run_time"].isoformat()
    assert len(manifest["outputs"]) == 2
    assert manifest["outputs"][0]["valid_time"] == "2026-01-02T01:00:00+00:00"
    norm = Normalization(0.2, 0.5)
    normalized = (np.log1p(case["rain"][:, ::-1, :]) - norm.mean) / norm.std
    with torch.inference_mode():
        expected = case["model"](torch.from_numpy(normalized.copy()).unsqueeze(0))
    expected = norm.inverse(expected[0, 0].numpy())[::-1, :]
    with rasterio.open(target / manifest["outputs"][0]["name"]) as raster:
        np.testing.assert_allclose(raster.read(1), expected, atol=1e-6)
    with pytest.raises(FileExistsError):
        run(args)


def test_failed_export_does_not_publish_partial_run(case, monkeypatch):
    from arafno._runtime import pipeline as entrypoint

    original = entrypoint.write_geotiff
    calls = []

    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("simulated disk failure")
        return original(*args)

    monkeypatch.setattr(entrypoint, "write_geotiff", fail_second)
    out = case["root"] / "outputs"
    status = main(
        [
            "--config",
            str(case["config"]),
            "--input-dir",
            str(case["root"] / "inputs"),
            "--output-dir",
            str(out),
            "--steps",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert status == 1
    assert not list(out.rglob("*.tif"))
    assert not list(out.rglob("manifest.json"))


def test_ftp_download_cache_and_corruption(case, monkeypatch):
    monkeypatch.setenv("FTP_JAXA_USER", "test")
    monkeypatch.setenv("FTP_JAXA_PASSWORD", "test")
    config = Config.load(case["config"])
    source = FTPSource(config.remote_directory, case["root"] / "cache", config.domain, config.ftp)
    record = catalog(["/now/latest/" + case["paths"][0].name], "gsmap_now")[0]
    downloads = []

    class FakeFTP:
        def retrbinary(self, command, callback):
            downloads.append(command)
            callback(case["paths"][0].read_bytes())

    monkeypatch.setattr(source, "_request", lambda fn: fn(FakeFTP()))
    path = source.download(record)
    assert source.download(record) == path
    assert len(downloads) == 1
    path.write_bytes(b"corrupt")
    assert source.download(record).read_bytes() == case["paths"][0].read_bytes()
    assert len(downloads) == 2
    assert not list(path.parent.glob("*.part"))


def test_ftp_archive_window_spans_days(case, monkeypatch):
    monkeypatch.setenv("FTP_JAXA_USER", "test")
    monkeypatch.setenv("FTP_JAXA_PASSWORD", "test")
    config = Config.load(case["config"])
    source = FTPSource("/archive/%Y/%m/%d", case["root"] / "cache", config.domain, config.ftp)
    directories = []

    class FakeFTP:
        def cwd(self, directory):
            self.directory = directory
            directories.append(directory)

        def nlst(self):
            index = 0 if self.directory.endswith("01/01") else 1
            return [case["paths"][index].name]

    monkeypatch.setattr(source, "_request", lambda fn: fn(FakeFTP()))
    observations = source.window("gsmap_now", 2, case["run_time"])
    assert len(observations) == 2
    assert directories == ["/archive/2026/01/01", "/archive/2026/01/02"]
