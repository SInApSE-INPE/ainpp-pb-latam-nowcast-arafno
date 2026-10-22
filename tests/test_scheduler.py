import argparse
import gzip
import os
from datetime import UTC, datetime, timedelta

import pytest

from arafno import acquisition
from arafno._runtime import scheduler
from arafno.acquisition import DataNotReady, FTPSource, Observation, parse_time
from arafno.config import Config
from arafno.entrypoint import BACKEND, parser, require_fresh, run


def poll_args(case):
    return parser().parse_args(
        [
            "--config",
            str(case["config"]),
            "--source",
            "ftp",
            "--scheduled",
            "--cache-dir",
            str(case["root"] / "cache"),
            "--output-dir",
            str(case["root"] / "outputs"),
            "--steps",
            "2",
            "--device",
            "cpu",
        ]
    )


def mock_listing(case, monkeypatch):
    monkeypatch.setenv("FTP_JAXA_USER", "test")
    monkeypatch.setenv("FTP_JAXA_PASSWORD", "test")
    monkeypatch.setattr(scheduler, "require_fresh", lambda *args: None)
    observations = [Observation(case["run_time"], "last.dat.gz", "/now/latest/last.dat.gz")]
    monkeypatch.setattr(FTPSource, "window", lambda *args, **kwargs: observations)


def test_completed_poll_skips_download_and_inference(case, monkeypatch):
    args = poll_args(case)
    local = argparse.Namespace(**vars(args))
    local.source = "local"
    local.input_dir = case["root"] / "inputs"
    target = run(local)
    stale = args.cache_dir / "0123456789abcdef" / "gsmap_now.20230101.0000.dat.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"expired")
    os.utime(stale, (0, 0))
    mock_listing(case, monkeypatch)
    monkeypatch.setattr(
        scheduler, "run", lambda *args: pytest.fail("Must not rerun completed forecast")
    )
    monkeypatch.setattr(FTPSource, "download", lambda *args: pytest.fail("Must not download"))
    assert scheduler.poll_once(args, BACKEND) is None
    assert not stale.exists()
    # Files must exist and pass their hashes; an old manifest alone is insufficient.
    next(target.glob("*.tif")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        scheduler.poll_once(args, BACKEND)


def test_new_hour_is_pinned_and_failed_window_is_retried(case, monkeypatch):
    args = poll_args(case)
    mock_listing(case, monkeypatch)
    calls = []

    def process(invocation, backend):
        calls.append(invocation.run_time)
        if len(calls) == 1:
            raise DataNotReady("missing predecessor")
        return case["root"] / "completed"

    monkeypatch.setattr(scheduler, "run", process)
    assert scheduler.poll_once(args, BACKEND) is None
    assert scheduler.poll_once(args, BACKEND) == case["root"] / "completed"
    assert calls == [case["run_time"], case["run_time"]]
    assert args.run_time is None


def test_failed_download_retries_on_next_poll(case, monkeypatch):
    args = poll_args(case)
    mock_listing(case, monkeypatch)
    calls = []

    def process(invocation, backend):
        calls.append(invocation.run_time)
        if len(calls) == 1:
            raise ConnectionError("FTP unavailable")
        return case["root"] / "completed"

    monkeypatch.setattr(scheduler, "run", process)
    with pytest.raises(ConnectionError):
        scheduler.poll_once(args, BACKEND)
    assert scheduler.poll_once(args, BACKEND) == case["root"] / "completed"
    assert len(calls) == 2


def test_no_available_observations_waits(case, monkeypatch):
    args = poll_args(case)
    mock_listing(case, monkeypatch)

    def unavailable(*args, **kwargs):
        raise DataNotReady("No hourly observations available yet")

    monkeypatch.setattr(FTPSource, "window", unavailable)
    monkeypatch.setattr(scheduler, "run", lambda *args: pytest.fail("Data are unavailable"))
    assert scheduler.poll_once(args, BACKEND) is None


def test_concurrent_poll_is_skipped_and_lock_is_reusable(case, monkeypatch):
    args = poll_args(case)
    monkeypatch.setattr(FTPSource, "window", lambda *args, **kwargs: pytest.fail("Lock not held"))
    with scheduler.poll_lock(args.output_dir, "gsmap_now", BACKEND) as acquired:
        assert acquired
        assert scheduler.poll_once(args, BACKEND) is None
    with scheduler.poll_lock(args.output_dir, "gsmap_now", BACKEND) as acquired:
        assert acquired


def test_scheduled_mode_rejects_fixed_run_time(case):
    args = poll_args(case)
    args.run_time = case["run_time"]
    with pytest.raises(ValueError, match="no --run-time"):
        scheduler.poll_once(args, BACKEND)


def test_stale_and_future_data_are_not_processed():
    with pytest.raises(DataNotReady, match="stale"):
        require_fresh(parse_time("2000-01-01T00:00:00Z"), 12)
    with pytest.raises(DataNotReady, match="future"):
        require_fresh(datetime.now(UTC) + timedelta(hours=1), 12)


def test_ftp_transfer_backoff_and_incomplete_gzip(case, monkeypatch):
    monkeypatch.setenv("FTP_JAXA_USER", "test")
    monkeypatch.setenv("FTP_JAXA_PASSWORD", "test")
    config = Config.load(case["config"])
    source = FTPSource(config.remote_directory, case["root"] / "cache", config.domain, config.ftp)
    calls, sleeps = [], []

    class FakeFTP:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, host):
            pass

        def login(self, user, password):
            pass

        def retrbinary(self, command, callback):
            calls.append(command)
            if len(calls) == 1:
                callback(b"partial transfer")
                raise OSError("network failure")
            if len(calls) == 2:
                callback(gzip.compress(b"truncated GSMaP grid"))
            else:
                callback(case["paths"][0].read_bytes())

    monkeypatch.setattr(acquisition.ftplib, "FTP", FakeFTP)
    monkeypatch.setattr(acquisition.time, "sleep", sleeps.append)
    record = Observation(case["run_time"], case["paths"][0].name, "/now/latest/file.dat.gz")
    path = source.download(record)
    assert path.read_bytes() == case["paths"][0].read_bytes()
    assert len(calls) == config.ftp.attempts
    assert sleeps == [0.01, 0.02]
    assert not list(path.parent.glob("*.part"))


def test_exhausted_ftp_attempts_leave_no_cache_file(case, monkeypatch):
    monkeypatch.setenv("FTP_JAXA_USER", "test")
    monkeypatch.setenv("FTP_JAXA_PASSWORD", "test")
    config = Config.load(case["config"])
    source = FTPSource(config.remote_directory, case["root"] / "cache", config.domain, config.ftp)
    calls, sleeps = [], []

    def fail_connect(*args, **kwargs):
        calls.append(1)
        raise OSError("offline")

    monkeypatch.setattr(acquisition.ftplib, "FTP", fail_connect)
    monkeypatch.setattr(acquisition.time, "sleep", sleeps.append)
    record = Observation(case["run_time"], "file.dat.gz", "/now/latest/file.dat.gz")
    with pytest.raises(ConnectionError, match="after 3 attempts"):
        source.download(record)
    assert len(calls) == config.ftp.attempts
    assert sleeps == [0.01, 0.02]
    assert not list(source.cache.rglob("*.dat.gz"))
    assert not list(source.cache.rglob("*.part"))
