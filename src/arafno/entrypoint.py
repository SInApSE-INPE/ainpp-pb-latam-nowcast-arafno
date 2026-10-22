"""ARAFNO CLI backed by the bundled operational runtime."""

from . import __version__
from ._runtime import pipeline
from ._runtime.backend import Backend
from ._runtime.pipeline import require_fresh as require_fresh
from ._runtime.pipeline import validate_runtime as validate_runtime
from .model import forecast_steps, load_model

BACKEND = Backend("ARAFNO", __version__, load_model, forecast_steps, default_steps=6)


def parser():
    return pipeline.parser(BACKEND)


def forecast_directory(root, product, run_time):
    return pipeline.forecast_directory(root, product, run_time, BACKEND)


def run(args):
    return pipeline.run(args, BACKEND)


def main(argv=None):
    return pipeline.main(BACKEND, argv)
