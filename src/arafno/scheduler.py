"""ARAFNO adapter for bundled scheduled polling."""

from ._runtime import scheduler
from .entrypoint import BACKEND


def poll_lock(output_dir, product):
    return scheduler.poll_lock(output_dir, product, BACKEND)


def completed_run(target, timestamp, product, steps):
    return scheduler.completed_run(target, timestamp, product, steps, BACKEND)


def poll_once(args):
    return scheduler.poll_once(args, BACKEND)
