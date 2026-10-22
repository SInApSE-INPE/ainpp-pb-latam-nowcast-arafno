from pathlib import Path

from arafno._runtime.deployment import render_units

if __name__ == "__main__":
    render_units(Path(__file__).resolve().parent.parent, "arafno", "ARAFNO", "arafno-now")
