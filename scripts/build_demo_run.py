"""Build the precomputed default run the app opens on (data/demo/default_run.pkl).

Runs the smart charging model once at every editable assumption's default,
starting today (London), and saves the result with its stamp. See
docs/explainers/demo-run.md.

    uv run python scripts/build_demo_run.py

``--vehicles`` and ``--weeks`` shrink the run for a quick check on a busy
machine; the app then labels the loaded run by how it differs from the
defaults instead of calling it the default.
"""

from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from axle_studio.model import assumptions
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.demo_run import DEMO_RUN_PATH, save_demo_run

HEARTBEAT_SECONDS = 30


def _heartbeat(began: float, done: threading.Event) -> None:
    """Print elapsed time until the run finishes; the model has no progress hook."""
    while not done.wait(HEARTBEAT_SECONDS):
        print(f"  still running, {time.perf_counter() - began:.0f} s so far", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vehicles", type=int, help="fleet size (default: the assumption's)")
    parser.add_argument("--weeks", type=int, help="simulated weeks (default: the assumption's)")
    parser.add_argument("--out", default=str(DEMO_RUN_PATH), help="output file")
    args = parser.parse_args()

    values = assumptions.editable_defaults()
    if args.vehicles is not None:
        values["vehicle_count"] = args.vehicles
    if args.weeks is not None:
        values["evaluation_world_count"] = args.weeks
    start = datetime.now(ZoneInfo("Europe/London")).date()

    print(
        f"Building {values['vehicle_count']:,} EVs x {values['evaluation_world_count']} weeks "
        f"from {start} (about 5 minutes at the defaults)...",
        flush=True,
    )
    began = time.perf_counter()
    done = threading.Event()
    threading.Thread(target=_heartbeat, args=(began, done), daemon=True).start()
    try:
        result = run_forecast_from_assumptions(start, model="action", values=values)
    finally:
        done.set()
    seconds = time.perf_counter() - began
    print(f"Model run finished in {seconds:.0f} s; saving...", flush=True)

    path = Path(args.out)
    stamp = save_demo_run(result, values, path, build_seconds=round(seconds, 1))
    print(
        f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB): "
        f"{values['vehicle_count']:,} EVs x {values['evaluation_world_count']} weeks "
        f"from {start}, built in {seconds:.1f} s at commit {stamp['git_commit']}."
    )


if __name__ == "__main__":
    main()
