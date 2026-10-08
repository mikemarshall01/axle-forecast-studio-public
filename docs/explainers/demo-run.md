# The precomputed default run

A full default run (1,000 EVs × 100 simulated weeks) takes a couple of minutes, so I let the app open on a result saved earlier. It appears as **Run 0 · precomputed default**: the run bar's identity line reads "Run 0 · precomputed · 1,000 EVs · 100 weeks · …", and the evidence badge's tooltip says it was saved earlier and loaded when the app opened. Loading a saved result is not a run. The model runs only when someone clicks Run simulation, and that run becomes Run 1. Run 0 stays in the history for Compare like any kept run, and shows as stale when the draft differs from it (including after midnight, when the study start date moves on).

## Rebuild it

```bash
uv run python scripts/build_demo_run.py
```

This runs the smart charging model at every editable assumption's default, starting today (London), and writes `data/demo/default_run.pkl` (ignored by Git; about 350 MB, about 2.5 minutes on the development machine). `--vehicles` and `--weeks` build a smaller file for a quick check; the app then labels it by how it differs from the defaults, for example "Run 0 · precomputed · Fleet size 1000 → 200 EVs (+1 more)".

Rebuild on the day of a demo, so the run starts today and the bar reads Current.

## When the app ignores the file

The file starts with a small stamp: a hash of the model source (`src/axle_studio/model/*.py`), the Python, NumPy and pandas versions, the package version, the Git commit, a hash of the run's settings, the creation time and the build time. The app loads the result only when the model-source hash and the library versions match the running code. Any model change therefore makes the file stale; the app then opens on the ordinary "Nothing to show yet" state with no error, until the file is rebuilt. Changes to the dashboard alone keep it valid.

I chose a pickle of the frozen `ForecastResult` as the format because it round-trips every field exactly with no per-field code. Pickle is only reliable between the same library versions (pinned by `uv.lock`), which is why the stamp checks them. The file is written only by the script above, on this machine; never load one from elsewhere.

The Docker image builds its own file while the image is built (`RUN python scripts/build_demo_run.py` in the `Dockerfile`), so a container opens on Run 0. Its study week starts on the build date, so rebuild the image on the day of a demo for the bar to read Current. A locally built `data/` folder is never copied into the image. The server loads the file once and every visitor's session shares that copy read-only.

## Usage logging

When the host sets `AXLE_USAGE_LOG=1`, the hosted demo can log anonymous page views and runs (which pages and lenses were opened, whether a run or download happened, how long a session lasted) with no IP address or other personal data (`src/axle_studio/ui/usage_log.py`); it stays off, and silent, everywhere else.
