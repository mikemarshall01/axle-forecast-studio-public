# Axle Forecast Studio: the Streamlit dashboard, with the precomputed default
# run built into the image so a new visitor opens on Run 0 straight away.
#
#   docker build -t axle-forecast-studio:local .
#   docker run --rm -p 127.0.0.1:8501:8501 axle-forecast-studio:local
#
# PORT sets the listening port (default 8501). AXLE_MAX_CONCURRENT_RUNS and
# AXLE_MAX_EV_WEEKS cap full runs on a shared host (src/axle_studio/ui/run_guard.py);
# both are unset here, so a local container has no caps.

FROM python:3.12-slim

# uv installs the exact versions in uv.lock. The saved run is a pickle whose
# stamp checks the NumPy and pandas versions, so it must be built and read by
# the same locked environment (ui/demo_run.py).
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# The package is installed editable from /app/src, so its paths resolve under
# /app: data/demo/ for the saved run and docs/explainers/ for How it works.
COPY src ./src
RUN uv sync --frozen --no-dev

COPY streamlit_app.py ./
COPY assets ./assets
COPY docs/explainers ./docs/explainers
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY scripts/build_demo_run.py ./scripts/build_demo_run.py

# The full default run (1,000 EVs x 100 simulated weeks, starting on the
# build date), written to data/demo/default_run.pkl. Loading it is not a run;
# the model runs in the container only after a Run click.
RUN python scripts/build_demo_run.py

# Run as an ordinary user; the app only reads files under /app.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.port=${PORT}"]
