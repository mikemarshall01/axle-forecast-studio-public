"""Browser audit: axe-core on every page and lens, plus screenshots at 1440 and 390 px.

What it does, in one browser session against a Streamlit server it starts
itself on a free port: opens the app, clicks Run simulation once and waits
for the run bar to show "Run 1 ·", then for every page in ``registry.PAGES``
and every lens on it, selects the lens, runs axe-core (WCAG 2.x rules via
``axe-playwright-python``) at 1440 px and saves full-page screenshots at
1440 and 390 px. Nothing is written inside the repository: the output folder
is dated and lives wherever ``--out`` points (default: the system temp dir).

Usage from the repository root (Playwright and a Chromium build must be
installed: ``uv sync --frozen --group dev`` then ``uv run playwright install
chromium``):

    uv run --frozen --group dev env PYTHONPATH=src python scripts/axe_audit.py \\
        --out /tmp/axle-ui-audit

Outputs under ``<out>/<YYYY-MM-DD-HHMM>/``: ``axe-<page>-<lens>.json`` (raw
axe results), ``<page>-<lens>-1440.png`` and ``-390.png``, and
``summary.md`` (violations by rule and impact, one row per view). Compare is
audited in its pre-pair state (one run only), which the summary notes.

This is a development check, not part of the default pytest run;
``tests/ui/test_axe_audit.py`` wraps it behind ``AXLE_BROWSER_AUDIT=1`` and
the ``slow`` marker. Ports 8501, 8502 and 8527 are never used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESERVED_PORTS = frozenset({8501, 8502, 8527})
DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}
RUN_TIMEOUT_S = 300


@dataclass(frozen=True, slots=True)
class ViewResult:
    page: str
    lens: str | None
    violations: list[dict[str, Any]]

    @property
    def slug(self) -> str:
        lens = re.sub(r"[^a-z0-9]+", "-", (self.lens or "page").lower()).strip("-")
        return f"{self.page}-{lens}"


def free_port(start: int = 8641) -> int:
    for port in range(start, start + 200):
        if port in RESERVED_PORTS:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def start_server(port: int, log: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO / "streamlit_app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    handle = log.open("w")
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return process
        if process.poll() is not None:
            raise RuntimeError(f"streamlit exited early, see {log}")
        time.sleep(0.5)
    process.terminate()
    raise RuntimeError(f"streamlit did not open port {port} in 60 s, see {log}")


def _settle(page: Any) -> None:
    """Wait for Streamlit to finish its script run (status widget gone)."""

    page.wait_for_timeout(300)
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="detached", timeout=120_000)
    page.wait_for_timeout(300)


def close_dialog(page: Any) -> None:
    """Dismiss an open Streamlit dialog (the app may open Edit assumptions on first load)."""

    dialog = page.locator('[data-testid="stDialog"]')
    if dialog.count() and dialog.first.is_visible():
        close = dialog.first.locator('button[aria-label="Close"]')
        if close.count():
            close.first.click()
        else:
            page.keyboard.press("Escape")
        dialog.first.wait_for(state="hidden", timeout=10_000)
        _settle(page)


def run_once(page: Any) -> None:
    close_dialog(page)
    page.get_by_role("button", name="Run simulation").first.click()
    page.get_by_text(re.compile(r"^Run 1 · ")).first.wait_for(timeout=RUN_TIMEOUT_S * 1000)
    _settle(page)


def select_lens(page: Any, lens: str) -> None:
    # A segmented control is a radiogroup of buttons whose text sits in a
    # markdown <p>, so match the inner text rather than an accessible name.
    option = page.locator(
        'button[data-variant="segmented_control"]',
        has_text=re.compile(rf"^\s*{re.escape(lens)}\s*$"),
    )
    option.first.click()
    _settle(page)


def go_to_page(page: Any, name: str) -> None:
    """Follow the top navigation link, which keeps the Streamlit session (and the run).

    A URL ``goto`` opens a fresh session with no result, so every page after
    the first would show its empty state.
    """

    page.get_by_role("link", name=re.compile(rf"^\s*{re.escape(name)}\s*$")).first.click()
    _settle(page)


def full_page_screenshot(page: Any, path: Path, viewport: dict[str, int]) -> None:
    """Screenshot the whole scrolled page at ``viewport``'s width.

    Streamlit scrolls its main container rather than the document, so
    Playwright's ``full_page`` captures only one viewport; growing the
    viewport to the container's scroll height captures everything.
    """

    page.set_viewport_size(viewport)
    page.wait_for_timeout(500)
    height = page.evaluate(
        "() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
        " return Math.max(m ? m.scrollHeight : 0, document.documentElement.scrollHeight); }"
    )
    page.set_viewport_size({"width": viewport["width"], "height": min(int(height) + 24, 12_000)})
    page.wait_for_timeout(500)
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size(viewport)


def audit_views(base_url: str, out: Path, *, screenshots: bool = True) -> list[ViewResult]:
    from axe_playwright_python.sync_playwright import Axe
    from playwright.sync_api import sync_playwright

    from axle_studio.ui.registry import PAGES

    axe = Axe()
    results: list[ViewResult] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=DESKTOP, color_scheme="dark")
        page = context.new_page()
        page.goto(base_url, wait_until="networkidle")
        _settle(page)
        run_once(page)
        for entry in PAGES:
            lenses: tuple[str | None, ...] = tuple(lens.name for lens in entry.lenses) or (None,)
            for lens in lenses:
                page.set_viewport_size(DESKTOP)
                go_to_page(page, entry.name)
                close_dialog(page)
                page.get_by_text(re.compile(r"^Run 1 · ")).first.wait_for(timeout=30_000)
                if lens is not None:
                    select_lens(page, lens)
                report = axe.run(page).response
                result = ViewResult(entry.slug, lens, list(report.get("violations", [])))
                results.append(result)
                (out / f"axe-{result.slug}.json").write_text(json.dumps(report, indent=1))
                if screenshots:
                    full_page_screenshot(page, out / f"{result.slug}-1440.png", DESKTOP)
                    full_page_screenshot(page, out / f"{result.slug}-390.png", PHONE)
        browser.close()
    return results


def summary(results: list[ViewResult]) -> str:
    lines = [
        "# UI audit: axe-core violations by view",
        "",
        "One run at the default draft size; Compare is in its pre-pair state.",
        "",
        "| View | Violations | Rules (impact × nodes) |",
        "| --- | --- | --- |",
    ]
    for result in results:
        rules = ", ".join(
            f"{v['id']} ({v.get('impact')} × {len(v.get('nodes', []))})" for v in result.violations
        )
        lines.append(f"| {result.slug} | {len(result.violations)} | {rules or '-'} |")
    by_rule: dict[str, int] = {}
    for result in results:
        for violation in result.violations:
            by_rule[violation["id"]] = by_rule.get(violation["id"], 0) + len(violation["nodes"])
    lines += ["", "## Nodes per rule, all views", ""]
    lines += [
        f"- {rule}: {count}" for rule, count in sorted(by_rule.items(), key=lambda kv: -kv[1])
    ]
    return "\n".join(lines) + "\n"


def run_audit(out_root: Path, *, screenshots: bool = True) -> tuple[Path, list[ViewResult]]:
    out = out_root / datetime.now().strftime("%Y-%m-%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = start_server(port, out / "streamlit.log")
    try:
        results = audit_views(f"http://127.0.0.1:{port}", out, screenshots=screenshots)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    (out / "summary.md").write_text(summary(results))
    return out, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "axle-ui-audit",
        help="folder for the dated output subfolder (outside the repository)",
    )
    parser.add_argument("--no-screenshots", action="store_true")
    args = parser.parse_args()
    out, results = run_audit(args.out, screenshots=not args.no_screenshots)
    print((out / "summary.md").read_text())
    print(f"written to {out}")
    return 1 if any(result.violations for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
