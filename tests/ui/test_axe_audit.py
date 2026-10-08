"""Browser accessibility audit (axe-core) over every page and lens: slow, opt-in.

Runs only with ``AXLE_BROWSER_AUDIT=1`` and Playwright plus its Chromium
build installed (``uv run playwright install chromium``); otherwise it
skips, so the default suite never starts a browser. Serious and critical
violations fail the test unless their rule id is in ``ACCEPTED`` with a
reason; moderate and minor ones are reported in the audit folder only. The
folder is dated and lives outside the repository.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

pytestmark = pytest.mark.slow

ACCEPTED: dict[str, str] = {
    "color-contrast": (
        "Recorded, not yet fixed (axle-conventions, 29 September 2026): the selected "
        "segmented-control label renders in the theme primary colour at 2.9-3.2:1; a theme "
        "decision for the final UI critique."
    ),
    "scrollable-region-focusable": (
        "Recorded, not yet fixed: the KaTeX display on How it works > The forecast is a "
        "scrollable block without keyboard focus; for the final UI critique."
    ),
}
"""axe rule id -> why a serious/critical finding is accepted for now."""


@pytest.mark.skipif(
    os.environ.get("AXLE_BROWSER_AUDIT") != "1", reason="set AXLE_BROWSER_AUDIT=1 to run"
)
def test_axe_finds_no_unaccepted_serious_violations() -> None:
    pytest.importorskip("playwright")
    pytest.importorskip("axe_playwright_python")
    from axe_audit import run_audit

    out, results = run_audit(Path(tempfile.gettempdir()) / "axle-ui-audit", screenshots=False)

    serious = [
        f"{result.slug}: {violation['id']} ({violation['impact']}, {len(violation['nodes'])} nodes)"
        for result in results
        for violation in result.violations
        if violation.get("impact") in ("serious", "critical") and violation["id"] not in ACCEPTED
    ]
    assert results, f"no views audited, see {out}"
    assert serious == [], f"see {out}/summary.md"
