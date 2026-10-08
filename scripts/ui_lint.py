"""Lint the UI source for values that belong in ``ui/style.py`` or the theme.

One source of truth for design tokens: colours, chart text size and app CSS
live in ``src/axle_studio/ui/style.py`` and ``.streamlit/config.toml``
(docs/DESIGN_TOKENS.md). A view that spells a colour, an ``rgba(...)``, a
``px`` length or a chart font size inline can drift from the tokens without
any test noticing, so this lint reports every such literal outside
``style.py``. Genuine exceptions are recorded, with a reason, in
``tests/ui/test_ui_lint.py`` rather than silenced here.

Usage from the repository root (exit 1 when anything is found):

    uv run --frozen --group dev python scripts/ui_lint.py
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "axle_studio" / "ui"
TOKEN_FILES = frozenset({"style.py"})

_HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_RGB = re.compile(r"rgba?\([^)]*\)")
_PX = re.compile(r"\b\d+(?:\.\d+)?px\b")
_FONT_SIZE = re.compile(r"font_size\s*=\s*(\d+)|font\s*=\s*\{[^}]*\"size\"\s*:\s*(\d+)")


@dataclass(frozen=True, slots=True)
class Finding:
    path: str  # relative to ``src/axle_studio/ui``
    line: int
    rule: str
    literal: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} [{self.rule}] {self.literal}"


def _string_tokens(source: str) -> list[tuple[int, str]]:
    """``(line, text)`` for every string literal, so comments never match."""

    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            found.append((token.start[0], token.string))
    return found


def lint_source(path: str, source: str) -> list[Finding]:
    """Findings for one module's source (``path`` is the relative label)."""

    findings: list[Finding] = []
    for line, text in _string_tokens(source):
        for rule, pattern in (("hex-colour", _HEX), ("rgba-colour", _RGB), ("px-length", _PX)):
            for match in pattern.finditer(text):
                findings.append(Finding(path, line, rule, match.group(0)))
    for number, code in enumerate(source.splitlines(), start=1):
        stripped = code.split("#", 1)[0]
        for match in _FONT_SIZE.finditer(stripped):
            size = match.group(1) or match.group(2)
            findings.append(Finding(path, number, "font-size", f"{size} (use CHART_FONT_PX)"))
    return findings


def lint_ui(ui_dir: Path = UI_DIR) -> list[Finding]:
    """Findings for every module under ``ui_dir`` except the token files."""

    findings: list[Finding] = []
    for path in sorted(ui_dir.rglob("*.py")):
        if path.name in TOKEN_FILES:
            continue
        findings.extend(lint_source(str(path.relative_to(ui_dir)), path.read_text()))
    return findings


def main() -> int:
    findings = lint_ui()
    print("\n".join(map(str, findings)) or "ui_lint: no findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
