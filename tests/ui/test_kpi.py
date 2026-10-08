"""The KPI tile component: markup, escaping, the four-per-row rule and badge lines."""

import pytest

from axle_studio.ui.components.kpi import badge_line, kpi, kpi_columns, kpi_html


class _RecordingStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return list(range(args[0])) if name == "columns" else self

        return record


def test_kpi_renders_label_value_unit_and_context_in_one_markdown_block() -> None:
    st = _RecordingStreamlit()

    kpi(st, "Peak", "1,234", unit="kW", context="P10–P90 1,100 to 1,400", help="Why")

    [(name, (body,), kwargs)] = st.calls
    assert name == "markdown"
    assert kwargs == {"unsafe_allow_html": True, "help": "Why", "width": "content"}
    assert '<div class="axle-kpi-label">Peak</div>' in body
    assert '<div class="axle-kpi-value">1,234<span class="axle-kpi-unit">kW</span></div>' in body
    assert '<div class="axle-kpi-context">P10–P90 1,100 to 1,400</div>' in body


def test_kpi_omits_empty_unit_and_context_and_escapes_text() -> None:
    body = kpi_html("A <b> & B", "−£5")

    assert "axle-kpi-unit" not in body
    assert "axle-kpi-context" not in body
    assert "A &lt;b&gt; &amp; B" in body


@pytest.mark.parametrize("count", [1, 4])
def test_kpi_columns_allows_one_to_four(count: int) -> None:
    assert len(kpi_columns(_RecordingStreamlit(), count)) == count


@pytest.mark.parametrize("count", [0, 5])
def test_kpi_columns_refuses_more_than_four(count: int) -> None:
    with pytest.raises(ValueError, match="1 to 4"):
        kpi_columns(_RecordingStreamlit(), count)


def test_badge_line_puts_text_values_in_a_grey_badge() -> None:
    st = _RecordingStreamlit()

    badge_line(st, "Action chosen", "Smart charging [illustrative]", help="What it is")

    [(name, (body,), kwargs)] = st.calls
    assert name == "markdown"
    assert body == "Action chosen :gray-badge[Smart charging \\[illustrative\\]]"
    assert kwargs == {"help": "What it is"}
