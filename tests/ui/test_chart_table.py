"""The shared chart block renders title, chart, caption and its data table."""

import pandas as pd
import plotly.graph_objects as go

from axle_studio.ui.components.chart_table import chart_block


class _RecordingStreamlit:
    """Records calls in order; ``expander`` returns itself as a context manager."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return record

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


def test_chart_block_renders_one_block_in_the_design_order() -> None:
    st = _RecordingStreamlit()
    frame = pd.DataFrame({"interval": [1, 2], "kW": [3.0, 4.0]})

    chart_block(
        st,
        go.Figure(go.Scatter(x=[1, 2], y=[3, 4])),
        title="Fleet home import (kW)",
        caption="P10–P90 across 2 simulated weeks. Model output.",
        frame=frame,
        definition="Grid import at home chargers.",
        height=380,
        key="fleet-import",
    )

    assert [name for name, _, _ in st.calls] == [
        "markdown",
        "plotly_chart",
        "caption",
        "expander",
        "dataframe",
        "caption",
    ]
    _, (figure,), chart_kwargs = st.calls[1]
    assert figure.layout.height == 380
    assert chart_kwargs["theme"] is None
    assert chart_kwargs["config"] == {"displayModeBar": False}
    assert st.calls[0][1] == ("**Fleet home import (kW)**",)
    assert st.calls[3][1] == ("Data and definition",)
    assert st.calls[4][1][0] is frame
