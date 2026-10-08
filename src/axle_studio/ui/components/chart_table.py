"""The one chart block every dashboard chart uses (design section 3.4).

A chart block is: a bold title stating the quantity and unit, the Plotly
chart, one caption line naming the spread and evidence class, and one
collapsed "Data and definition" expander holding the same frame as a table
plus a short definition. One helper, rather than per-view layout, keeps every
chart the same shape and guarantees each chart has its table equivalent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..style import PLOTLY_CONFIG, style_figure


def chart_block(
    st: Any,
    figure: go.Figure,
    *,
    title: str,
    caption: str,
    frame: pd.DataFrame,
    definition: str,
    height: int,
    key: str | None = None,
) -> None:
    """Render one chart with its caption and its data table.

    ``height`` comes from ``style.CHART_HEIGHTS`` (decision 0004 item 26).
    ``caption`` names the spread ("P10–P90 across 100 simulated weeks" or
    "P5–P95 across EVs") and the evidence class. ``frame`` is the data the
    chart plots, shown unchanged; the block never calculates anything.
    """

    st.markdown(f"**{title}**")
    # theme=None: the Axle template, not Streamlit's theme, styles the chart,
    # so the legend-below and margin defaults hold at every width.
    st.plotly_chart(
        style_figure(figure, height=height),
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )
    st.caption(caption)
    with st.expander("Data and definition"):
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(definition)
