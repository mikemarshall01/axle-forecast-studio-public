"""Axle Forecast Studio: five pages in top navigation with a run bar on every page.

Layout follows docs/DASHBOARD_DESIGN.md sections 0 to 3 (decision 0004 item
28). Navigation sits in the top bar and the run bar at the top of the main
area, so Run is reachable at phone width without opening a menu. The sidebar
is not used.
"""

from pathlib import Path

from axle_studio.ui.pages import page_renderer
from axle_studio.ui.registry import PAGES
from axle_studio.ui.run_controller import execute_requested_run, render_run_bar
from axle_studio.ui.style import APP_CSS

WORDMARK_PATH = Path(__file__).parent / "assets" / "wordmark.svg"


def main() -> None:
    """Build navigation, draw the run bar and the page, then honour a Run click.

    Nothing here runs the model unless the Run button was clicked (AGENTS.md).
    """

    import streamlit as st

    st.set_page_config(
        page_title="Axle Forecast Studio",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.logo(str(WORDMARK_PATH), size="large")
    # The one app-wide CSS block (polish plan G1): main padding and the KPI
    # tile's type, the two things config.toml has no setting for.
    st.html(APP_CSS)
    pages = [
        st.Page(page_renderer(st, page), title=page.name, url_path=page.slug) for page in PAGES
    ]
    navigation = st.navigation(pages, position="top")
    # The entry script draws the run bar above whichever page runs, so every
    # page shares one model control, status chip, Edit assumptions and Run.
    status_slot = render_run_bar(st)
    navigation.run()
    # The model runs last, after the page is drawn, so the page stays readable
    # during the run (design section 3.1).
    if status_slot is not None:
        execute_requested_run(st, status_slot)


if __name__ == "__main__":
    main()
