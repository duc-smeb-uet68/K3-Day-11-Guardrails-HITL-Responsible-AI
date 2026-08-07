from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_live_model():
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15)
    app.run()

    assert not app.exception
    assert any("VinBank Secure Assistant" in title.value for title in app.title)
    assert any("Security Console" in tab.label for tab in app.tabs)

