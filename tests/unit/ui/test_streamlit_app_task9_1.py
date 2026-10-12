"""Task 9.1 UI shell tests. Streamlit's AppTest exercises actual page rendering."""
from pathlib import Path
from types import SimpleNamespace

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from src.ui.app import MAX_FILES, validate_uploads

APP = str(Path(__file__).resolve().parents[3] / "src" / "ui" / "app.py")


def test_upload_format_validation():
    assert validate_uploads([SimpleNamespace(name="report.PDF"), SimpleNamespace(name="plan.docx")]) == []
    assert len(validate_uploads([SimpleNamespace(name="report.exe")])) == 1
    assert len(validate_uploads([SimpleNamespace(name=f"file{i}.pdf") for i in range(MAX_FILES + 1)])) == 1


def test_navigation_and_session_state():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert at.session_state["raid_session_id"]
    assert at.session_state["raid_current_page"] == "Upload"
    session_id = at.session_state["raid_session_id"]
    at.sidebar.radio[0].set_value("Review").run()
    assert not at.exception
    assert at.session_state["raid_current_page"] == "Review"
    assert at.session_state["raid_session_id"] == session_id
    at.sidebar.radio[0].set_value("Export").run()
    assert not at.exception
    assert at.session_state["raid_current_page"] == "Export"
