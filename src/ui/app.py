"""PM RAID Agent — Streamlit UI shell (Task 9.1).

Run from project root: streamlit run src/ui/app.py
Document parsing, review and export are connected in later tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Streamlit executes this script directly, so ensure project-root imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_EXTENSIONS = frozenset({"pdf", "docx", "xlsx"})
MAX_FILES = 20
PAGES = ("Upload", "Review", "Export")


def initialize_session_state() -> None:
    """Initialize once; reruns must not discard analysis or decisions."""
    defaults = {
        "raid_session_id": None,
        "raid_uploaded_files": [],
        "raid_documents": [],
        "raid_workflow": None,
        "raid_thread_config": None,
        "raid_workflow_state": None,
        "raid_review_decisions": {},
        "raid_export_result": None,
        "raid_current_page": "Upload",
        "raid_navigation": "Upload",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.raid_session_id is None:
        st.session_state.raid_session_id = str(uuid4())


def validate_uploads(files) -> list[str]:
    """Return user-facing validation errors without reading file contents."""
    errors = []
    if len(files) > MAX_FILES:
        errors.append(f"Select at most {MAX_FILES} files.")
    for file in files:
        filename = getattr(file, "name", "")
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in SUPPORTED_EXTENSIONS:
            errors.append(f"Unsupported format: {filename or '(unnamed file)'}")
    return errors


def render_upload_page() -> None:
    st.header("Upload project documents")
    files = st.file_uploader(
        "Choose PDF, DOCX or XLSX files",
        type=sorted(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="raid_file_uploader",
    )
    if not files:
        st.info("Upload project documents to begin RAID analysis.")
        return
    errors = validate_uploads(files)
    if errors:
        for error in errors:
            st.error(error)
        return
    st.session_state.raid_uploaded_files = list(files)
    st.success(f"{len(files)} document(s) selected.")
    for file in files:
        st.write(f"• {file.name}")
    st.caption("Document processing and analysis will be connected in Task 9.2.")


def render_review_page() -> None:
    st.header("Review RAID findings")
    st.info("Finding review and approval controls will be connected in Task 9.3.")


def render_export_page() -> None:
    st.header("Export approved RAID findings")
    st.info("Excel download controls will be connected in Task 9.4.")


def main() -> None:
    st.set_page_config(page_title="PM RAID Agent", page_icon="📋", layout="wide")
    initialize_session_state()
    st.title("PM RAID Agent")
    st.caption("Evidence-based project RAID analysis with explicit PM approval")
    page = st.sidebar.radio(
        "Navigation", PAGES, key="raid_navigation",
    )
    st.session_state.raid_current_page = page
    if page == "Upload":
        render_upload_page()
    elif page == "Review":
        render_review_page()
    else:
        render_export_page()


if __name__ == "__main__":
    main()
