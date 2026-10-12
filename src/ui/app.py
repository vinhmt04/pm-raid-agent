"""PM RAID Agent — Streamlit UI shell (Task 9.1).

Run from project root: streamlit run src/ui/app.py
Document parsing, review and export are connected in later tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from typing import Any

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
        "raid_processing_results": [],
        "raid_processing_errors": [],
        "raid_processed_fingerprint": None,
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


@dataclass
class ProcessingResult:
    filename: str
    status: str
    sections: int
    characters: int
    error: str = ""


def process_uploaded_documents(files, factory=None):
    """Parse uploads via existing document processors; isolate individual failures.

    Only validated files are written to a short-lived private temporary folder.
    File names are generated, preventing directory traversal and collisions.
    """
    errors = validate_uploads(files)
    if errors:
        raise ValueError("; ".join(errors))
    if factory is None:
        from src.document_processing.factory import DocumentProcessorFactory
        factory = DocumentProcessorFactory()

    documents, results = [], []
    with TemporaryDirectory(prefix="raid_upload_") as temp_dir:
        for index, upload in enumerate(files):
            original_name = str(upload.name)
            suffix = "." + original_name.rsplit(".", 1)[-1].lower()
            temp_path = Path(temp_dir) / f"document_{index}{suffix}"
            try:
                # Streamlit UploadedFile.getvalue() does not alter stream position.
                payload = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError("File is empty or cannot be read")
                temp_path.write_bytes(payload)
                metadata = factory.process_document(temp_path)
                metadata.filename = original_name
                status = str(metadata.processing_status)
                sections = len(metadata.content_sections)
                characters = sum(len(section.text) for section in metadata.content_sections)
                results.append(ProcessingResult(original_name, status, sections,
                                                characters, metadata.error_message or ""))
                if status in ("success", "partial_success") and sections:
                    documents.append(metadata)
            except Exception as exc:
                results.append(ProcessingResult(original_name, "error", 0, 0, str(exc)))
    return documents, results


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

    if st.button("Process documents", key="raid_process_documents"):
        # Invalidate stale review/export state when processing a new batch.
        st.session_state.raid_documents = []
        st.session_state.raid_processing_results = []
        st.session_state.raid_processing_errors = []
        st.session_state.raid_workflow = None
        st.session_state.raid_workflow_state = None
        st.session_state.raid_thread_config = None
        st.session_state.raid_review_decisions = {}
        st.session_state.raid_export_result = None
        with st.spinner("Extracting document content..."):
            documents, results = process_uploaded_documents(files)
        st.session_state.raid_documents = documents
        st.session_state.raid_processing_results = results
        st.session_state.raid_processing_errors = [r.error for r in results if r.error]

    results = st.session_state.raid_processing_results
    if results:
        st.subheader("Processing results")
        for result in results:
            message = (f"{result.filename}: {result.status} — "
                       f"{result.sections} sections, {result.characters} characters")
            if result.status == "success":
                st.success(message)
            elif result.status == "partial_success":
                st.warning(message)
            else:
                st.error(message)
            if result.error:
                st.caption(result.error)
        st.info(f"{len(st.session_state.raid_documents)} document(s) ready for RAID analysis.")
        st.caption("Start LangGraph analysis will be connected in Task 11.1.")


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
