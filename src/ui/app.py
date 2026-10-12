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
        "raid_review_edits": {},
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


def start_raid_analysis(documents, session_id, workflow_factory=None):
    """Run the real LangGraph graph and return its persisted checkpoint.

    A failed run is never stored in Streamlit session state as successful.
    """
    if not documents:
        raise ValueError("Process at least one document before starting RAID analysis.")
    if workflow_factory is None:
        from src.workflow.langgraph_workflow import RAIDWorkflow
        workflow_factory = RAIDWorkflow
    workflow = workflow_factory({})
    run = workflow.start_analysis(documents, session_id)
    config = run["thread_config"]
    workflow.graph.invoke(run["initial_state"], config=config)
    state, findings, interrupted = get_review_snapshot(workflow, config)
    if state is None:
        raise RuntimeError("LangGraph did not persist an analysis checkpoint.")
    if state.get("workflow_status") == "error":
        raise RuntimeError("; ".join(state.get("error_messages", [])) or "Analysis failed.")
    # The analysis node catches individual provider errors; do not silently
    # present a successful empty result if every document analysis failed.
    analyses = state.get("individual_analyses", [])
    if analyses and all(getattr(a, "processing_errors", None) for a in analyses):
        raise RuntimeError("All document analyses failed: " + "; ".join(
            str(error) for analysis in analyses for error in analysis.processing_errors
        ))
    return workflow, config, state, findings, interrupted


def run_analysis_in_session(documents, session_id, session_state, workflow_factory=None):
    """Persist only a successful analysis; keep review/export in sync."""
    workflow, config, state, findings, interrupted = start_raid_analysis(
        documents, session_id, workflow_factory
    )
    session_state["raid_workflow"] = workflow
    session_state["raid_thread_config"] = config
    session_state["raid_workflow_state"] = state
    session_state["raid_review_decisions"] = {}
    session_state["raid_review_edits"] = {}
    session_state["raid_export_result"] = None
    return findings, interrupted


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
        st.session_state.raid_review_edits = {}
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
        if st.session_state.raid_workflow is None:
            if st.button("Start RAID analysis", key="raid_start_analysis_upload"):
                try:
                    with st.spinner("Analyzing documents with LangGraph..."):
                        findings, interrupted = run_analysis_in_session(
                            st.session_state.raid_documents,
                            st.session_state.raid_session_id,
                            st.session_state,
                        )
                    st.success(f"Analysis completed: {len(findings)} RAID findings.")
                    if interrupted:
                        st.info("Open Review to approve or reject findings.")
                    elif not findings:
                        st.info("No supported RAID findings identified.")
                except Exception as exc:
                    st.error(f"RAID analysis failed: {exc}")
        else:
            st.info("Analysis is available on the Review page.")


def get_review_snapshot(workflow, thread_config):
    """Read the actual checkpoint; never infer approval from UI state."""
    snapshot = workflow.graph.get_state(thread_config)
    if not snapshot or not snapshot.values:
        return None, [], False
    state = snapshot.values
    manager = state.get("raid_state_manager")
    findings = manager.get_all_findings() if manager is not None else []
    interrupted = any(getattr(task, "interrupts", ()) for task in (snapshot.tasks or ()))
    return state, findings, interrupted


def validate_review_decisions(findings, decisions):
    """Only explicit approve/reject choices for every pending item are accepted."""
    pending = {f.id for f in findings if f.status.value == "pending"}
    if set(decisions) != pending or any(
        value not in ("approved", "rejected") for value in decisions.values()
    ):
        raise ValueError("Choose Approve or Reject for every pending RAID finding.")
    return {finding_id: decisions[finding_id] for finding_id in pending}


def apply_review_edits(manager, edits):
    """Edit content without changing approval status or source citations."""
    for finding_id, values in edits.items():
        finding = manager.get_finding(finding_id)
        if finding is None:
            raise ValueError(f"Unknown RAID finding: {finding_id}")
        if not isinstance(values, dict) or not set(values).issubset(
            {"description", "recommendations"}
        ):
            raise ValueError("Only description and recommendations may be edited")
        for field_name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
            manager.update_finding(finding_id, **{field_name: value.strip()})
    return manager


def submit_review(workflow, session_id, decisions, edits):
    """Resume first, then persist staged content edits in the completed checkpoint.

    Updating the graph checkpoint *before* Command(resume) can replace the
    interrupted task and lose its resume target.
    """
    config = {"configurable": {"thread_id": session_id}}
    state, findings, interrupted = get_review_snapshot(workflow, config)
    if not interrupted:
        raise ValueError("Workflow is not waiting for PM review")
    validated = validate_review_decisions(findings, decisions)
    # Validate all edits before any approval is applied.
    from copy import deepcopy
    preview = deepcopy(state["raid_state_manager"])
    apply_review_edits(preview, edits)
    result = workflow.resume_workflow(session_id, validated)
    completed = result["state"]
    manager = completed["raid_state_manager"]
    apply_review_edits(manager, edits)
    if edits:
        workflow.graph.update_state(config, {"raid_state_manager": manager})
    return workflow.graph.get_state(config).values


def render_review_page() -> None:
    st.header("Review RAID findings")
    if st.session_state.raid_workflow is None:
        documents = st.session_state.raid_documents
        if not documents:
            st.info("Process project documents on the Upload page first.")
            return
        st.info("Start analysis to extract RAID findings before PM review.")
        if st.button("Start RAID analysis", key="raid_start_analysis"):
            try:
                with st.spinner("Analyzing documents with LangGraph..."):
                    run_analysis_in_session(
                        documents,
                        st.session_state.raid_session_id,
                        st.session_state,
                    )
            except Exception as exc:
                st.error(f"RAID analysis failed: {exc}")
                return
        else:
            return

    workflow = st.session_state.raid_workflow
    config = st.session_state.raid_thread_config
    if not config:
        st.error("Workflow thread configuration is missing.")
        return
    try:
        state, findings, interrupted = get_review_snapshot(workflow, config)
    except Exception as exc:
        st.error(f"Cannot load RAID review: {exc}")
        return
    if state is None:
        st.warning("No workflow checkpoint found.")
        return
    st.session_state.raid_workflow_state = state
    if state.get("workflow_status") == "error":
        st.error("Analysis failed. Check workflow errors before reviewing.")
        for message in state.get("error_messages", []):
            st.caption(message)
        return
    if not findings:
        st.info("No RAID findings were identified for review.")
        return

    pending = [f for f in findings if f.status.value == "pending"]
    st.caption(f"{len(findings)} findings · {len(pending)} awaiting decision")
    decisions = st.session_state.raid_review_decisions
    edits = st.session_state.raid_review_edits
    for finding in findings:
        with st.expander(f"{finding.type.value.upper()} · {finding.description[:90]}", expanded=True):
            st.write(f"Evidence: {finding.evidence_type.value} · Status: {finding.status.value}")
            st.write(f"Impact: {finding.impact_assessment or 'Not assessed'}")
            for citation in finding.source_citations:
                st.caption(f"Source: {citation.filename} · {citation.location}")
                st.write(citation.excerpt)
            if finding.related_items:
                st.caption("Related findings: " + ", ".join(finding.related_items))
            description = st.text_area("Description", value=edits.get(finding.id, {}).get(
                "description", finding.description), key=f"raid_desc_{finding.id}", disabled=not interrupted)
            recommendations = st.text_area("Recommendations", value=edits.get(finding.id, {}).get(
                "recommendations", finding.recommendations), key=f"raid_rec_{finding.id}", disabled=not interrupted)
            changed = {}
            if description != finding.description:
                changed["description"] = description
            if recommendations != finding.recommendations:
                changed["recommendations"] = recommendations
            if changed:
                edits[finding.id] = changed
            else:
                edits.pop(finding.id, None)
            if finding.status.value == "pending" and interrupted:
                selection = st.radio("PM decision", ["Not decided", "Approve", "Reject"],
                    index={"approved": 1, "rejected": 2}.get(decisions.get(finding.id), 0),
                    key=f"raid_decision_{finding.id}", horizontal=True)
                if selection == "Not decided":
                    decisions.pop(finding.id, None)
                else:
                    decisions[finding.id] = "approved" if selection == "Approve" else "rejected"
    if interrupted and pending:
        st.caption("Editing does not approve a finding. All decisions must be explicit.")
        if st.button("Submit PM decisions", key="raid_submit_review"):
            try:
                updated = submit_review(workflow, st.session_state.raid_session_id,
                                        dict(decisions), dict(edits))
                st.session_state.raid_workflow_state = updated
                st.session_state.raid_review_edits = {}
                st.success("PM decisions saved to workflow checkpoint.")
            except Exception as exc:
                st.error(f"Review not submitted: {exc}")
    elif state.get("approval_complete"):
        st.success("PM review completed. Approved findings are ready for export.")


def prepare_approved_export(workflow, thread_config, service=None):
    """Use the persisted workflow checkpoint, never editable UI state."""
    from src.export.export_service import RAIDExportService, NoApprovedFindingsError

    if workflow is None or not thread_config:
        raise ValueError("Complete RAID analysis and PM review before exporting.")
    state, findings, interrupted = get_review_snapshot(workflow, thread_config)
    if state is None:
        raise ValueError("No workflow checkpoint is available for export.")
    if interrupted or not state.get("approval_complete", False):
        raise ValueError("Complete PM approval before exporting RAID findings.")
    if state.get("workflow_status") == "error":
        raise ValueError("Cannot export a failed RAID workflow.")
    manager = state.get("raid_state_manager")
    if manager is None:
        raise ValueError("RAID findings are missing from the workflow checkpoint.")
    export_service = service if service is not None else RAIDExportService()
    return export_service.prepare_download(manager)


def render_export_page() -> None:
    st.header("Export approved RAID findings")
    workflow = st.session_state.raid_workflow
    config = st.session_state.raid_thread_config
    if workflow is None or not config:
        st.info("Analyze documents and complete PM review before exporting.")
        return
    try:
        state, findings, interrupted = get_review_snapshot(workflow, config)
    except Exception as exc:
        st.error(f"Cannot load export status: {exc}")
        return
    if state is None:
        st.warning("No workflow checkpoint found.")
        return
    approved = [finding for finding in findings if finding.status.value == "approved"]
    st.write(f"Approved findings: {len(approved)} / {len(findings)}")
    if interrupted or not state.get("approval_complete", False):
        st.warning("Finish PM review and approval before exporting.")
        return
    if state.get("workflow_status") == "error":
        st.error("Workflow failed. Excel export is unavailable.")
        return
    if not approved:
        st.info("No approved findings to export. Rejected findings are excluded.")
        return

    # Keep prepared bytes across Streamlit reruns, but invalidate if checkpoint changes.
    snapshot_signature = tuple(sorted(
        (f.id, f.status.value, f.description, f.recommendations)
        for f in findings
    ))
    cached = st.session_state.raid_export_result
    if cached is not None and cached[0] != snapshot_signature:
        st.session_state.raid_export_result = None
        cached = None
    if st.button("Generate Excel export", key="raid_generate_export"):
        try:
            result = prepare_approved_export(workflow, config)
            st.session_state.raid_export_result = (snapshot_signature, result)
            cached = st.session_state.raid_export_result
        except Exception as exc:
            st.session_state.raid_export_result = None
            st.error(f"Excel export failed: {exc}")
            return
    if cached is not None:
        result = cached[1]
        st.success(result.message)
        st.download_button(
            "Download approved RAID Excel",
            data=result.data,
            file_name=result.filename,
            mime=result.mime_type,
            key="raid_download_excel",
        )
        st.caption("Only approved RAID findings are included in the workbook.")


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
