"""Task 11.1 — Streamlit/LangGraph integration regression tests."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from src.ui.app import start_raid_analysis, run_analysis_in_session


def workflow_factory(state=None, *, interrupted=True, analyses=None):
    workflow = Mock()
    if state is None:
        state = {"workflow_status": "awaiting_review", "raid_state_manager": Mock(),
                 "individual_analyses": analyses or []}
    workflow.start_analysis.return_value = {
        "initial_state": {"documents": ["doc"]},
        "thread_config": {"configurable": {"thread_id": "session-1"}},
    }
    workflow.graph.get_state.return_value = SimpleNamespace(
        values=state,
        tasks=[SimpleNamespace(interrupts=["pending"])] if interrupted else [],
    )
    if state.get("raid_state_manager") is not None:
        state["raid_state_manager"].get_all_findings.return_value = ["finding"]
    return workflow


def test_analysis_runs_graph_and_reads_checkpoint():
    workflow = workflow_factory()
    result = start_raid_analysis(["doc"], "session-1", lambda config: workflow)
    assert result[0] is workflow
    assert result[3] == ["finding"]
    assert result[4] is True
    workflow.graph.invoke.assert_called_once_with(
        {"documents": ["doc"]},
        config={"configurable": {"thread_id": "session-1"}},
    )


def test_session_updates_only_after_success():
    workflow = workflow_factory()
    session = {"raid_workflow": None, "raid_export_result": "stale"}
    run_analysis_in_session(["doc"], "session-1", session, lambda config: workflow)
    assert session["raid_workflow"] is workflow
    assert session["raid_export_result"] is None
    assert session["raid_review_decisions"] == {}


def test_empty_input_does_not_initialize_workflow():
    factory = Mock()
    with pytest.raises(ValueError, match="at least one"):
        start_raid_analysis([], "session-1", factory)
    factory.assert_not_called()


def test_graph_error_does_not_persist_session():
    workflow = workflow_factory()
    workflow.graph.invoke.side_effect = RuntimeError("LLM timeout")
    session = {"raid_workflow": None}
    with pytest.raises(RuntimeError, match="LLM timeout"):
        run_analysis_in_session(["doc"], "session-1", session, lambda config: workflow)
    assert session["raid_workflow"] is None


def test_checkpoint_error_is_not_treated_as_success():
    workflow = workflow_factory({"workflow_status": "error", "error_messages": ["bad response"]})
    with pytest.raises(RuntimeError, match="bad response"):
        start_raid_analysis(["doc"], "session-1", lambda config: workflow)


def test_all_document_llm_failures_are_reported():
    analysis = SimpleNamespace(processing_errors=["LLM provider is not configured"])
    workflow = workflow_factory(analyses=[analysis])
    with pytest.raises(RuntimeError, match="not configured"):
        start_raid_analysis(["doc"], "session-1", lambda config: workflow)
