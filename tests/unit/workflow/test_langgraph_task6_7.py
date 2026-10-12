"""Task 6.7: actual LangGraph interrupt/resume tests (no LLM required)."""
import uuid
import pytest
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.workflow.langgraph_workflow import RAIDWorkflow, WorkflowState
from src.workflow.raid_state_manager import RAIDStateManager
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation, Status


def setup_review(count=2):
    workflow = RAIDWorkflow({})
    graph = StateGraph(WorkflowState)
    graph.add_node("review_management", workflow.manage_review_approval)
    graph.set_entry_point("review_management")
    graph.add_node("review_round_boundary", lambda state: {})
    graph.add_conditional_edges(
        "review_management",
        lambda state: "done" if state["approval_complete"] else "review_again",
        {"done": END, "review_again": "review_round_boundary"},
    )
    graph.add_edge("review_round_boundary", "review_management")
    workflow.graph = graph.compile(checkpointer=workflow.checkpointer)
    manager = RAIDStateManager()
    findings = []
    for i in range(count):
        finding = RAIDFinding(type=RAIDType.RISK, description=f"Risk {i}",
                             evidence_type=EvidenceType.FACT,
                             source_citations=[SourceCitation("a.txt", "section 1", "evidence")])
        manager.add_finding(finding)
        findings.append(finding)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    state = {"raid_state_manager": manager, "session_id": session_id,
             "approval_complete": False, "pending_review_findings": [],
             "review_decisions": {}, "workflow_status": "in_progress",
             "current_phase": "review"}
    return workflow, config, state, findings


def test_interrupt_waits_for_pm():
    workflow, config, state, findings = setup_review()
    result = workflow.graph.invoke(state, config=config)
    assert result.get("__interrupt__")
    assert all(f.status == Status.PENDING for f in findings)
    assert workflow.graph.get_state(config).next == ("review_management",)


def test_explicit_approval_and_rejection():
    workflow, config, state, findings = setup_review()
    workflow.graph.invoke(state, config=config)
    result = workflow.resume_workflow(state["session_id"],
        {findings[0].id: "approved", findings[1].id: "rejected"})
    manager = result["state"]["raid_state_manager"]
    assert manager.get_finding(findings[0].id).status == Status.APPROVED
    assert manager.get_finding(findings[1].id).status == Status.REJECTED
    assert result["state"]["workflow_status"] == "completed"
    assert result["state"]["approval_complete"] is True


@pytest.mark.parametrize("decisions", [{}, {"unknown": "approved"}])
def test_invalid_decisions_do_not_resume(decisions):
    workflow, config, state, findings = setup_review()
    workflow.graph.invoke(state, config=config)
    with pytest.raises(ValueError):
        workflow.resume_workflow(state["session_id"], decisions)
    assert workflow.graph.get_state(config).next == ("review_management",)


def test_partial_review_can_resume_again():
    workflow, config, state, findings = setup_review()
    workflow.graph.invoke(state, config=config)
    first = workflow.resume_workflow(state["session_id"], {findings[0].id: "approved"})
    assert first["state"]["approval_complete"] is False
    assert first["state"]["workflow_status"] == "awaiting_review"
    assert first["state"]["raid_state_manager"].get_finding(findings[0].id).status == Status.APPROVED
    assert first["state"]["raid_state_manager"].get_finding(findings[1].id).status == Status.PENDING
    assert any(t.interrupts for t in workflow.graph.get_state(config).tasks)
    second = workflow.resume_workflow(state["session_id"], {findings[1].id: "rejected"})
    assert second["state"]["approval_complete"] is True
    assert second["state"]["workflow_status"] == "completed"
    assert second["state"]["raid_state_manager"].get_finding(findings[1].id).status == Status.REJECTED


def test_no_findings_finishes_without_interrupt():
    workflow, config, state, findings = setup_review(0)
    result = workflow.graph.invoke(state, config=config)
    assert not result.get("__interrupt__")
    assert result["workflow_status"] == "completed"


def test_cannot_resume_nonexistent_session():
    workflow, config, state, findings = setup_review()
    with pytest.raises(ValueError):
        workflow.resume_workflow(state["session_id"], {findings[0].id: "approved"})
