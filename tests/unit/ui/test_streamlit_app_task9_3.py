"""Task 9.3: PM review state, edits and approval controls."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.ui.app import apply_review_edits, get_review_snapshot, submit_review, validate_review_decisions
from src.workflow.raid_state_manager import RAIDStateManager
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation, Status


def finding():
    return RAIDFinding(type=RAIDType.RISK, description="Delivery risk",
                       evidence_type=EvidenceType.FACT,
                       source_citations=[SourceCitation("plan.pdf", "Page 2", "Late vendor delivery")])


def test_partial_decisions_and_valid_labels():
    a, b = finding(), finding()
    assert validate_review_decisions([a, b], {a.id: "approved"}) == {a.id: "approved"}
    with pytest.raises(ValueError):
        validate_review_decisions([a], {a.id: "pending"})
    assert validate_review_decisions([a], {a.id: "rejected"}) == {a.id: "rejected"}


def test_edits_do_not_approve_or_modify_citations():
    manager = RAIDStateManager()
    item = finding()
    manager.add_finding(item)
    citations = deepcopy(item.source_citations)
    apply_review_edits(manager, {item.id: {"description": "Updated risk", "recommendations": "Escalate"}})
    assert item.description == "Updated risk"
    assert item.recommendations == "Escalate"
    assert item.status == Status.PENDING
    assert item.source_citations == citations


def test_reject_invalid_edits():
    manager = RAIDStateManager()
    item = finding()
    manager.add_finding(item)
    with pytest.raises(ValueError):
        apply_review_edits(manager, {item.id: {"status": "approved"}})
    with pytest.raises(ValueError):
        apply_review_edits(manager, {item.id: {"description": " "}})


def test_checkpoint_snapshot_recognizes_interrupt():
    manager = RAIDStateManager()
    item = finding()
    manager.add_finding(item)
    workflow = Mock()
    workflow.graph.get_state.return_value = SimpleNamespace(
        values={"raid_state_manager": manager}, tasks=[SimpleNamespace(interrupts=["review"])])
    state, findings, interrupted = get_review_snapshot(workflow, {"configurable": {"thread_id": "x"}})
    assert state is not None
    assert findings == [item]
    assert interrupted is True


def test_submit_review_resumes_and_persists_edits():
    manager = RAIDStateManager()
    item = finding()
    manager.add_finding(item)
    before = {"raid_state_manager": manager}
    workflow = Mock()
    workflow.graph.get_state.side_effect = [
        SimpleNamespace(values=before, tasks=[SimpleNamespace(interrupts=["review"])]),
        SimpleNamespace(values={"raid_state_manager": manager}, tasks=[]),
    ]
    def resume(session_id, decisions):
        assert decisions == {item.id: "approved"}
        manager.get_finding(item.id).approve()
        return {"state": {"raid_state_manager": manager}}
    workflow.resume_workflow.side_effect = resume
    result = submit_review(workflow, "session", {item.id: "approved"},
                           {item.id: {"description": "Updated risk"}})
    assert result["raid_state_manager"].get_finding(item.id).status == Status.APPROVED
    assert item.description == "Updated risk"
    workflow.graph.update_state.assert_called_once()


def test_submit_review_does_not_resume_when_invalid():
    manager = RAIDStateManager()
    item = finding()
    manager.add_finding(item)
    workflow = Mock()
    workflow.graph.get_state.return_value = SimpleNamespace(
        values={"raid_state_manager": manager}, tasks=[SimpleNamespace(interrupts=["review"])])
    with pytest.raises(ValueError):
        submit_review(workflow, "session", {}, {})
    workflow.resume_workflow.assert_not_called()
