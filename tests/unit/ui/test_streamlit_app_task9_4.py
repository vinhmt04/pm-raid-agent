"""Task 9.4 — approval-gated Excel export UI service integration."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from src.ui.app import prepare_approved_export
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation
from src.workflow.raid_state_manager import RAIDStateManager
from src.export.export_service import RAIDExportService, NoApprovedFindingsError, EXCEL_MIME_TYPE


def make_finding():
    return RAIDFinding(type=RAIDType.RISK, description="Schedule risk",
                       evidence_type=EvidenceType.FACT,
                       source_citations=[SourceCitation("plan.pdf", "Page 1", "Vendor delay")])


def checkpoint(manager, *, approved=True, interrupted=False, status="completed"):
    workflow = Mock()
    workflow.graph.get_state.return_value = SimpleNamespace(
        values={"raid_state_manager": manager, "approval_complete": approved,
                "workflow_status": status},
        tasks=[SimpleNamespace(interrupts=["review"])] if interrupted else [],
    )
    return workflow


CONFIG = {"configurable": {"thread_id": "test"}}


def test_export_contains_only_approved_findings():
    manager = RAIDStateManager()
    accepted, rejected = make_finding(), make_finding()
    accepted.approve()
    rejected.reject()
    manager.add_finding(accepted)
    manager.add_finding(rejected)
    result = prepare_approved_export(checkpoint(manager), CONFIG)
    assert result.approved_count == 1
    assert result.filename.endswith(".xlsx")
    assert result.mime_type == EXCEL_MIME_TYPE
    assert result.data[:2] == b"PK"


@pytest.mark.parametrize("approved,interrupted,status", [
    (False, False, "completed"),
    (True, True, "completed"),
    (True, False, "error"),
])
def test_export_blocked_without_completed_approval(approved, interrupted, status):
    manager = RAIDStateManager()
    item = make_finding()
    item.approve()
    manager.add_finding(item)
    with pytest.raises(ValueError):
        prepare_approved_export(checkpoint(manager, approved=approved,
                                           interrupted=interrupted, status=status), CONFIG)


def test_export_rejects_no_approved_findings():
    manager = RAIDStateManager()
    manager.add_finding(make_finding())
    with pytest.raises(NoApprovedFindingsError):
        prepare_approved_export(checkpoint(manager), CONFIG)


def test_export_rejects_missing_workflow_or_checkpoint():
    with pytest.raises(ValueError, match="Complete RAID"):
        prepare_approved_export(None, None)
    workflow = Mock()
    workflow.graph.get_state.return_value = SimpleNamespace(values={}, tasks=[])
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_approved_export(workflow, CONFIG)


def test_export_service_error_propagates():
    manager = RAIDStateManager()
    item = make_finding()
    item.approve()
    manager.add_finding(item)
    service = Mock(spec=RAIDExportService)
    service.prepare_download.side_effect = RuntimeError("disk error")
    with pytest.raises(RuntimeError, match="disk error"):
        prepare_approved_export(checkpoint(manager), CONFIG, service=service)
