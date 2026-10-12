"""Task 6.4 cross-document synthesis tests (mocked provider, no API key)."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.workflow.langgraph_workflow import RAIDWorkflow, DocumentAnalysisResult
from src.document_processing.base import DocumentMetadata, ContentSection
from src.workflow.raid_state_manager import RAIDStateManager


def doc(name, text):
    return DocumentMetadata(filename=name, file_type="pdf", content_sections=[ContentSection(location="Page 1", text=text, metadata={})])


def state():
    a, b = doc("a.pdf", "Vendor delivery is delayed"), doc("b.pdf", "Vendor delivery is delayed")
    def result(d):
        return DocumentAnalysisResult(d, [{
            "type": "risk", "description": "Vendor delivery delay",
            "evidence_type": "fact", "source_citations": [{
                "filename": d.filename, "location": "Page 1",
                "excerpt": "Vendor delivery is delayed"}]}], "ok")
    return {"individual_analyses": [result(a), result(b)],
            "raid_state_manager": RAIDStateManager(), "error_messages": []}


def workflow():
    return RAIDWorkflow({"provider_type": "openai"})


def provider(payload):
    return SimpleNamespace(is_configured=lambda: True,
                           generate_response=lambda req: SimpleNamespace(content=json.dumps(payload)))


def test_merge_preserves_both_source_citations():
    w, s = workflow(), state()
    with patch("src.workflow.langgraph_workflow.llm_factory.get_provider", return_value=provider({
        "groups": [[0, 1]], "relationships": [], "contradictions": []})):
        output = w._perform_cross_document_synthesis(s)
    assert len(output.synthesized_findings) == 1
    assert {c.filename for c in output.synthesized_findings[0].source_citations} == {"a.pdf", "b.pdf"}


def test_cross_source_relationship_and_contradiction():
    w, s = workflow(), state()
    payload = {"groups": [], "relationships": [{"a": 0, "b": 1, "type": "dependency"}],
               "contradictions": [{"a": 0, "b": 1, "explanation": "Conflicting reports"}]}
    with patch("src.workflow.langgraph_workflow.llm_factory.get_provider", return_value=provider(payload)):
        output = w._perform_cross_document_synthesis(s)
    assert len(output.synthesized_findings) == 2
    assert len(output.relationships_identified) == 1
    assert len(output.contradictions_found) == 1
    assert len(output.contradictions_found[0]["source_citations"]) == 2


def test_invalid_indices_ignored():
    w, s = workflow(), state()
    payload = {"groups": [[0, 999]], "relationships": [{"a": 0, "b": 999, "type": "blocks"}],
               "contradictions": [{"a": 0, "b": 0, "explanation": "not cross-source"}]}
    with patch("src.workflow.langgraph_workflow.llm_factory.get_provider", return_value=provider(payload)):
        output = w._perform_cross_document_synthesis(s)
    assert len(output.synthesized_findings) == 2
    assert not output.relationships_identified and not output.contradictions_found


def test_provider_failure_preserves_candidates():
    w, s = workflow(), state()
    with patch("src.workflow.langgraph_workflow.llm_factory.get_provider", side_effect=RuntimeError("offline")):
        output = w._perform_cross_document_synthesis(s)
    assert len(output.synthesized_findings) == 2
    assert output.relationships_identified == []


def test_node_registers_relationships():
    w, s = workflow(), state()
    payload = {"groups": [], "relationships": [{"a": 0, "b": 1, "type": "related"}],
               "contradictions": []}
    with patch("src.workflow.langgraph_workflow.llm_factory.get_provider", return_value=provider(payload)):
        update = w.synthesize_cross_document(s)
    assert update["current_phase"] == "classification"
    assert len(s["raid_state_manager"].get_all_findings()) == 2
