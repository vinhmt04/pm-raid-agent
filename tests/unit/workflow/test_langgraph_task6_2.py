"""Focused Task 6.2 tests; place under tests/unit/workflow/."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

try:
    from src.workflow import langgraph_workflow as module
except ImportError:
    from workflow import langgraph_workflow as module


def document(text="Vendor API delivery is delayed"):
    return SimpleNamespace(filename="plan.docx", content_sections=[
        SimpleNamespace(location="section_1", text=text, metadata={})
    ])


def analyze(monkeypatch, response):
    provider = Mock()
    provider.is_configured.return_value = True
    provider.generate_response.return_value = SimpleNamespace(content=response)
    monkeypatch.setattr(module.llm_factory, "get_provider", lambda: provider)
    workflow = object.__new__(module.RAIDWorkflow)
    result = workflow._analyze_single_document(document(), {})
    return result, provider


def test_extracts_grounded_raid_with_real_location(monkeypatch):
    response = '{"findings":[{"type":"dependency","description":"Vendor API delayed",' \
               '"evidence_type":"fact","excerpt":"Vendor API delivery is delayed",' \
               '"impact_assessment":"Release impact","recommendations":"Escalate"}]}'
    result, provider = analyze(monkeypatch, response)
    assert not result.processing_errors
    assert len(result.potential_findings) == 1
    assert result.potential_findings[0]["source_citations"] == [{
        "filename": "plan.docx", "location": "section_1",
        "excerpt": "Vendor API delivery is delayed"
    }]
    assert "Vendor API delivery is delayed" in provider.generate_response.call_args.args[0].messages[-1]["content"]


def test_rejects_hallucinated_excerpt(monkeypatch):
    response = '{"findings":[{"type":"risk","description":"Invented",' \
               '"evidence_type":"inference","excerpt":"Nonexistent quote"}]}'
    result, _ = analyze(monkeypatch, response)
    assert result.potential_findings == []
    assert result.processing_errors


def test_empty_findings(monkeypatch):
    result, _ = analyze(monkeypatch, '{"findings":[]}')
    assert result.potential_findings == []
    assert not result.processing_errors


def test_malformed_json_reports_error(monkeypatch):
    result, _ = analyze(monkeypatch, 'not json')
    assert result.potential_findings == []
    assert result.processing_errors


def test_unconfigured_provider_reports_error(monkeypatch):
    provider = Mock()
    provider.is_configured.return_value = False
    monkeypatch.setattr(module.llm_factory, "get_provider", lambda: provider)
    result = object.__new__(module.RAIDWorkflow)._analyze_single_document(document(), {})
    assert result.processing_errors
    assert result.potential_findings == []
