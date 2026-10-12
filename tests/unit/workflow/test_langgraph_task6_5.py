"""Task 6.5: evidence classification tests (mock LLM, no API calls)."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.workflow.langgraph_workflow import RAIDWorkflow
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation, Status


def finding(description='Delivery is delayed', evidence_type=EvidenceType.UNCERTAINTY, count=1):
    return RAIDFinding(
        type=RAIDType.ISSUE,
        description=description,
        evidence_type=evidence_type,
        source_citations=[SourceCitation(filename=f'doc{i}.txt', location='section 1',
                                         excerpt='The delivery milestone was missed.')
                          for i in range(count)],
    )


def classify(findings, responses, configured=True):
    workflow = RAIDWorkflow({})
    provider = SimpleNamespace(
        is_configured=lambda: configured,
        generate_response=lambda request: SimpleNamespace(content=next(responses)),
    )
    state = {'raid_state_manager': SimpleNamespace(get_all_findings=lambda: findings)}
    with patch('src.workflow.langgraph_workflow.llm_factory.get_provider', return_value=provider):
        result = workflow._classify_finding_evidence(state)
    return result


def test_single_citation_can_be_fact():
    f = finding()
    result = classify([f], iter([json.dumps({'evidence_type': 'fact', 'reasoning': 'Explicit'})]))
    assert result == {f.id: 'fact'}
    assert f.metadata['evidence_classification_reasoning'] == 'Explicit'


def test_multiple_citations_can_be_uncertain():
    f = finding(count=2)
    assert classify([f], iter(['{"evidence_type":"uncertainty"}']))[f.id] == 'uncertainty'


def test_inference_label():
    f = finding()
    assert classify([f], iter(['{"evidence_type":"inference"}']))[f.id] == 'inference'


def test_invalid_response_preserves_original_type():
    f = finding(evidence_type=EvidenceType.FACT)
    assert classify([f], iter(['{"evidence_type":"unsupported"}'])) == {}
    assert f.evidence_type == EvidenceType.FACT


def test_provider_not_configured_preserves_original_type():
    f = finding(evidence_type=EvidenceType.INFERENCE)
    assert classify([f], iter([]), configured=False) == {}
    assert f.evidence_type == EvidenceType.INFERENCE


def test_one_failure_does_not_block_next_finding():
    first, second = finding(), finding()
    result = classify([first, second], iter(['invalid JSON', '{"evidence_type":"fact"}']))
    assert first.id not in result
    assert result[second.id] == 'fact'


def test_no_findings_requires_no_provider():
    workflow = RAIDWorkflow({})
    state = {'raid_state_manager': SimpleNamespace(get_all_findings=lambda: [])}
    assert workflow._classify_finding_evidence(state) == {}
