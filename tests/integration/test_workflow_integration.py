"""
Integration tests for LangGraph workflow with other components.
Tests end-to-end workflow integration and compatibility.
"""

import pytest
from datetime import datetime
from typing import List

from src.workflow.langgraph_workflow import RAIDWorkflow, WorkflowState
from src.workflow.raid_state_manager import RAIDStateManager
from src.document_processing.base import DocumentMetadata, ContentSection
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation


@pytest.fixture
def sample_document_metadata():
    """Create sample document metadata for integration testing."""
    return [
        DocumentMetadata(
            filename="project_charter.pdf",
            file_type="pdf",
            content_sections=[
                ContentSection(
                    location="Executive Summary",
                    text="The project aims to deliver a new customer portal by Q2 2024. Key risks include timeline constraints and resource availability.",
                    metadata={"section": "executive_summary", "confidence": 0.9}
                ),
                ContentSection(
                    location="Risk Assessment - Page 3",
                    text="Budget overrun risk is high due to uncertain third-party integration costs. Timeline dependency on vendor delivery.",
                    metadata={"page": 3, "section": "risks"}
                )
            ]
        ),
        DocumentMetadata(
            filename="requirements_spec.docx",
            file_type="docx",
            content_sections=[
                ContentSection(
                    location="Section 2.1 - Assumptions",
                    text="We assume the legacy system API will remain stable throughout the project. Customer data migration is expected to complete within 2 weeks.",
                    metadata={"section": "2.1", "heading": "Assumptions"}
                ),
                ContentSection(
                    location="Section 4.3 - Dependencies",
                    text="Project success depends on completion of infrastructure upgrade by IT team. Third-party authentication service must be available.",
                    metadata={"section": "4.3", "heading": "Dependencies"}
                )
            ]
        )
    ]


@pytest.fixture
def workflow_config():
    """Create integration test LLM provider configuration."""
    return {
        "provider_type": "openai",
        "api_key": "test_integration_key",
        "model": "gpt-3.5-turbo",
        "timeout": 30
    }


class TestWorkflowIntegration:
    """Integration tests for the complete workflow system."""
    
    def test_full_workflow_initialization(self, workflow_config, sample_document_metadata):
        """Test complete workflow initialization with real components."""
        # Create workflow
        workflow = RAIDWorkflow(workflow_config)
        
        # Start analysis
        session_id = "integration_test_001"
        result = workflow.start_analysis(sample_document_metadata, session_id)
        
        # Verify workflow components are properly integrated
        assert "initial_state" in result
        assert "thread_config" in result
        assert "graph" in result
        
        initial_state = result["initial_state"]
        
        # Verify state structure
        assert initial_state["session_id"] == session_id
        assert len(initial_state["documents"]) == 2
        assert isinstance(initial_state["raid_state_manager"], RAIDStateManager)
        assert initial_state["current_phase"] == "individual_analysis"
        assert initial_state["workflow_status"] == "in_progress"
        
        # Verify documents are properly integrated
        for i, doc in enumerate(initial_state["documents"]):
            assert isinstance(doc, DocumentMetadata)
            assert doc.filename == sample_document_metadata[i].filename
            assert len(doc.content_sections) > 0
    
    def test_workflow_state_management_integration(self, workflow_config, sample_document_metadata):
        """Test that workflow state integrates properly with RAID state manager."""
        workflow = RAIDWorkflow(workflow_config)
        session_id = "integration_test_002"
        
        # Start workflow and get initial state
        result = workflow.start_analysis(sample_document_metadata, session_id)
        initial_state = result["initial_state"]
        
        # Create test findings and add to RAID state manager
        test_finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Budget overrun risk due to third-party costs",
            evidence_type=EvidenceType.FACT,
            source_citations=[
                SourceCitation(
                    filename="project_charter.pdf",
                    location="Risk Assessment - Page 3",
                    excerpt="Budget overrun risk is high due to uncertain third-party integration costs"
                )
            ],
            impact_assessment="High financial impact on project budget",
            recommendations="Establish contingency fund and negotiate fixed-price contracts"
        )
        
        # Add finding to state manager
        raid_state_manager = initial_state["raid_state_manager"]
        finding_id = raid_state_manager.add_finding(test_finding)
        
        # Verify integration
        assert finding_id in raid_state_manager
        assert len(raid_state_manager) == 1
        
        retrieved_finding = raid_state_manager.get_finding(finding_id)
        assert retrieved_finding is not None
        assert retrieved_finding.type == RAIDType.RISK
        assert len(retrieved_finding.source_citations) == 1
        assert retrieved_finding.source_citations[0].filename == "project_charter.pdf"
    
    def test_workflow_phase_transitions(self, workflow_config, sample_document_metadata):
        """Validate node transitions; review must run in a compiled graph."""
        workflow = RAIDWorkflow(workflow_config)
        state = workflow.start_analysis(sample_document_metadata, "integration_test_003")["initial_state"]
        updates = workflow.analyze_individual_documents(state)
        assert updates["current_phase"] == "synthesis"
        state.update(updates)
        updates = workflow.synthesize_cross_document(state)
        assert updates["current_phase"] == "classification"
        state.update(updates)
        updates = workflow.classify_evidence(state)
        assert updates["current_phase"] == "review"
        # No synthetic findings: with no LLM evidence, review can legitimately
        # finish with zero pending items. Actual interrupt is tested separately.

    def test_cross_document_analysis_integration(self, workflow_config, sample_document_metadata):
        """Test cross-document analysis integrates properly with multiple documents."""
        workflow = RAIDWorkflow(workflow_config)
        session_id = "integration_test_004"
        
        # Start workflow
        result = workflow.start_analysis(sample_document_metadata, session_id)
        state = result["initial_state"]
        
        # Stub source-grounded document extraction; no live LLM/API dependency.
        from src.workflow.langgraph_workflow import DocumentAnalysisResult
        def fake_analyze(document, _state):
            section = document.content_sections[0]
            return DocumentAnalysisResult(
                document_metadata=document,
                potential_findings=[{
                    "type": "risk",
                    "description": f"Risk documented in {document.filename}",
                    "evidence_type": "fact",
                    "source_citations": [{"filename": document.filename,
                                          "location": section.location,
                                          "excerpt": section.text[:60]}],
                }],
                analysis_summary="Fixture finding",
            )
        workflow._analyze_single_document = fake_analyze

        # Run individual analysis
        individual_updates = workflow.analyze_individual_documents(state)
        state.update(individual_updates)
        
        # Verify individual analyses created
        individual_analyses = state["individual_analyses"]
        assert len(individual_analyses) == 2  # One per document
        
        # Check that analyses reference correct documents
        for i, analysis in enumerate(individual_analyses):
            assert analysis.document_metadata.filename == sample_document_metadata[i].filename
            assert len(analysis.potential_findings) > 0  # Should have some findings
        
        # Run cross-document synthesis
        synthesis_updates = workflow.synthesize_cross_document(state)
        state.update(synthesis_updates)
        
        # Verify cross-document synthesis results
        synthesis = state["cross_document_synthesis"]
        assert synthesis is not None
        assert len(synthesis.synthesized_findings) > 0
        
        # Verify findings were added to RAID state manager
        raid_manager = state["raid_state_manager"]
        all_findings = raid_manager.get_all_findings()
        assert len(all_findings) == len(synthesis.synthesized_findings)
    
    def test_approval_workflow_integration(self, workflow_config):
        """Review interrupts and resumes with explicit, per-finding decisions."""
        from langgraph.graph import StateGraph, END
        workflow = RAIDWorkflow(workflow_config)
        graph = StateGraph(WorkflowState)
        graph.add_node("review_management", workflow.manage_review_approval)
        graph.set_entry_point("review_management")
        graph.add_edge("review_management", END)
        workflow.graph = graph.compile(checkpointer=workflow.checkpointer)
        manager = RAIDStateManager()
        finding1 = RAIDFinding(type=RAIDType.RISK, description="Test risk 1",
            evidence_type=EvidenceType.FACT,
            source_citations=[SourceCitation("test1.pdf", "Page 1", "Sample text")])
        finding2 = RAIDFinding(type=RAIDType.ASSUMPTION, description="Test assumption 1",
            evidence_type=EvidenceType.INFERENCE,
            source_citations=[SourceCitation("test2.docx", "Section 2", "Sample text")])
        manager.add_finding(finding1)
        manager.add_finding(finding2)
        session_id = "integration_test_005"
        config = {"configurable": {"thread_id": session_id}}
        paused = workflow.graph.invoke({"raid_state_manager": manager, "session_id": session_id}, config=config)
        assert paused.get("__interrupt__")
        assert workflow.graph.get_state(config).next == ("review_management",)
        resumed = workflow.resume_workflow(session_id, {finding1.id: "approved", finding2.id: "rejected"})
        saved = resumed["state"]["raid_state_manager"]
        assert resumed["state"]["workflow_status"] == "completed"
        assert saved.get_finding(finding1.id).status.value == "approved"
        assert saved.get_finding(finding2.id).status.value == "rejected"
        assert [f.id for f in saved.get_approved_findings()] == [finding1.id]

    def test_session_checkpointing_integration(self, workflow_config, sample_document_metadata):
        """Checkpoint is only resumable after an actual graph interrupt."""
        workflow = RAIDWorkflow(workflow_config)
        result = workflow.start_analysis(sample_document_metadata, "integration_test_006")
        assert result["thread_config"]["configurable"]["thread_id"] == "integration_test_006"
        with pytest.raises(ValueError, match="No pending review interrupt"):
            workflow.resume_workflow("integration_test_006", {"unknown": "approved"})

    def test_error_handling_integration(self, workflow_config):
        """Test error handling integrates properly across components."""
        workflow = RAIDWorkflow(workflow_config)
        
        # Test with invalid document metadata
        invalid_documents = [None]  # This should cause errors
        session_id = "integration_test_007"
        
        # This should handle the error gracefully
        try:
            result = workflow.start_analysis(invalid_documents, session_id)
            
            # If it doesn't raise an exception, verify error handling in state
            initial_state = result["initial_state"]
            assert "error_messages" in initial_state
            
        except Exception as e:
            # Error handling worked - this is expected behavior
            assert "documents" in str(e).lower() or "none" in str(e).lower()
    
    def test_workflow_statistics_integration(self, workflow_config):
        """Test workflow statistics integrate with RAID state management."""
        workflow = RAIDWorkflow(workflow_config)
        session_id = "integration_test_008"
        
        # Get statistics for non-existent session
        stats = workflow.get_workflow_statistics(session_id)
        assert "error" in stats or "has_checkpoint" in stats
        
        # Start a workflow
        result = workflow.start_analysis([], session_id)
        
        # Get statistics for existing session
        stats = workflow.get_workflow_statistics(session_id)
        assert "session_id" in stats or "error" in stats
        assert stats.get("session_id") == session_id or "error" in stats