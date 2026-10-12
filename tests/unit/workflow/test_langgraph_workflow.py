"""
Unit tests for LangGraph workflow orchestration.
Tests the workflow state management and node structure.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from typing import List

from src.workflow.langgraph_workflow import (
    RAIDWorkflow, 
    WorkflowState, 
    DocumentAnalysisResult, 
    CrossDocumentSynthesis
)
from src.document_processing.base import DocumentMetadata, ContentSection
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation


@pytest.fixture
def sample_documents():
    """Create sample document metadata for testing."""
    return [
        DocumentMetadata(
            filename="test_doc1.pdf",
            file_type="pdf",
            content_sections=[
                ContentSection(
                    location="Page 1",
                    text="This is a sample project document with potential risks.",
                    metadata={"page_number": 1}
                )
            ]
        ),
        DocumentMetadata(
            filename="test_doc2.docx",
            file_type="docx",
            content_sections=[
                ContentSection(
                    location="Section 2",
                    text="Project assumptions and dependencies are documented here.",
                    metadata={"section": "Requirements"}
                )
            ]
        )
    ]


@pytest.fixture
def workflow_config():
    """Create test LLM provider configuration."""
    return {
        "provider_type": "openai",
        "api_key": "test_key",
        "model": "gpt-3.5-turbo"
    }


@pytest.fixture
def raid_workflow(workflow_config):
    """Create RAIDWorkflow instance for testing."""
    return RAIDWorkflow(workflow_config)


class TestRAIDWorkflow:
    """Test cases for RAIDWorkflow class."""
    
    def test_workflow_initialization(self, workflow_config):
        """Test workflow initializes with correct configuration."""
        workflow = RAIDWorkflow(workflow_config)
        
        assert workflow.llm_provider_config == workflow_config
        assert workflow.checkpointer is not None
        assert workflow.graph is not None
    
    def test_start_analysis(self, raid_workflow, sample_documents):
        """Test starting a new analysis workflow."""
        session_id = "test_session_123"
        
        result = raid_workflow.start_analysis(sample_documents, session_id)
        
        # Verify result structure
        assert "initial_state" in result
        assert "thread_config" in result
        assert "graph" in result
        
        # Verify initial state
        initial_state = result["initial_state"]
        assert initial_state["session_id"] == session_id
        assert len(initial_state["documents"]) == len(sample_documents)
        assert initial_state["current_phase"] == "individual_analysis"
        assert initial_state["workflow_status"] == "in_progress"
        assert initial_state["raid_state_manager"] is not None
        
        # Verify thread config
        thread_config = result["thread_config"]
        assert thread_config["configurable"]["thread_id"] == session_id
    
    def test_resume_workflow(self, raid_workflow):
        """Test resuming workflow with review decisions."""
        session_id = "test_session_123"
        review_decisions = {
            "finding_1": "approved",
            "finding_2": "rejected"
        }
        
        result = raid_workflow.resume_workflow(session_id, review_decisions)
        
        # Verify result structure
        assert "thread_config" in result
        assert "update_state" in result
        
        # Verify update state
        update_state = result["update_state"]
        assert update_state["review_decisions"] == review_decisions
        assert update_state["approval_complete"] is True
        assert "last_updated_at" in update_state
    
    def test_analyze_individual_documents_node(self, raid_workflow, sample_documents):
        """Test individual document analysis node."""
        # Create workflow state
        state = WorkflowState(
            documents=sample_documents,
            individual_analyses=[],
            cross_document_synthesis=None,
            raid_state_manager=Mock(),
            current_phase="individual_analysis",
            workflow_status="in_progress",
            error_messages=[],
            llm_provider_config=raid_workflow.llm_provider_config,
            llm_conversation_history=[],
            session_id="test_session",
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[],
            review_decisions={},
            approval_complete=False
        )
        
        # Execute node
        updates = raid_workflow.analyze_individual_documents(state)
        
        # Verify updates
        assert "individual_analyses" in updates
        assert "current_phase" in updates
        assert updates["current_phase"] == "synthesis"
        assert len(updates["individual_analyses"]) == len(sample_documents)
        
        # Verify analysis results
        for analysis in updates["individual_analyses"]:
            assert isinstance(analysis, DocumentAnalysisResult)
            assert analysis.document_metadata is not None
            assert isinstance(analysis.potential_findings, list)
            assert isinstance(analysis.analysis_summary, str)
    
    def test_synthesize_cross_document_node(self, raid_workflow, sample_documents):
        """Test cross-document synthesis node."""
        from src.workflow.raid_state_manager import RAIDStateManager
        
        # Create individual analyses
        individual_analyses = [
            DocumentAnalysisResult(
                document_metadata=sample_documents[0],
                potential_findings=[{
                    "type": "risk",
                    "description": "Test risk finding",
                    "evidence_type": "inference",
                    "source_citations": [{
                        "filename": "test_doc1.pdf",
                        "location": "Page 1",
                        "excerpt": "Sample excerpt"
                    }]
                }],
                analysis_summary="Test analysis"
            )
        ]
        
        # Create workflow state
        state = WorkflowState(
            documents=sample_documents,
            individual_analyses=individual_analyses,
            cross_document_synthesis=None,
            raid_state_manager=RAIDStateManager(),
            current_phase="synthesis",
            workflow_status="in_progress",
            error_messages=[],
            llm_provider_config=raid_workflow.llm_provider_config,
            llm_conversation_history=[],
            session_id="test_session",
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[],
            review_decisions={},
            approval_complete=False
        )
        
        # Execute node
        updates = raid_workflow.synthesize_cross_document(state)
        
        # Verify updates
        assert "cross_document_synthesis" in updates
        assert "current_phase" in updates
        assert updates["current_phase"] == "classification"
        
        synthesis = updates["cross_document_synthesis"]
        assert isinstance(synthesis, CrossDocumentSynthesis)
        assert isinstance(synthesis.synthesized_findings, list)
        assert len(synthesis.synthesized_findings) > 0
    
    def test_classify_evidence_node(self, raid_workflow):
        """Test evidence classification node."""
        from src.workflow.raid_state_manager import RAIDStateManager
        
        # Create RAID state manager with a test finding
        raid_state_manager = RAIDStateManager()
        test_finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Test risk finding",
            evidence_type=EvidenceType.INFERENCE,
            source_citations=[SourceCitation("test.pdf", "Page 1", "Sample text")]
        )
        raid_state_manager.add_finding(test_finding)
        
        # Create workflow state
        state = WorkflowState(
            documents=[],
            individual_analyses=[],
            cross_document_synthesis=None,
            raid_state_manager=raid_state_manager,
            current_phase="classification",
            workflow_status="in_progress",
            error_messages=[],
            llm_provider_config=raid_workflow.llm_provider_config,
            llm_conversation_history=[],
            session_id="test_session",
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[],
            review_decisions={},
            approval_complete=False
        )
        
        # Execute node
        updates = raid_workflow.classify_evidence(state)
        
        # Verify updates
        assert "current_phase" in updates
        assert updates["current_phase"] == "review"
        assert "last_updated_at" in updates
    
    def test_manage_review_approval_initial(self, raid_workflow):
        """Test review and approval management node - initial setup."""
        from src.workflow.raid_state_manager import RAIDStateManager
        
        # Create RAID state manager with pending findings
        raid_state_manager = RAIDStateManager()
        test_finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Test risk finding",
            evidence_type=EvidenceType.INFERENCE,
            source_citations=[SourceCitation("test.pdf", "Page 1", "Sample text")]
        )
        raid_state_manager.add_finding(test_finding)
        
        # Create workflow state (initial review setup)
        state = WorkflowState(
            documents=[],
            individual_analyses=[],
            cross_document_synthesis=None,
            raid_state_manager=raid_state_manager,
            current_phase="review",
            workflow_status="in_progress",
            error_messages=[],
            llm_provider_config=raid_workflow.llm_provider_config,
            llm_conversation_history=[],
            session_id="test_session",
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[],
            review_decisions={},
            approval_complete=False
        )
        
        # Execute node
        updates = raid_workflow.manage_review_approval(state)
        
        # Verify updates
        assert "pending_review_findings" in updates
        assert "workflow_status" in updates
        assert updates["workflow_status"] == "awaiting_review"
        assert len(updates["pending_review_findings"]) == 1
    
    def test_manage_review_approval_processing(self, raid_workflow):
        """Test review and approval management node - processing decisions."""
        from src.workflow.raid_state_manager import RAIDStateManager
        
        # Create RAID state manager with pending findings
        raid_state_manager = RAIDStateManager()
        test_finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Test risk finding",
            evidence_type=EvidenceType.INFERENCE,
            source_citations=[SourceCitation("test.pdf", "Page 1", "Sample text")]
        )
        finding_id = raid_state_manager.add_finding(test_finding)
        
        # Create workflow state (processing decisions)
        state = WorkflowState(
            documents=[],
            individual_analyses=[],
            cross_document_synthesis=None,
            raid_state_manager=raid_state_manager,
            current_phase="review",
            workflow_status="awaiting_review",
            error_messages=[],
            llm_provider_config=raid_workflow.llm_provider_config,
            llm_conversation_history=[],
            session_id="test_session",
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[finding_id],
            review_decisions={finding_id: "approved"},
            approval_complete=True
        )
        
        # Execute node
        updates = raid_workflow.manage_review_approval(state)
        
        # Verify updates
        assert "workflow_status" in updates
        assert "current_phase" in updates
        assert updates["workflow_status"] == "completed"
        assert updates["current_phase"] == "completed"
        
        # Verify finding was approved
        finding = raid_state_manager.get_finding(finding_id)
        assert finding.status.value == "approved"


class TestWorkflowState:
    """Test cases for WorkflowState TypedDict structure."""
    
    def test_workflow_state_structure(self, sample_documents):
        """Test that WorkflowState has the expected structure."""
        from src.workflow.raid_state_manager import RAIDStateManager
        
        # Create a complete workflow state
        state = WorkflowState(
            documents=sample_documents,
            individual_analyses=[],
            cross_document_synthesis=None,
            raid_state_manager=RAIDStateManager(),
            current_phase="individual_analysis",
            workflow_status="in_progress",
            error_messages=[],
            llm_provider_config={"provider": "test"},
            llm_conversation_history=[],
            session_id="test_session",
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[],
            review_decisions={},
            approval_complete=False
        )
        
        # Verify all required fields are present
        required_fields = [
            "documents", "individual_analyses", "cross_document_synthesis",
            "raid_state_manager", "current_phase", "workflow_status",
            "error_messages", "llm_provider_config", "llm_conversation_history",
            "session_id", "workflow_started_at", "last_updated_at",
            "pending_review_findings", "review_decisions", "approval_complete"
        ]
        
        for field in required_fields:
            assert field in state


class TestDocumentAnalysisResult:
    """Test cases for DocumentAnalysisResult dataclass."""
    
    def test_document_analysis_result_creation(self, sample_documents):
        """Test creating DocumentAnalysisResult."""
        result = DocumentAnalysisResult(
            document_metadata=sample_documents[0],
            potential_findings=[
                {
                    "type": "risk",
                    "description": "Test finding",
                    "evidence_type": "fact"
                }
            ],
            analysis_summary="Test summary"
        )
        
        assert result.document_metadata == sample_documents[0]
        assert len(result.potential_findings) == 1
        assert result.analysis_summary == "Test summary"
        assert result.processing_errors == []


class TestCrossDocumentSynthesis:
    """Test cases for CrossDocumentSynthesis dataclass."""
    
    def test_cross_document_synthesis_creation(self):
        """Test creating CrossDocumentSynthesis."""
        test_finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Test risk",
            evidence_type=EvidenceType.FACT,
            source_citations=[SourceCitation("test.pdf", "Page 1", "Sample text")]
        )
        
        synthesis = CrossDocumentSynthesis(
            synthesized_findings=[test_finding],
            relationships_identified=[{"finding_id1": "id1", "finding_id2": "id2"}],
            contradictions_found=[],
            synthesis_summary="Test synthesis"
        )
        
        assert len(synthesis.synthesized_findings) == 1
        assert len(synthesis.relationships_identified) == 1
        assert synthesis.contradictions_found == []
        assert synthesis.synthesis_summary == "Test synthesis"