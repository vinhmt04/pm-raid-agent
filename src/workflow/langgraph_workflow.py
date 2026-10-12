"""
LangGraph workflow orchestration for PM RAID Agent MVP.
Manages the multi-phase RAID analysis pipeline through discrete workflow nodes.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import sys
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Handle imports based on execution context
try:
    from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, Status, SourceCitation
    from src.workflow.raid_state_manager import RAIDStateManager
    from src.document_processing.base import DocumentMetadata
    from src.logging_config import get_logger
except ImportError:
    # Fallback for direct execution or testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, Status, SourceCitation
    from workflow.raid_state_manager import RAIDStateManager
    from document_processing.base import DocumentMetadata
    from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentAnalysisResult:
    """Result from analyzing an individual document."""
    document_metadata: DocumentMetadata
    potential_findings: List[Dict[str, Any]]  # Raw findings data before object creation
    analysis_summary: str
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class CrossDocumentSynthesis:
    """Result from cross-document synthesis analysis."""
    synthesized_findings: List[RAIDFinding]
    relationships_identified: List[Dict[str, str]]  # Pairs of finding IDs and relationship types
    contradictions_found: List[Dict[str, Any]]
    synthesis_summary: str


class WorkflowState(TypedDict):
    """
    LangGraph workflow state for RAID analysis pipeline.
    
    Tracks the analysis progress and findings through all workflow phases:
    - Document processing and individual analysis
    - Cross-document synthesis 
    - Evidence classification
    - Review and approval management
    """
    # Input documents and metadata
    documents: List[DocumentMetadata]
    
    # Individual document analysis results
    individual_analyses: List[DocumentAnalysisResult]
    
    # Cross-document synthesis results
    cross_document_synthesis: Optional[CrossDocumentSynthesis]
    
    # RAID state management
    raid_state_manager: RAIDStateManager
    
    # Workflow control and status
    current_phase: str  # "individual_analysis", "synthesis", "classification", "review"
    workflow_status: str  # "in_progress", "awaiting_review", "completed", "error"
    error_messages: List[str]
    
    # LLM integration state
    llm_provider_config: Dict[str, Any]
    llm_conversation_history: List[BaseMessage]
    
    # Session and checkpoint data
    session_id: str
    workflow_started_at: datetime
    last_updated_at: datetime
    
    # Human-in-the-loop review state
    pending_review_findings: List[str]  # Finding IDs awaiting approval
    review_decisions: Dict[str, str]  # Finding ID -> "approved" | "rejected"
    approval_complete: bool


class RAIDWorkflow:
    """
    LangGraph-based workflow orchestration for RAID analysis.
    
    Coordinates the multi-phase analysis process:
    1. Individual Document Analysis - Extract potential RAID items from each document
    2. Cross-Document Synthesis - Analyze evidence across documents, identify relationships
    3. Evidence Classification - Classify findings by evidence type (fact/inference/uncertainty)
    4. Review and Approval - Human-in-the-loop review with interrupt/resume capability
    """
    
    def __init__(self, llm_provider_config: Dict[str, Any]):
        """
        Initialize the RAID workflow with LLM provider configuration.
        
        Args:
            llm_provider_config: Configuration for LLM provider integration
        """
        self.llm_provider_config = llm_provider_config
        self.checkpointer = MemorySaver()  # InMemorySaver for session persistence
        self.graph = self._build_workflow_graph()
        
        logger.info("RAIDWorkflow initialized", 
                   provider_config=list(llm_provider_config.keys()))
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with nodes and edges.
        
        Returns:
            Configured StateGraph for RAID analysis
        """
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("individual_analysis", self.analyze_individual_documents)
        workflow.add_node("cross_document_synthesis", self.synthesize_cross_document)
        workflow.add_node("evidence_classification", self.classify_evidence)
        workflow.add_node("review_management", self.manage_review_approval)
        
        # Define workflow edges
        workflow.set_entry_point("individual_analysis")
        workflow.add_edge("individual_analysis", "cross_document_synthesis")
        workflow.add_edge("cross_document_synthesis", "evidence_classification")
        workflow.add_edge("evidence_classification", "review_management")
        workflow.add_edge("review_management", END)
        
        # Compile graph with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    def start_analysis(self, 
                      documents: List[DocumentMetadata], 
                      session_id: str) -> Dict[str, Any]:
        """
        Start a new RAID analysis workflow.
        
        Args:
            documents: List of processed document metadata
            session_id: Unique session identifier
            
        Returns:
            Initial workflow state and thread configuration
        """
        # Initialize workflow state
        initial_state = WorkflowState(
            documents=documents,
            individual_analyses=[],
            cross_document_synthesis=None,
            raid_state_manager=RAIDStateManager(),
            current_phase="individual_analysis",
            workflow_status="in_progress",
            error_messages=[],
            llm_provider_config=self.llm_provider_config,
            llm_conversation_history=[],
            session_id=session_id,
            workflow_started_at=datetime.now(),
            last_updated_at=datetime.now(),
            pending_review_findings=[],
            review_decisions={},
            approval_complete=False
        )
        
        # Create thread configuration for checkpointing
        thread_config = {"configurable": {"thread_id": session_id}}
        
        logger.info("RAID analysis workflow started",
                   session_id=session_id,
                   documents_count=len(documents))
        
        return {
            "initial_state": initial_state,
            "thread_config": thread_config,
            "graph": self.graph
        }
    
    def resume_workflow(self, session_id: str, review_decisions: Dict[str, str]) -> Dict[str, Any]:
        """
        Resume workflow after human review with approval decisions.
        
        Args:
            session_id: Session identifier
            review_decisions: Dictionary mapping finding IDs to approval decisions
            
        Returns:
            Updated workflow state and continuation info
        """
        thread_config = {"configurable": {"thread_id": session_id}}
        
        # Update state with review decisions
        update_state = {
            "review_decisions": review_decisions,
            "approval_complete": True,
            "last_updated_at": datetime.now()
        }
        
        logger.info("Workflow resumed with review decisions",
                   session_id=session_id,
                   decisions_count=len(review_decisions))
        
        return {
            "thread_config": thread_config,
            "update_state": update_state
        }
    
    # Workflow Node Functions
    
    def analyze_individual_documents(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Node 1: Individual Document Analysis
        
        Extract potential RAID items from each uploaded document independently.
        Tags findings with source document and location metadata.
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates with individual analysis results
        """
        logger.info("Starting individual document analysis",
                   documents_count=len(state["documents"]))
        
        analyses = []
        
        for document in state["documents"]:
            try:
                # Analyze individual document
                # This would integrate with LLM provider for actual analysis
                analysis_result = self._analyze_single_document(document, state)
                analyses.append(analysis_result)
                
                logger.debug("Document analyzed",
                           filename=document.filename,
                           findings_count=len(analysis_result.potential_findings))
                
            except Exception as e:
                error_msg = f"Error analyzing document {document.filename}: {str(e)}"
                logger.error("Document analysis failed",
                           filename=document.filename,
                           error=str(e))
                
                # Create error analysis result
                analyses.append(DocumentAnalysisResult(
                    document_metadata=document,
                    potential_findings=[],
                    analysis_summary=f"Analysis failed: {str(e)}",
                    processing_errors=[error_msg]
                ))
        
        # Update workflow state
        updates = {
            "individual_analyses": analyses,
            "current_phase": "synthesis",
            "last_updated_at": datetime.now()
        }
        
        # Collect any errors
        all_errors = []
        for analysis in analyses:
            all_errors.extend(analysis.processing_errors)
        
        if all_errors:
            updates["error_messages"] = state.get("error_messages", []) + all_errors
        
        logger.info("Individual document analysis completed",
                   total_analyses=len(analyses),
                   total_findings=sum(len(a.potential_findings) for a in analyses))
        
        return updates
    
    def synthesize_cross_document(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Node 2: Cross-Document Synthesis
        
        Analyze evidence across all documents before producing final findings.
        Detect relationships and contradictions between documents.
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates with synthesis results
        """
        logger.info("Starting cross-document synthesis")
        
        try:
            # Perform cross-document analysis
            # This would integrate with LLM provider for comprehensive synthesis
            synthesis_result = self._perform_cross_document_synthesis(state)
            
            # Add synthesized findings to RAID state manager
            for finding in synthesis_result.synthesized_findings:
                state["raid_state_manager"].add_finding(finding)
            
            # Add relationships between findings
            for relationship in synthesis_result.relationships_identified:
                finding_id1 = relationship.get("finding_id1")
                finding_id2 = relationship.get("finding_id2")
                if finding_id1 and finding_id2:
                    state["raid_state_manager"].add_relationship(finding_id1, finding_id2)
            
            updates = {
                "cross_document_synthesis": synthesis_result,
                "current_phase": "classification",
                "last_updated_at": datetime.now()
            }
            
            logger.info("Cross-document synthesis completed",
                       findings_count=len(synthesis_result.synthesized_findings),
                       relationships_count=len(synthesis_result.relationships_identified))
            
            return updates
            
        except Exception as e:
            error_msg = f"Cross-document synthesis failed: {str(e)}"
            logger.error("Synthesis failed", error=str(e))
            
            return {
                "error_messages": state.get("error_messages", []) + [error_msg],
                "workflow_status": "error",
                "last_updated_at": datetime.now()
            }
    
    def classify_evidence(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Node 3: Evidence Classification
        
        Classify findings as fact, inference, or uncertainty.
        Link each finding to supporting document evidence.
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates with evidence classification
        """
        logger.info("Starting evidence classification")
        
        try:
            # Classify evidence for all findings
            # This would integrate with LLM provider for evidence analysis
            classification_results = self._classify_finding_evidence(state)
            
            # Update findings with evidence classifications
            for finding_id, evidence_type in classification_results.items():
                finding = state["raid_state_manager"].get_finding(finding_id)
                if finding:
                    # Update evidence type using proper enum
                    if isinstance(evidence_type, str):
                        evidence_type = EvidenceType(evidence_type.lower())
                    finding.evidence_type = evidence_type
                    finding.updated_at = datetime.now()
            
            updates = {
                "current_phase": "review",
                "last_updated_at": datetime.now()
            }
            
            logger.info("Evidence classification completed",
                       classified_findings=len(classification_results))
            
            return updates
            
        except Exception as e:
            error_msg = f"Evidence classification failed: {str(e)}"
            logger.error("Evidence classification failed", error=str(e))
            
            return {
                "error_messages": state.get("error_messages", []) + [error_msg],
                "workflow_status": "error",
                "last_updated_at": datetime.now()
            }
    
    def manage_review_approval(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Node 4: Review and Approval Management
        
        Present findings for PM review with approve/reject options.
        Use interrupt() to pause for review and Command(resume=...) for approval decisions.
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates with review management
        """
        logger.info("Starting review and approval management")
        
        try:
            # Check if this is initial review setup or processing decisions
            if not state.get("approval_complete", False):
                # Initial setup - prepare findings for review
                pending_findings = state["raid_state_manager"].get_pending_findings()
                pending_ids = [finding.id for finding in pending_findings]
                
                updates = {
                    "pending_review_findings": pending_ids,
                    "workflow_status": "awaiting_review",
                    "last_updated_at": datetime.now()
                }
                
                logger.info("Findings prepared for review",
                           pending_count=len(pending_ids))
                
                # This would trigger interrupt() in actual LangGraph execution
                # The workflow pauses here until resume_workflow() is called
                return updates
            
            else:
                # Process review decisions
                review_decisions = state.get("review_decisions", {})
                
                # Apply approval decisions to findings
                approved_count = 0
                rejected_count = 0
                
                for finding_id, decision in review_decisions.items():
                    finding = state["raid_state_manager"].get_finding(finding_id)
                    if finding:
                        if decision.lower() == "approved":
                            finding.approve()
                            approved_count += 1
                        elif decision.lower() == "rejected":
                            finding.reject()
                            rejected_count += 1
                
                updates = {
                    "workflow_status": "completed",
                    "current_phase": "completed",
                    "last_updated_at": datetime.now()
                }
                
                logger.info("Review decisions processed",
                           approved_count=approved_count,
                           rejected_count=rejected_count)
                
                return updates
        
        except Exception as e:
            error_msg = f"Review management failed: {str(e)}"
            logger.error("Review management failed", error=str(e))
            
            return {
                "error_messages": state.get("error_messages", []) + [error_msg],
                "workflow_status": "error",
                "last_updated_at": datetime.now()
            }
    
    # Helper Methods for LLM Integration (Stubs for now)
    
    def _analyze_single_document(self, 
                                document: DocumentMetadata, 
                                state: WorkflowState) -> DocumentAnalysisResult:
        """
        Analyze individual document for RAID items.
        
        This is a stub implementation. In the actual system, this would:
        1. Use the configured LLM provider to analyze document content
        2. Extract potential RAID items with evidence
        3. Tag findings with source metadata
        
        Args:
            document: Document metadata to analyze
            state: Current workflow state
            
        Returns:
            Analysis result with potential findings
        """
        # Stub implementation - would integrate with actual LLM provider
        potential_findings = [
            {
                "type": "risk",
                "description": f"Potential risk identified in {document.filename}",
                "evidence_type": "inference",
                "source_citations": [{
                    "filename": document.filename,
                    "location": "Section 1",
                    "excerpt": "Sample text excerpt"
                }],
                "impact_assessment": "",
                "recommendations": ""
            }
        ]
        
        return DocumentAnalysisResult(
            document_metadata=document,
            potential_findings=potential_findings,
            analysis_summary=f"Analyzed {document.filename}: {len(potential_findings)} potential findings"
        )
    
    def _perform_cross_document_synthesis(self, state: WorkflowState) -> CrossDocumentSynthesis:
        """
        Perform cross-document synthesis of findings.
        
        This is a stub implementation. In the actual system, this would:
        1. Use the configured LLM provider for comprehensive analysis
        2. Compare findings across documents
        3. Identify relationships and contradictions
        4. Generate final consolidated findings
        
        Args:
            state: Current workflow state
            
        Returns:
            Cross-document synthesis results
        """
        # Stub implementation - would integrate with actual LLM provider
        # Convert potential findings to RAIDFinding objects
        synthesized_findings = []
        
        for analysis in state["individual_analyses"]:
            for finding_data in analysis.potential_findings:
                # Create SourceCitation objects
                citations = [
                    SourceCitation(**citation) 
                    for citation in finding_data["source_citations"]
                ]
                
                # Create RAIDFinding
                finding = RAIDFinding(
                    type=RAIDType(finding_data["type"]),
                    description=finding_data["description"],
                    evidence_type=EvidenceType(finding_data["evidence_type"]),
                    source_citations=citations,
                    impact_assessment=finding_data.get("impact_assessment", ""),
                    recommendations=finding_data.get("recommendations", "")
                )
                
                synthesized_findings.append(finding)
        
        return CrossDocumentSynthesis(
            synthesized_findings=synthesized_findings,
            relationships_identified=[],  # Would be populated by LLM analysis
            contradictions_found=[],      # Would be populated by LLM analysis
            synthesis_summary="Cross-document synthesis completed"
        )
    
    def _classify_finding_evidence(self, state: WorkflowState) -> Dict[str, str]:
        """
        Classify evidence for all findings.
        
        This is a stub implementation. In the actual system, this would:
        1. Use the configured LLM provider to analyze evidence
        2. Classify each finding as fact, inference, or uncertainty
        3. Provide reasoning for classifications
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary mapping finding IDs to evidence types
        """
        # Stub implementation - would integrate with actual LLM provider
        classifications = {}
        
        all_findings = state["raid_state_manager"].get_all_findings()
        for finding in all_findings:
            # Simple stub classification based on source citations
            if len(finding.source_citations) > 1:
                classifications[finding.id] = "fact"
            else:
                classifications[finding.id] = "inference"
        
        return classifications
    
    def get_workflow_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about the current workflow state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with workflow statistics
        """
        thread_config = {"configurable": {"thread_id": session_id}}
        
        try:
            # Get current state from checkpointer
            checkpoint = self.checkpointer.get(thread_config)
            if not checkpoint:
                return {"error": "No workflow state found for session"}
            
            # Extract statistics from state
            # This would be implemented based on actual checkpoint structure
            return {
                "session_id": session_id,
                "has_checkpoint": True,
                "checkpoint_exists": checkpoint is not None
            }
            
        except Exception as e:
            logger.error("Failed to get workflow statistics",
                        session_id=session_id,
                        error=str(e))
            return {"error": f"Failed to get statistics: {str(e)}"}
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up workflow session data.
        
        Args:
            session_id: Session identifier to clean up
            
        Returns:
            True if cleanup was successful
        """
        try:
            # Note: MemorySaver doesn't have explicit cleanup methods
            # Session data will be cleared when the object is destroyed
            logger.info("Workflow session marked for cleanup",
                       session_id=session_id)
            return True
            
        except Exception as e:
            logger.error("Failed to cleanup workflow session",
                        session_id=session_id,
                        error=str(e))
            return False