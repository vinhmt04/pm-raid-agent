"""
LangGraph workflow orchestration for PM RAID Agent MVP.
Manages the multi-phase RAID analysis pipeline through discrete workflow nodes.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import json
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
# LLM provider imports
try:
    from src.llm_providers.factory import llm_factory
    from src.llm_providers.base import LLMRequest, LLMProviderError
except ImportError:
    from llm_providers.factory import llm_factory
    from llm_providers.base import LLMRequest, LLMProviderError




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
    
    # Helper Methods for LLM Integration
    
    def _analyze_single_document(self,
                                 document: DocumentMetadata,
                                 state: WorkflowState) -> DocumentAnalysisResult:
        """Extract source-grounded RAID candidates from each document section."""
        findings: List[Dict[str, Any]] = []
        errors: List[str] = []
        sections = [section for section in document.content_sections
                    if isinstance(section.text, str) and section.text.strip()]
        if not sections:
            return DocumentAnalysisResult(document, [], "No extractable text to analyze")

        try:
            provider = llm_factory.get_provider()
            if not provider.is_configured():
                raise LLMProviderError("LLM provider is not configured")
        except Exception as exc:
            message = f"LLM provider initialization failed: {exc}"
            return DocumentAnalysisResult(document, [], message, [message])

        for section in sections:
            location = str(section.location)
            content = section.text.strip()
            request = LLMRequest(
                system_prompt=(
                    "You are a project management RAID analyst. Analyze ONLY the "
                    "provided source text. Return a JSON object with a 'findings' "
                    "array. Each finding must have: type (risk, assumption, issue, "
                    "dependency), description, evidence_type (fact, inference, "
                    "uncertainty), excerpt (an EXACT contiguous substring of the "
                    "source), impact_assessment, recommendations. Do not invent "
                    "facts or citations. Return an empty array if no RAID items "
                    "are supported. Output JSON only, without markdown fences."
                ),
                messages=[{"role": "user", "content": (
                    f"Document: {document.filename}\n"
                    f"Location: {location}\nSource text:\n{content}"
                )}],
                temperature=0,
            )
            try:
                response = provider.generate_response(request)
                raw = response.content.strip()
                # Some providers wrap JSON in a markdown code fence.
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    if len(lines) >= 3 and lines[-1].strip() == "```":
                        raw = "\n".join(lines[1:-1])
                payload = json.loads(raw)
                items = payload.get("findings") if isinstance(payload, dict) else None
                if not isinstance(items, list):
                    raise ValueError("Expected a JSON object containing a findings array")

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    raid_type = str(item.get("type", "")).lower().strip()
                    evidence_type = str(item.get("evidence_type", "")).lower().strip()
                    description = item.get("description")
                    excerpt = item.get("excerpt")
                    if (raid_type not in {"risk", "assumption", "issue", "dependency"}
                            or evidence_type not in {"fact", "inference", "uncertainty"}
                            or not isinstance(description, str) or not description.strip()
                            or not isinstance(excerpt, str) or not excerpt.strip()
                            or excerpt.strip() not in content):
                        errors.append(f"Discarded invalid or ungrounded finding at {location}")
                        continue
                    findings.append({
                        "type": raid_type,
                        "description": description.strip(),
                        "evidence_type": evidence_type,
                        "source_citations": [{
                            "filename": document.filename,
                            "location": location,
                            "excerpt": excerpt.strip(),
                        }],
                        "impact_assessment": str(item.get("impact_assessment") or ""),
                        "recommendations": str(item.get("recommendations") or ""),
                    })
            except Exception as exc:
                message = f"Failed analyzing {document.filename} at {location}: {exc}"
                logger.error("Document section analysis failed", filename=document.filename,
                             location=location, error=str(exc))
                errors.append(message)

        return DocumentAnalysisResult(
            document_metadata=document,
            potential_findings=findings,
            analysis_summary=(f"Analyzed {document.filename}: {len(findings)} RAID "
                              f"candidates from {len(sections)} sections"),
            processing_errors=errors,
        )

    def _perform_cross_document_synthesis(self, state: WorkflowState) -> CrossDocumentSynthesis:
        """Consolidate document candidates and inspect cross-source links using an LLM.

        The model can only refer to numbered candidates. Source citations are
        always copied from verified individual analyses, never model-generated.
        """
        candidates = []
        for analysis in state["individual_analyses"]:
            for raw in analysis.potential_findings:
                try:
                    citations = [SourceCitation(**c) if isinstance(c, dict) else c
                                 for c in raw["source_citations"]]
                    candidate = RAIDFinding(
                        type=RAIDType(raw["type"]),
                        description=raw["description"],
                        evidence_type=EvidenceType(raw["evidence_type"]),
                        source_citations=citations,
                        impact_assessment=raw.get("impact_assessment", ""),
                        recommendations=raw.get("recommendations", ""),
                    )
                    candidates.append(candidate)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Skipping invalid document finding", error=str(exc))

        if not candidates:
            return CrossDocumentSynthesis([], [], [], "No valid RAID candidates")

        # Preserve every grounded candidate when there is no model or when it
        # fails; never silently invent relationships or contradictions.
        def fallback(reason: str) -> CrossDocumentSynthesis:
            return CrossDocumentSynthesis(candidates, [], [], reason)

        # A single document does not provide cross-document evidence.
        source_files = {c.filename for finding in candidates
                        for c in finding.source_citations}
        if len(source_files) < 2:
            return fallback("Single-document findings; no cross-document synthesis")

        try:
            provider = llm_factory.get_provider()
            if not provider.is_configured():
                return fallback("LLM unavailable; retaining individual findings")

            indexed = [
                {"index": i, "type": f.type.value, "description": f.description,
                 "evidence_type": f.evidence_type.value,
                 "citations": [{"filename": c.filename, "location": c.location,
                                "excerpt": c.excerpt} for c in f.source_citations]}
                for i, f in enumerate(candidates)
            ]
            request = LLMRequest(
                system_prompt=(
                    "You are a project RAID evidence synthesis analyst. Compare "
                    "the indexed findings from different documents. Output ONLY "
                    "JSON with arrays 'groups', 'relationships', 'contradictions'. "
                    "groups are arrays of integer candidate indices that refer to "
                    "the SAME underlying RAID item; omit ungrouped indices. "
                    "relationships contain {a, b, type} for two candidate indices "
                    "with a supported relationship. contradictions contain "
                    "{a, b, explanation} for directly conflicting evidence. "
                    "Only use given indices. Never fabricate evidence. "
                    "Use empty arrays if none are supported."
                ),
                messages=[{"role": "user", "content": json.dumps(indexed, ensure_ascii=False)}],
                temperature=0,
            )
            raw = provider.generate_response(request).content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    raw = "\n".join(lines[1:-1])
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not all(
                isinstance(payload.get(key), list)
                for key in ("groups", "relationships", "contradictions")
            ):
                raise ValueError("Invalid cross-document synthesis JSON schema")

            def valid_index(value):
                return type(value) is int and 0 <= value < len(candidates)

            def cross_source(i, j):
                left = {c.filename for c in candidates[i].source_citations}
                right = {c.filename for c in candidates[j].source_citations}
                return bool(left - right or right - left)

            # Merge only disjoint, independently grounded duplicate groups.
            used = set()
            groups = []
            for group in payload["groups"]:
                if (not isinstance(group, list) or len(group) < 2
                        or not all(valid_index(i) for i in group)
                        or len(set(group)) != len(group)
                        or any(i in used for i in group)
                        or len({candidates[i].type for i in group}) != 1
                        or len({c.filename for i in group
                                for c in candidates[i].source_citations}) < 2):
                    continue
                groups.append(group)
                used.update(group)

            consolidated = []
            index_to_finding = {}
            for group in groups:
                primary = candidates[group[0]]
                citations = []
                seen = set()
                for i in group:
                    for citation in candidates[i].source_citations:
                        key = (citation.filename, citation.location, citation.excerpt)
                        if key not in seen:
                            seen.add(key)
                            citations.append(citation)
                primary.source_citations = citations
                primary.metadata["merged_candidate_indices"] = list(group)
                consolidated.append(primary)
                for i in group:
                    index_to_finding[i] = primary
            for i, candidate in enumerate(candidates):
                if i not in used:
                    consolidated.append(candidate)
                    index_to_finding[i] = candidate

            relationships = []
            contradictions = []
            seen_pairs = set()
            for item in payload["relationships"]:
                if not isinstance(item, dict):
                    continue
                i, j = item.get("a"), item.get("b")
                relation_type = item.get("type")
                if (not valid_index(i) or not valid_index(j) or i == j
                        or not cross_source(i, j)
                        or not isinstance(relation_type, str)
                        or not relation_type.strip()):
                    continue
                first, second = index_to_finding[i], index_to_finding[j]
                if first.id == second.id:
                    continue
                key = tuple(sorted((first.id, second.id)))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    relationships.append({"finding_id1": first.id,
                                          "finding_id2": second.id,
                                          "relationship_type": relation_type.strip()})

            seen_contradictions = set()
            for item in payload["contradictions"]:
                if not isinstance(item, dict):
                    continue
                i, j = item.get("a"), item.get("b")
                explanation = item.get("explanation")
                if (not valid_index(i) or not valid_index(j) or i == j
                        or not cross_source(i, j)
                        or not isinstance(explanation, str)
                        or not explanation.strip()):
                    continue
                first, second = index_to_finding[i], index_to_finding[j]
                key = tuple(sorted((i, j)))
                if key not in seen_contradictions:
                    seen_contradictions.add(key)
                    contradictions.append({"finding_id1": first.id,
                                           "finding_id2": second.id,
                                           "explanation": explanation.strip(),
                                           "source_citations": [
                                               {"filename": c.filename,
                                                "location": c.location,
                                                "excerpt": c.excerpt}
                                               for c in (candidates[i].source_citations
                                                         + candidates[j].source_citations)]})
            return CrossDocumentSynthesis(
                synthesized_findings=consolidated,
                relationships_identified=relationships,
                contradictions_found=contradictions,
                synthesis_summary=(f"Synthesized {len(candidates)} candidates into "
                                   f"{len(consolidated)} findings; "
                                   f"{len(relationships)} relationships and "
                                   f"{len(contradictions)} contradictions"),
            )
        except Exception as exc:
            logger.warning("Cross-document LLM synthesis failed; preserving findings",
                           error=str(exc))
            return fallback(f"Cross-document synthesis unavailable: {exc}")

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