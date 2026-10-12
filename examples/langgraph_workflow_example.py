"""
Example demonstrating LangGraph workflow orchestration for RAID analysis.
Shows how to use the workflow components for document analysis and finding management.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow.langgraph_workflow import RAIDWorkflow, WorkflowState
from workflow.raid_state_manager import RAIDStateManager
from document_processing.base import DocumentMetadata, ContentSection
from raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, SourceCitation


def create_sample_documents():
    """Create sample document metadata for demonstration."""
    
    # Sample project charter document
    project_charter = DocumentMetadata(
        filename="project_charter.pdf",
        file_type="pdf",
        content_sections=[
            ContentSection(
                location="Executive Summary",
                text="The Customer Portal 2024 project will deliver a new web-based customer interface by Q2 2024. The project budget is $500K with a 6-month timeline.",
                metadata={"section": "executive_summary", "page": 1}
            ),
            ContentSection(
                location="Risk Assessment - Section 3",
                text="Major risks include: 1) Timeline dependency on third-party API integration, 2) Budget uncertainty due to cloud infrastructure costs, 3) Resource availability during peak season.",
                metadata={"section": "risk_assessment", "page": 3}
            ),
            ContentSection(
                location="Success Criteria - Section 4",
                text="Project success assumes stable legacy system APIs and completion of security audit by March 2024.",
                metadata={"section": "success_criteria", "page": 4}
            )
        ]
    )
    
    # Sample requirements specification
    requirements_spec = DocumentMetadata(
        filename="requirements_specification.docx",
        file_type="docx",
        content_sections=[
            ContentSection(
                location="Section 2.1 - Functional Requirements",
                text="The system must integrate with existing customer database and provide real-time account information. Single sign-on is required for all users.",
                metadata={"section": "2.1", "heading": "Functional Requirements"}
            ),
            ContentSection(
                location="Section 5.2 - Technical Dependencies",
                text="Project depends on completion of infrastructure upgrade by IT Operations team. Customer authentication service must be available with 99.9% uptime SLA.",
                metadata={"section": "5.2", "heading": "Technical Dependencies"}
            ),
            ContentSection(
                location="Section 6.1 - Current Issues",
                text="Legacy system experiencing performance degradation. Customer data export functionality is currently broken and needs immediate attention.",
                metadata={"section": "6.1", "heading": "Current Issues"}
            )
        ]
    )
    
    return [project_charter, requirements_spec]


def demonstrate_workflow_initialization():
    """Demonstrate workflow initialization and basic setup."""
    print("=" * 60)
    print("LangGraph RAID Workflow Demonstration")
    print("=" * 60)
    
    # Create LLM provider configuration
    llm_config = {
        "provider_type": "openai",
        "api_key": "demo_api_key",
        "model": "gpt-3.5-turbo",
        "temperature": 0.3
    }
    
    # Initialize workflow
    print("\n1. Initializing LangGraph RAID Workflow...")
    workflow = RAIDWorkflow(llm_config)
    print(f"   ✓ Workflow initialized with {llm_config['provider_type']} provider")
    print(f"   ✓ Checkpointer configured: {type(workflow.checkpointer).__name__}")
    print(f"   ✓ Graph nodes: individual_analysis, cross_document_synthesis, evidence_classification, review_management")
    
    return workflow


def demonstrate_analysis_startup(workflow):
    """Demonstrate starting a new RAID analysis."""
    print("\n2. Starting RAID Analysis Session...")
    
    # Create sample documents
    documents = create_sample_documents()
    session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"   Documents to analyze: {len(documents)}")
    for doc in documents:
        print(f"     - {doc.filename} ({doc.file_type.upper()}): {len(doc.content_sections)} sections")
    
    # Start analysis
    result = workflow.start_analysis(documents, session_id)
    
    initial_state = result["initial_state"]
    thread_config = result["thread_config"]
    
    print(f"   ✓ Analysis started with session ID: {session_id}")
    print(f"   ✓ Initial phase: {initial_state['current_phase']}")
    print(f"   ✓ Workflow status: {initial_state['workflow_status']}")
    print(f"   ✓ Thread configuration ready for checkpointing")
    
    return result, documents


def demonstrate_phase_execution(workflow, initial_state):
    """Demonstrate executing workflow phases."""
    print("\n3. Executing Workflow Phases...")
    
    state = initial_state.copy()
    
    # Phase 1: Individual Document Analysis
    print("   Phase 1: Individual Document Analysis")
    individual_updates = workflow.analyze_individual_documents(state)
    state.update(individual_updates)
    
    analyses = state["individual_analyses"]
    total_findings = sum(len(analysis.potential_findings) for analysis in analyses)
    print(f"     ✓ Analyzed {len(analyses)} documents")
    print(f"     ✓ Identified {total_findings} potential findings")
    print(f"     ✓ Next phase: {state['current_phase']}")
    
    # Phase 2: Cross-Document Synthesis
    print("   Phase 2: Cross-Document Synthesis")
    synthesis_updates = workflow.synthesize_cross_document(state)
    state.update(synthesis_updates)
    
    synthesis = state["cross_document_synthesis"]
    raid_manager = state["raid_state_manager"]
    
    print(f"     ✓ Synthesized {len(synthesis.synthesized_findings)} final findings")
    print(f"     ✓ Identified {len(synthesis.relationships_identified)} relationships")
    print(f"     ✓ RAID state manager contains {len(raid_manager)} findings")
    print(f"     ✓ Next phase: {state['current_phase']}")
    
    # Phase 3: Evidence Classification
    print("   Phase 3: Evidence Classification")
    classification_updates = workflow.classify_evidence(state)
    state.update(classification_updates)
    
    print(f"     ✓ Classified evidence for all findings")
    print(f"     ✓ Next phase: {state['current_phase']}")
    
    # Phase 4: Review Management (Initial Setup)
    print("   Phase 4: Review Management - Setup")
    review_updates = workflow.manage_review_approval(state)
    state.update(review_updates)
    
    pending_count = len(state["pending_review_findings"])
    print(f"     ✓ Prepared {pending_count} findings for review")
    print(f"     ✓ Workflow status: {state['workflow_status']}")
    print(f"     ✓ Workflow paused for human review")
    
    return state


def demonstrate_findings_analysis(state):
    """Demonstrate analyzing the generated RAID findings."""
    print("\n4. Analyzing Generated RAID Findings...")
    
    raid_manager = state["raid_state_manager"]
    all_findings = raid_manager.get_all_findings()
    
    print(f"   Total findings: {len(all_findings)}")
    
    # Analyze by type
    stats = raid_manager.get_statistics()
    print("   Findings by type:")
    for raid_type, count in stats["by_type"].items():
        print(f"     - {raid_type.upper()}: {count}")
    
    print("   Findings by evidence type:")
    for evidence_type, count in stats["by_evidence_type"].items():
        print(f"     - {evidence_type.upper()}: {count}")
    
    # Show sample findings
    print("\n   Sample findings:")
    for i, finding in enumerate(all_findings[:3]):  # Show first 3
        print(f"     {i+1}. {finding.type.value.upper()}: {finding.description[:60]}...")
        print(f"        Evidence: {finding.evidence_type.value}")
        print(f"        Sources: {len(finding.source_citations)} citation(s)")
        if finding.source_citations:
            citation = finding.source_citations[0]
            print(f"        Primary source: {citation.filename} ({citation.location})")
    
    return all_findings


def demonstrate_review_simulation(workflow, state, session_id):
    """Demonstrate simulating human review and approval decisions."""
    print("\n5. Simulating Human Review Process...")
    
    pending_findings = state["pending_review_findings"]
    print(f"   Findings pending review: {len(pending_findings)}")
    
    # Simulate approval decisions
    review_decisions = {}
    raid_manager = state["raid_state_manager"]
    
    for i, finding_id in enumerate(pending_findings):
        finding = raid_manager.get_finding(finding_id)
        if finding:
            # Simulate decision based on finding type (demo logic)
            if finding.type in [RAIDType.RISK, RAIDType.ISSUE]:
                decision = "approved"  # Approve risks and issues
            else:
                decision = "approved" if i % 2 == 0 else "rejected"  # Alternate for assumptions/dependencies
            
            review_decisions[finding_id] = decision
            print(f"     - {finding.type.value.upper()}: {decision}")
    
    # Resume workflow with decisions
    print(f"\n   Resuming workflow with {len(review_decisions)} decisions...")
    resume_result = workflow.resume_workflow(session_id, review_decisions)
    
    # Update state with final processing
    state["review_decisions"] = review_decisions
    state["approval_complete"] = True
    
    final_updates = workflow.manage_review_approval(state)
    state.update(final_updates)
    
    print(f"   ✓ Workflow status: {state['workflow_status']}")
    print(f"   ✓ Current phase: {state['current_phase']}")
    
    # Show final statistics
    approved_findings = raid_manager.get_approved_findings()
    rejected_findings = raid_manager.filter_findings(status="rejected")
    
    print(f"   ✓ Final results: {len(approved_findings)} approved, {len(rejected_findings)} rejected")
    
    return approved_findings


def demonstrate_session_management(workflow, session_id, state):
    """Demonstrate session management capabilities."""
    print("\n6. Session Management...")
    
    # Get workflow statistics
    # This demo invokes nodes directly, so no LangGraph checkpoint is created.
    stats = {
        "session_id": session_id,
        "workflow_status": state["workflow_status"],
        "current_phase": state["current_phase"],
        "raid_statistics": state["raid_state_manager"].get_statistics(),
    }
    print(f"   Session statistics:")
    for key, value in stats.items():
        print(f"     - {key}: {value}")
    
    # Cleanup session
    cleanup_success = workflow.cleanup_session(session_id)
    print(f"   ✓ Session cleanup: {'successful' if cleanup_success else 'failed'}")


def main():
    """Run the complete LangGraph workflow demonstration."""
    try:
        # Initialize workflow
        workflow = demonstrate_workflow_initialization()
        
        # Start analysis
        result, documents = demonstrate_analysis_startup(workflow)
        initial_state = result["initial_state"]
        session_id = initial_state["session_id"]
        
        # Execute phases
        final_state = demonstrate_phase_execution(workflow, initial_state)
        
        # Analyze findings
        findings = demonstrate_findings_analysis(final_state)
        
        # Simulate review
        approved_findings = demonstrate_review_simulation(workflow, final_state, session_id)
        
        # Session management
        demonstrate_session_management(workflow, session_id, final_state)
        
        # Summary
        print("\n" + "=" * 60)
        print("Workflow Demonstration Complete")
        print("=" * 60)
        print(f"✓ Processed {len(documents)} documents")
        print(f"✓ Generated {len(findings)} total findings")
        print(f"✓ Approved {len(approved_findings)} findings for export")
        print("✓ Demonstrated all workflow phases")
        print("✓ Session management completed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)