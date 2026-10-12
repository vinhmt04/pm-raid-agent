#!/usr/bin/env python3
"""
Example demonstrating RAID State Management functionality.
Shows how to use the RAIDStateManager for managing RAID findings collections.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow.raid_state_manager import RAIDStateManager
from raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, Status, SourceCitation


def main():
    """Demonstrate RAID state management functionality."""
    print("=== RAID State Manager Example ===\n")
    
    # Create a state manager
    manager = RAIDStateManager()
    print(f"1. Initialized state manager (empty): {len(manager)} findings")
    
    # Create some sample findings
    risk_citation = SourceCitation(
        filename="project_plan.docx",
        location="Section 3.2",
        excerpt="Timeline appears aggressive given resource constraints"
    )
    
    risk_finding = RAIDFinding(
        type=RAIDType.RISK,
        description="Project timeline may be too aggressive",
        evidence_type=EvidenceType.INFERENCE,
        source_citations=[risk_citation],
        impact_assessment="High - may cause project delays",
        recommendations="Add 2-week buffer to critical path activities"
    )
    
    issue_citation = SourceCitation(
        filename="budget_review.xlsx",
        location="Sheet 'Current', A15",
        excerpt="Database licensing costs exceed budget by $50K"
    )
    
    issue_finding = RAIDFinding(
        type=RAIDType.ISSUE,
        description="Database licensing costs over budget",
        evidence_type=EvidenceType.FACT,
        source_citations=[issue_citation],
        impact_assessment="Medium - requires budget reallocation",
        recommendations="Negotiate enterprise discount or consider alternatives"
    )
    
    dependency_citation = SourceCitation(
        filename="integration_spec.pdf",
        location="Page 12",
        excerpt="System integration depends on vendor API v2.0 release"
    )
    
    dependency_finding = RAIDFinding(
        type=RAIDType.DEPENDENCY,
        description="Integration blocked pending vendor API release",
        evidence_type=EvidenceType.FACT,
        source_citations=[dependency_citation],
        impact_assessment="High - blocks testing phase",
        recommendations="Establish fallback plan with vendor API v1.5"
    )
    
    # Add findings to the manager
    print("2. Adding findings to state manager...")
    manager.add_finding(risk_finding)
    manager.add_finding(issue_finding)
    manager.add_finding(dependency_finding)
    print(f"   Total findings: {len(manager)}")
    
    # Demonstrate filtering
    print("\n3. Filtering demonstrations:")
    risks = manager.filter_findings(type=RAIDType.RISK)
    print(f"   Risk findings: {len(risks)}")
    
    facts = manager.filter_findings(evidence_type=EvidenceType.FACT)
    print(f"   Fact-based findings: {len(facts)}")
    
    pending = manager.get_pending_findings()
    print(f"   Pending approval: {len(pending)}")
    
    # Demonstrate relationships
    print("\n4. Adding relationships between findings...")
    manager.add_relationship(risk_finding.id, dependency_finding.id)
    print(f"   Added relationship between risk and dependency")
    
    related = manager.get_related_findings(risk_finding.id)
    print(f"   Risk finding has {len(related)} related items")
    
    # Demonstrate approval workflow
    print("\n5. Approval workflow demonstration:")
    manager.update_finding(risk_finding.id, status=Status.APPROVED)
    manager.update_finding(issue_finding.id, status=Status.APPROVED)
    # Leave dependency as pending
    
    approved = manager.get_approved_findings()
    pending = manager.get_pending_findings()
    print(f"   Approved findings: {len(approved)}")
    print(f"   Pending findings: {len(pending)}")
    
    # Demonstrate bulk operations
    print("\n6. Bulk operations example:")
    all_ids = [f.id for f in manager.get_all_findings()]
    results = manager.bulk_approve_findings(all_ids)
    successful_approvals = sum(results.values())
    print(f"   Bulk approved {successful_approvals} out of {len(all_ids)} findings")
    
    # Show statistics
    print("\n7. Collection statistics:")
    stats = manager.get_statistics()
    print(f"   Total findings: {stats['total_findings']}")
    print(f"   By type: {dict(stats['by_type'])}")
    print(f"   By status: {dict(stats['by_status'])}")
    print(f"   With relationships: {stats['with_relationships']}")
    
    # Demonstrate state export/import
    print("\n8. State persistence example:")
    exported_state = manager.export_state()
    print(f"   Exported state with {len(exported_state['findings'])} findings")
    
    # Create new manager and import state
    new_manager = RAIDStateManager()
    new_manager.import_state(exported_state)
    print(f"   Imported to new manager: {len(new_manager)} findings")
    
    # Verify data integrity
    imported_stats = new_manager.get_statistics()
    print(f"   Imported statistics match: {imported_stats['total_findings'] == stats['total_findings']}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()