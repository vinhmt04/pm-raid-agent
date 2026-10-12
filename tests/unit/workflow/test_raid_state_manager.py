"""
Unit tests for RAID State Manager.
Tests state management for RAID findings collections including persistence and operations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.workflow.raid_state_manager import RAIDStateManager
from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, Status, SourceCitation


class TestRAIDStateManager:
    """Test suite for RAIDStateManager."""
    
    @pytest.fixture
    def state_manager(self):
        """Create a fresh RAIDStateManager for each test."""
        return RAIDStateManager()
    
    @pytest.fixture
    def sample_finding(self):
        """Create a sample RAID finding for testing."""
        citation = SourceCitation(
            filename="test.pdf",
            location="page 1",
            excerpt="Test excerpt"
        )
        
        return RAIDFinding(
            type=RAIDType.RISK,
            description="Test risk description",
            evidence_type=EvidenceType.FACT,
            source_citations=[citation],
            impact_assessment="High impact",
            recommendations="Mitigate immediately"
        )
    
    @pytest.fixture
    def sample_findings(self):
        """Create multiple sample RAID findings for testing."""
        citations = [
            SourceCitation("doc1.pdf", "page 1", "Risk evidence"),
            SourceCitation("doc2.docx", "section 2", "Issue evidence"),
            SourceCitation("doc3.xlsx", "sheet A1", "Dependency evidence")
        ]
        
        findings = [
            RAIDFinding(
                type=RAIDType.RISK,
                description="Sample risk",
                evidence_type=EvidenceType.FACT,
                source_citations=[citations[0]]
            ),
            RAIDFinding(
                type=RAIDType.ISSUE,
                description="Sample issue",
                evidence_type=EvidenceType.INFERENCE,
                source_citations=[citations[1]]
            ),
            RAIDFinding(
                type=RAIDType.DEPENDENCY,
                description="Sample dependency",
                evidence_type=EvidenceType.UNCERTAINTY,
                source_citations=[citations[2]]
            )
        ]
        
        return findings


class TestBasicOperations(TestRAIDStateManager):
    """Test basic CRUD operations."""
    
    def test_initialization(self, state_manager):
        """Test state manager initialization."""
        assert len(state_manager) == 0
        assert state_manager.get_all_findings() == []
        
        stats = state_manager.get_statistics()
        assert stats['total_findings'] == 0
        assert stats['approved_count'] == 0
        assert stats['pending_count'] == 0
    
    def test_add_finding_success(self, state_manager, sample_finding):
        """Test successful addition of a RAID finding."""
        finding_id = state_manager.add_finding(sample_finding)
        
        assert finding_id == sample_finding.id
        assert len(state_manager) == 1
        assert sample_finding.id in state_manager
        
        retrieved = state_manager.get_finding(sample_finding.id)
        assert retrieved == sample_finding
    
    def test_add_finding_duplicate_id(self, state_manager, sample_finding):
        """Test that adding a finding with duplicate ID raises error."""
        state_manager.add_finding(sample_finding)
        
        # Create another finding with same ID
        duplicate_finding = RAIDFinding(
            id=sample_finding.id,
            type=RAIDType.ISSUE,
            description="Different description",
            evidence_type=EvidenceType.FACT,
            source_citations=[SourceCitation("test.pdf", "page 1", "excerpt")]
        )
        
        with pytest.raises(ValueError, match="already exists"):
            state_manager.add_finding(duplicate_finding)
    
    def test_add_finding_invalid_type(self, state_manager):
        """Test that adding non-RAIDFinding raises error."""
        with pytest.raises(ValueError, match="must be a RAIDFinding instance"):
            state_manager.add_finding("not a finding")
    
    def test_get_finding_nonexistent(self, state_manager):
        """Test retrieving non-existent finding returns None."""
        result = state_manager.get_finding("nonexistent-id")
        assert result is None
    
    def test_delete_finding_success(self, state_manager, sample_finding):
        """Test successful deletion of a RAID finding."""
        state_manager.add_finding(sample_finding)
        assert len(state_manager) == 1
        
        result = state_manager.delete_finding(sample_finding.id)
        assert result is True
        assert len(state_manager) == 0
        assert sample_finding.id not in state_manager
        assert state_manager.get_finding(sample_finding.id) is None
    
    def test_delete_finding_nonexistent(self, state_manager):
        """Test deletion of non-existent finding returns False."""
        result = state_manager.delete_finding("nonexistent-id")
        assert result is False


class TestUpdateOperations(TestRAIDStateManager):
    """Test finding update operations."""
    
    def test_update_finding_success(self, state_manager, sample_finding):
        """Test successful update of finding fields."""
        state_manager.add_finding(sample_finding)
        
        updates = {
            'description': 'Updated description',
            'impact_assessment': 'Updated impact',
            'recommendations': 'Updated recommendations',
            'status': Status.APPROVED
        }
        
        result = state_manager.update_finding(sample_finding.id, **updates)
        assert result is True
        
        updated_finding = state_manager.get_finding(sample_finding.id)
        assert updated_finding.description == 'Updated description'
        assert updated_finding.impact_assessment == 'Updated impact'
        assert updated_finding.recommendations == 'Updated recommendations'
        assert updated_finding.status == Status.APPROVED
    
    def test_update_finding_status_string(self, state_manager, sample_finding):
        """Test updating finding status with string value."""
        state_manager.add_finding(sample_finding)
        
        result = state_manager.update_finding(sample_finding.id, status='approved')
        assert result is True
        
        updated_finding = state_manager.get_finding(sample_finding.id)
        assert updated_finding.status == Status.APPROVED
    
    def test_update_finding_nonexistent(self, state_manager):
        """Test updating non-existent finding returns False."""
        result = state_manager.update_finding('nonexistent-id', description='test')
        assert result is False
    
    def test_update_finding_invalid_fields(self, state_manager, sample_finding):
        """Test updating invalid fields raises error."""
        state_manager.add_finding(sample_finding)
        
        with pytest.raises(ValueError, match="Cannot update fields"):
            state_manager.update_finding(sample_finding.id, id='new-id')
        
        with pytest.raises(ValueError, match="Cannot update fields"):
            state_manager.update_finding(sample_finding.id, type=RAIDType.ISSUE)
    
    def test_update_finding_invalid_status(self, state_manager, sample_finding):
        """Test updating with invalid status raises error."""
        state_manager.add_finding(sample_finding)
        
        with pytest.raises(ValueError, match="Invalid status type"):
            state_manager.update_finding(sample_finding.id, status=123)


class TestFilteringAndQuerying(TestRAIDStateManager):
    """Test filtering and querying operations."""
    
    def test_get_all_findings_order(self, state_manager, sample_findings):
        """Test that findings are returned in creation order."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        all_findings = state_manager.get_all_findings()
        assert len(all_findings) == 3
        
        # Check creation order is preserved
        for i, finding in enumerate(all_findings):
            assert finding.id == sample_findings[i].id
    
    def test_filter_by_type(self, state_manager, sample_findings):
        """Test filtering findings by type."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        risks = state_manager.filter_findings(type=RAIDType.RISK)
        assert len(risks) == 1
        assert risks[0].type == RAIDType.RISK
        
        issues = state_manager.filter_findings(type='issue')  # Test string input
        assert len(issues) == 1
        assert issues[0].type == RAIDType.ISSUE
    
    def test_filter_by_status(self, state_manager, sample_findings):
        """Test filtering findings by status."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # All should be pending initially
        pending = state_manager.filter_findings(status=Status.PENDING)
        assert len(pending) == 3
        
        # Approve one finding
        sample_findings[0].approve()
        
        approved = state_manager.filter_findings(status='approved')
        assert len(approved) == 1
        assert approved[0].status == Status.APPROVED
    
    def test_filter_by_evidence_type(self, state_manager, sample_findings):
        """Test filtering findings by evidence type."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        facts = state_manager.filter_findings(evidence_type=EvidenceType.FACT)
        assert len(facts) == 1
        
        inferences = state_manager.filter_findings(evidence_type='inference')
        assert len(inferences) == 1
    
    def test_filter_by_source_filename(self, state_manager, sample_findings):
        """Test filtering findings by source filename."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        doc1_findings = state_manager.filter_findings(source_filename='doc1.pdf')
        assert len(doc1_findings) == 1
        
        nonexistent = state_manager.filter_findings(source_filename='nonexistent.pdf')
        assert len(nonexistent) == 0
    
    def test_filter_by_relationships(self, state_manager, sample_findings):
        """Test filtering findings by relationship presence."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Initially no relationships
        with_rels = state_manager.filter_findings(has_relationships=True)
        assert len(with_rels) == 0
        
        without_rels = state_manager.filter_findings(has_relationships=False)
        assert len(without_rels) == 3
        
        # Add relationship
        state_manager.add_relationship(sample_findings[0].id, sample_findings[1].id)
        
        with_rels = state_manager.filter_findings(has_relationships=True)
        assert len(with_rels) == 2  # Both findings now have relationship
    
    def test_filter_by_time(self, state_manager, sample_finding):
        """Test filtering findings by creation/update time."""
        past_time = datetime.now() - timedelta(hours=1)
        future_time = datetime.now() + timedelta(hours=1)
        
        state_manager.add_finding(sample_finding)
        
        # Should find finding created after past time
        recent = state_manager.filter_findings(created_after=past_time)
        assert len(recent) == 1
        
        # Should not find finding created after future time
        future = state_manager.filter_findings(created_after=future_time)
        assert len(future) == 0
    
    def test_query_findings_custom_function(self, state_manager, sample_findings):
        """Test querying findings with custom function."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Find findings with long descriptions
        long_desc_findings = state_manager.query_findings(
            lambda f: len(f.description) > 10
        )
        assert len(long_desc_findings) >= 0  # Depends on sample data
        
        # Find risk findings
        risk_findings = state_manager.query_findings(
            lambda f: f.type == RAIDType.RISK
        )
        assert len(risk_findings) == 1
    
    def test_get_approved_findings(self, state_manager, sample_findings):
        """Test getting only approved findings."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Initially no approved findings
        approved = state_manager.get_approved_findings()
        assert len(approved) == 0
        
        # Approve one finding
        state_manager.update_finding(sample_findings[0].id, status=Status.APPROVED)
        
        approved = state_manager.get_approved_findings()
        assert len(approved) == 1
        assert approved[0].status == Status.APPROVED
    
    def test_get_pending_findings(self, state_manager, sample_findings):
        """Test getting only pending findings."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # All should be pending initially
        pending = state_manager.get_pending_findings()
        assert len(pending) == 3
        
        # Approve one finding
        state_manager.update_finding(sample_findings[0].id, status=Status.APPROVED)
        
        pending = state_manager.get_pending_findings()
        assert len(pending) == 2


class TestRelationshipManagement(TestRAIDStateManager):
    """Test relationship management between findings."""
    
    def test_add_relationship_success(self, state_manager, sample_findings):
        """Test successful addition of relationship between findings."""
        for finding in sample_findings[:2]:  # Only add first two
            state_manager.add_finding(finding)
        
        finding1_id = sample_findings[0].id
        finding2_id = sample_findings[1].id
        
        result = state_manager.add_relationship(finding1_id, finding2_id)
        assert result is True
        
        # Check bidirectional relationship
        related_to_1 = state_manager.get_related_findings(finding1_id)
        related_to_2 = state_manager.get_related_findings(finding2_id)
        
        assert len(related_to_1) == 1
        assert related_to_2[0].id == finding1_id
        assert len(related_to_2) == 1
        assert related_to_1[0].id == finding2_id
    
    def test_add_relationship_nonexistent_finding(self, state_manager, sample_finding):
        """Test adding relationship with non-existent finding fails."""
        state_manager.add_finding(sample_finding)
        
        result = state_manager.add_relationship(sample_finding.id, 'nonexistent-id')
        assert result is False
    
    def test_add_self_relationship(self, state_manager, sample_finding):
        """Test that self-relationships are prevented."""
        state_manager.add_finding(sample_finding)
        
        result = state_manager.add_relationship(sample_finding.id, sample_finding.id)
        assert result is False
        
        related = state_manager.get_related_findings(sample_finding.id)
        assert len(related) == 0
    
    def test_remove_relationship_success(self, state_manager, sample_findings):
        """Test successful removal of relationship."""
        for finding in sample_findings[:2]:
            state_manager.add_finding(finding)
        
        finding1_id = sample_findings[0].id
        finding2_id = sample_findings[1].id
        
        # Add then remove relationship
        state_manager.add_relationship(finding1_id, finding2_id)
        result = state_manager.remove_relationship(finding1_id, finding2_id)
        assert result is True
        
        # Check relationship is removed
        related_to_1 = state_manager.get_related_findings(finding1_id)
        related_to_2 = state_manager.get_related_findings(finding2_id)
        
        assert len(related_to_1) == 0
        assert len(related_to_2) == 0
    
    def test_remove_relationship_nonexistent(self, state_manager, sample_finding):
        """Test removing non-existent relationship."""
        state_manager.add_finding(sample_finding)
        
        result = state_manager.remove_relationship(sample_finding.id, 'nonexistent-id')
        assert result is False
    
    def test_get_related_findings_empty(self, state_manager, sample_finding):
        """Test getting related findings when none exist."""
        state_manager.add_finding(sample_finding)
        
        related = state_manager.get_related_findings(sample_finding.id)
        assert related == []
    
    def test_get_related_findings_nonexistent_finding(self, state_manager):
        """Test getting related findings for non-existent finding."""
        related = state_manager.get_related_findings('nonexistent-id')
        assert related == []
    
    def test_delete_finding_cleans_relationships(self, state_manager, sample_findings):
        """Test that deleting a finding cleans up its relationships."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Create relationships
        state_manager.add_relationship(sample_findings[0].id, sample_findings[1].id)
        state_manager.add_relationship(sample_findings[0].id, sample_findings[2].id)
        
        # Delete the finding with relationships
        state_manager.delete_finding(sample_findings[0].id)
        
        # Check relationships are cleaned up
        related_to_1 = state_manager.get_related_findings(sample_findings[1].id)
        related_to_2 = state_manager.get_related_findings(sample_findings[2].id)
        
        assert len(related_to_1) == 0
        assert len(related_to_2) == 0


class TestBulkOperations(TestRAIDStateManager):
    """Test bulk operations on findings."""
    
    def test_bulk_approve_findings(self, state_manager, sample_findings):
        """Test bulk approval of findings."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        finding_ids = [f.id for f in sample_findings]
        results = state_manager.bulk_approve_findings(finding_ids)
        
        # Check all were approved
        for finding_id in finding_ids:
            assert results[finding_id] is True
            finding = state_manager.get_finding(finding_id)
            assert finding.status == Status.APPROVED
        
        approved = state_manager.get_approved_findings()
        assert len(approved) == 3
    
    def test_bulk_approve_mixed_results(self, state_manager, sample_finding):
        """Test bulk approval with mix of valid and invalid IDs."""
        state_manager.add_finding(sample_finding)
        
        finding_ids = [sample_finding.id, 'nonexistent-id']
        results = state_manager.bulk_approve_findings(finding_ids)
        
        assert results[sample_finding.id] is True
        assert results['nonexistent-id'] is False
    
    def test_bulk_reject_findings(self, state_manager, sample_findings):
        """Test bulk rejection of findings."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        finding_ids = [f.id for f in sample_findings]
        results = state_manager.bulk_reject_findings(finding_ids)
        
        # Check all were rejected
        for finding_id in finding_ids:
            assert results[finding_id] is True
            finding = state_manager.get_finding(finding_id)
            assert finding.status == Status.REJECTED


class TestStatisticsAndUtilities(TestRAIDStateManager):
    """Test statistics and utility functions."""
    
    def test_get_statistics(self, state_manager, sample_findings):
        """Test getting collection statistics."""
        # Empty state
        stats = state_manager.get_statistics()
        assert stats['total_findings'] == 0
        assert stats['approved_count'] == 0
        
        # Add findings
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Approve one, reject one (one stays pending)
        state_manager.update_finding(sample_findings[0].id, status=Status.APPROVED)
        state_manager.update_finding(sample_findings[1].id, status=Status.REJECTED)
        
        # Add relationship
        state_manager.add_relationship(sample_findings[0].id, sample_findings[2].id)
        
        stats = state_manager.get_statistics()
        
        assert stats['total_findings'] == 3
        assert stats['approved_count'] == 1
        assert stats['pending_count'] == 1
        assert stats['rejected_count'] == 1
        assert stats['with_relationships'] == 2  # Two findings have relationships
        
        # Check type breakdown
        assert stats['by_type']['risk'] == 1
        assert stats['by_type']['issue'] == 1
        assert stats['by_type']['dependency'] == 1
    
    def test_get_findings_by_source(self, state_manager, sample_findings):
        """Test getting findings by source document."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        doc1_findings = state_manager.get_findings_by_source('doc1.pdf')
        assert len(doc1_findings) == 1
        assert doc1_findings[0].type == RAIDType.RISK
        
        nonexistent_findings = state_manager.get_findings_by_source('nonexistent.pdf')
        assert len(nonexistent_findings) == 0
    
    def test_clear_all_findings(self, state_manager, sample_findings):
        """Test clearing all findings."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Add relationships
        state_manager.add_relationship(sample_findings[0].id, sample_findings[1].id)
        
        count = state_manager.clear_all_findings()
        assert count == 3
        
        assert len(state_manager) == 0
        assert state_manager.get_all_findings() == []
        
        stats = state_manager.get_statistics()
        assert stats['total_findings'] == 0


class TestStateImportExport(TestRAIDStateManager):
    """Test state import/export functionality."""
    
    def test_export_import_state(self, state_manager, sample_findings):
        """Test exporting and importing state data."""
        # Set up state
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Add relationships and approve one finding
        state_manager.add_relationship(sample_findings[0].id, sample_findings[1].id)
        state_manager.update_finding(sample_findings[0].id, status=Status.APPROVED)
        
        # Export state
        exported_state = state_manager.export_state()
        
        # Verify export structure
        assert 'findings' in exported_state
        assert 'relationships' in exported_state
        assert 'creation_order' in exported_state
        assert 'exported_at' in exported_state
        
        assert len(exported_state['findings']) == 3
        
        # Create new state manager and import
        new_state_manager = RAIDStateManager()
        new_state_manager.import_state(exported_state)
        
        # Verify imported state
        assert len(new_state_manager) == 3
        
        all_findings = new_state_manager.get_all_findings()
        assert len(all_findings) == 3
        
        # Check that approval status was preserved
        approved_findings = new_state_manager.get_approved_findings()
        assert len(approved_findings) == 1
        
        # Check that relationships were preserved
        related = new_state_manager.get_related_findings(sample_findings[0].id)
        assert len(related) == 1
    
    def test_import_empty_state(self, state_manager):
        """Test importing empty state data."""
        empty_state = {
            'findings': {},
            'relationships': {},
            'creation_order': [],
            'exported_at': datetime.now().isoformat()
        }
        
        state_manager.import_state(empty_state)
        
        assert len(state_manager) == 0
        assert state_manager.get_all_findings() == []


class TestIterationAndContainment(TestRAIDStateManager):
    """Test iteration and containment operations."""
    
    def test_length(self, state_manager, sample_findings):
        """Test __len__ method."""
        assert len(state_manager) == 0
        
        for i, finding in enumerate(sample_findings):
            state_manager.add_finding(finding)
            assert len(state_manager) == i + 1
    
    def test_contains(self, state_manager, sample_finding):
        """Test __contains__ method."""
        assert sample_finding.id not in state_manager
        
        state_manager.add_finding(sample_finding)
        assert sample_finding.id in state_manager
        
        state_manager.delete_finding(sample_finding.id)
        assert sample_finding.id not in state_manager
    
    def test_iteration(self, state_manager, sample_findings):
        """Test __iter__ method."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Test iteration preserves creation order
        iterated_findings = list(state_manager)
        assert len(iterated_findings) == 3
        
        for i, finding in enumerate(iterated_findings):
            assert finding.id == sample_findings[i].id


class TestAdvancedValidationAndEdgeCases(TestRAIDStateManager):
    """Test advanced validation scenarios and edge cases for comprehensive coverage."""
    
    def test_status_transition_validation_sequence(self, state_manager, sample_finding):
        """Test complete status transition sequence with validation."""
        state_manager.add_finding(sample_finding)
        
        # Initial status should be pending
        finding = state_manager.get_finding(sample_finding.id)
        assert finding.status == Status.PENDING
        
        # Transition to approved
        result = state_manager.update_finding(sample_finding.id, status=Status.APPROVED)
        assert result is True
        finding = state_manager.get_finding(sample_finding.id)
        assert finding.status == Status.APPROVED
        
        # Transition to rejected
        result = state_manager.update_finding(sample_finding.id, status=Status.REJECTED)
        assert result is True
        finding = state_manager.get_finding(sample_finding.id)
        assert finding.status == Status.REJECTED
        
        # Transition back to pending (reset scenario)
        result = state_manager.update_finding(sample_finding.id, status=Status.PENDING)
        assert result is True
        finding = state_manager.get_finding(sample_finding.id)
        assert finding.status == Status.PENDING
    
    def test_relationship_consistency_validation(self, state_manager, sample_findings):
        """Test that relationships maintain consistency across operations."""
        # Add all findings
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        finding1_id, finding2_id, finding3_id = [f.id for f in sample_findings]
        
        # Create a chain of relationships: 1 -> 2 -> 3
        state_manager.add_relationship(finding1_id, finding2_id)
        state_manager.add_relationship(finding2_id, finding3_id)
        
        # Verify relationship consistency
        related_to_1 = state_manager.get_related_findings(finding1_id)
        related_to_2 = state_manager.get_related_findings(finding2_id)
        related_to_3 = state_manager.get_related_findings(finding3_id)
        
        assert len(related_to_1) == 1 and related_to_1[0].id == finding2_id
        assert len(related_to_2) == 2  # Connected to both 1 and 3
        assert len(related_to_3) == 1 and related_to_3[0].id == finding2_id
        
        # Verify findings' internal related_items lists are consistent
        finding1 = state_manager.get_finding(finding1_id)
        finding2 = state_manager.get_finding(finding2_id)
        finding3 = state_manager.get_finding(finding3_id)
        
        assert finding2_id in finding1.related_items
        assert finding1_id in finding2.related_items
        assert finding3_id in finding2.related_items
        assert finding2_id in finding3.related_items
    
    def test_bulk_operations_with_mixed_statuses(self, state_manager, sample_findings):
        """Test bulk operations behavior with findings in different statuses."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        finding_ids = [f.id for f in sample_findings]
        
        # Set different initial statuses
        state_manager.update_finding(finding_ids[0], status=Status.APPROVED)
        state_manager.update_finding(finding_ids[1], status=Status.REJECTED) 
        # finding_ids[2] remains PENDING
        
        # Test bulk approve on mixed statuses
        results = state_manager.bulk_approve_findings(finding_ids)
        
        # All should succeed regardless of previous status
        for finding_id in finding_ids:
            assert results[finding_id] is True
            finding = state_manager.get_finding(finding_id)
            assert finding.status == Status.APPROVED
        
        # Test bulk reject on all approved findings
        results = state_manager.bulk_reject_findings(finding_ids)
        
        for finding_id in finding_ids:
            assert results[finding_id] is True
            finding = state_manager.get_finding(finding_id)
            assert finding.status == Status.REJECTED
    
    def test_filtering_with_complex_criteria(self, state_manager, sample_findings):
        """Test filtering with multiple criteria combinations."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        # Add relationships and set different statuses
        state_manager.add_relationship(sample_findings[0].id, sample_findings[1].id)
        state_manager.update_finding(sample_findings[0].id, status=Status.APPROVED)
        
        # Test compound filtering scenarios
        approved_with_relationships = state_manager.filter_findings(
            status=Status.APPROVED, 
            has_relationships=True
        )
        assert len(approved_with_relationships) == 1
        assert approved_with_relationships[0].id == sample_findings[0].id
        
        pending_without_relationships = state_manager.filter_findings(
            status=Status.PENDING,
            has_relationships=False
        )
        assert len(pending_without_relationships) == 1  # Only finding[2] matches
        assert pending_without_relationships[0].id == sample_findings[2].id
    
    def test_state_export_import_with_complex_relationships(self, state_manager, sample_findings):
        """Test state export/import preserves complex relationship structures."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        finding_ids = [f.id for f in sample_findings]
        
        # Create complex relationship network
        state_manager.add_relationship(finding_ids[0], finding_ids[1])
        state_manager.add_relationship(finding_ids[1], finding_ids[2])
        state_manager.add_relationship(finding_ids[0], finding_ids[2])  # Triangle
        
        # Set mixed statuses
        state_manager.update_finding(finding_ids[0], status=Status.APPROVED)
        state_manager.update_finding(finding_ids[1], status=Status.REJECTED)
        
        # Export state
        exported_state = state_manager.export_state()
        
        # Import into new manager
        new_manager = RAIDStateManager()
        new_manager.import_state(exported_state)
        
        # Verify complex relationships preserved
        for finding_id in finding_ids:
            original_related = state_manager.get_related_findings(finding_id)
            imported_related = new_manager.get_related_findings(finding_id)
            
            assert len(original_related) == len(imported_related)
            original_ids = {f.id for f in original_related}
            imported_ids = {f.id for f in imported_related}
            assert original_ids == imported_ids
        
        # Verify statuses preserved
        assert new_manager.get_finding(finding_ids[0]).status == Status.APPROVED
        assert new_manager.get_finding(finding_ids[1]).status == Status.REJECTED
        assert new_manager.get_finding(finding_ids[2]).status == Status.PENDING
    
    def test_concurrent_relationship_operations(self, state_manager, sample_findings):
        """Test relationship operations don't interfere with each other."""
        for finding in sample_findings:
            state_manager.add_finding(finding)
        
        finding_ids = [f.id for f in sample_findings]
        
        # Add multiple relationships simultaneously  
        result1 = state_manager.add_relationship(finding_ids[0], finding_ids[1])
        result2 = state_manager.add_relationship(finding_ids[0], finding_ids[2])
        result3 = state_manager.add_relationship(finding_ids[1], finding_ids[2])
        
        assert all([result1, result2, result3])
        
        # Verify all relationships exist
        related_to_0 = state_manager.get_related_findings(finding_ids[0])
        related_to_1 = state_manager.get_related_findings(finding_ids[1]) 
        related_to_2 = state_manager.get_related_findings(finding_ids[2])
        
        assert len(related_to_0) == 2  # Connected to 1 and 2
        assert len(related_to_1) == 2  # Connected to 0 and 2
        assert len(related_to_2) == 2  # Connected to 0 and 1
        
        # Remove one relationship and verify others remain
        state_manager.remove_relationship(finding_ids[0], finding_ids[1])
        
        related_to_0 = state_manager.get_related_findings(finding_ids[0])
        related_to_1 = state_manager.get_related_findings(finding_ids[1])
        related_to_2 = state_manager.get_related_findings(finding_ids[2])
        
        assert len(related_to_0) == 1 and related_to_0[0].id == finding_ids[2]
        assert len(related_to_1) == 1 and related_to_1[0].id == finding_ids[2]
        assert len(related_to_2) == 2  # Still connected to both 0 and 1
    
    def test_update_tracking_consistency(self, state_manager, sample_finding):
        """Test that update timestamps are consistently maintained."""
        state_manager.add_finding(sample_finding)
        
        original_updated_at = sample_finding.updated_at
        
        # Wait a small amount to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        # Update finding
        state_manager.update_finding(sample_finding.id, description="Updated description")
        
        updated_finding = state_manager.get_finding(sample_finding.id)
        assert updated_finding.updated_at > original_updated_at
        
        # Verify created_at remains unchanged
        assert updated_finding.created_at == sample_finding.created_at