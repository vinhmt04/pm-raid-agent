"""
Unit tests for RAID finding data models.
Tests RAIDFinding, SourceCitation, and related enums.
"""

import pytest
from datetime import datetime
from unittest.mock import patch
import uuid

from src.raid_models import (
    RAIDFinding,
    RAIDType,
    EvidenceType,
    Status,
    SourceCitation
)


class TestSourceCitation:
    """Test cases for SourceCitation model."""
    
    def test_valid_source_citation_creation(self):
        """Test creating a valid source citation."""
        citation = SourceCitation(
            filename="project_plan.pdf",
            location="Page 5, Section 2.1",
            excerpt="Risk of budget overrun due to vendor delays"
        )
        
        assert citation.filename == "project_plan.pdf"
        assert citation.location == "Page 5, Section 2.1"
        assert citation.excerpt == "Risk of budget overrun due to vendor delays"
    
    def test_source_citation_validation_empty_filename(self):
        """Test source citation validation with empty filename."""
        with pytest.raises(ValueError, match="Filename is required"):
            SourceCitation(
                filename="",
                location="Page 1",
                excerpt="Some text"
            )
    
    def test_source_citation_validation_none_filename(self):
        """Test source citation validation with None filename."""
        with pytest.raises(ValueError, match="Filename is required"):
            SourceCitation(
                filename=None,
                location="Page 1", 
                excerpt="Some text"
            )
    
    def test_source_citation_validation_empty_location(self):
        """Test source citation validation with empty location."""
        with pytest.raises(ValueError, match="Location is required"):
            SourceCitation(
                filename="test.pdf",
                location="",
                excerpt="Some text"
            )
    
    def test_source_citation_validation_none_location(self):
        """Test source citation validation with None location."""
        with pytest.raises(ValueError, match="Location is required"):
            SourceCitation(
                filename="test.pdf",
                location=None,
                excerpt="Some text"
            )
    
    def test_source_citation_excerpt_conversion(self):
        """Test source citation excerpt type conversion."""
        citation = SourceCitation(
            filename="test.pdf",
            location="Page 1",
            excerpt=None
        )
        
        assert citation.excerpt == ""
    
    def test_source_citation_string_representation(self):
        """Test source citation string representation."""
        citation = SourceCitation(
            filename="test.pdf",
            location="Page 1",
            excerpt="Short text"
        )
        
        expected = "test.pdf (Page 1): Short text"
        assert str(citation) == expected
    
    def test_source_citation_string_representation_long_excerpt(self):
        """Test source citation string representation with long excerpt."""
        long_excerpt = "This is a very long excerpt that should be truncated when displayed in string representation"
        citation = SourceCitation(
            filename="test.pdf",
            location="Page 1",
            excerpt=long_excerpt
        )
        
        result = str(citation)
        assert "test.pdf (Page 1):" in result
        assert "..." in result
        assert len(result) < len(long_excerpt)


class TestRAIDFindingEnums:
    """Test cases for RAID finding enums."""
    
    def test_raid_type_enum_values(self):
        """Test RAIDType enum has correct values."""
        assert RAIDType.RISK.value == "risk"
        assert RAIDType.ASSUMPTION.value == "assumption"
        assert RAIDType.ISSUE.value == "issue"
        assert RAIDType.DEPENDENCY.value == "dependency"
    
    def test_evidence_type_enum_values(self):
        """Test EvidenceType enum has correct values."""
        assert EvidenceType.FACT.value == "fact"
        assert EvidenceType.INFERENCE.value == "inference"
        assert EvidenceType.UNCERTAINTY.value == "uncertainty"
    
    def test_status_enum_values(self):
        """Test Status enum has correct values."""
        assert Status.PENDING.value == "pending"
        assert Status.APPROVED.value == "approved"
        assert Status.REJECTED.value == "rejected"


class TestRAIDFinding:
    """Test cases for RAIDFinding model."""
    
    @pytest.fixture
    def sample_citation(self):
        """Sample source citation for testing."""
        return SourceCitation(
            filename="project_plan.pdf",
            location="Page 5",
            excerpt="Budget may exceed allocated amount"
        )
    
    @pytest.fixture
    def sample_finding(self, sample_citation):
        """Sample RAID finding for testing."""
        return RAIDFinding(
            type=RAIDType.RISK,
            description="Risk of budget overrun",
            evidence_type=EvidenceType.FACT,
            source_citations=[sample_citation],
            impact_assessment="High impact on project delivery",
            recommendations="Monitor budget closely"
        )
    
    def test_valid_raid_finding_creation(self, sample_citation):
        """Test creating a valid RAID finding."""
        finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Risk of budget overrun",
            evidence_type=EvidenceType.FACT,
            source_citations=[sample_citation]
        )
        
        assert finding.type == RAIDType.RISK
        assert finding.description == "Risk of budget overrun"
        assert finding.evidence_type == EvidenceType.FACT
        assert len(finding.source_citations) == 1
        assert finding.status == Status.PENDING  # default
        assert finding.impact_assessment == ""  # default
        assert finding.recommendations == ""  # default
        assert finding.related_items == []  # default
        assert isinstance(finding.id, str)
        assert isinstance(finding.created_at, datetime)
        assert isinstance(finding.updated_at, datetime)
    
    def test_raid_finding_string_type_conversion(self, sample_citation):
        """Test RAID finding creation with string type values."""
        finding = RAIDFinding(
            type="risk",  # string instead of enum
            description="Risk description",
            evidence_type="fact",  # string instead of enum
            source_citations=[sample_citation],
            status="pending"  # string instead of enum
        )
        
        assert finding.type == RAIDType.RISK
        assert finding.evidence_type == EvidenceType.FACT
        assert finding.status == Status.PENDING
    
    def test_raid_finding_invalid_type(self, sample_citation):
        """Test RAID finding validation with invalid type."""
        with pytest.raises(ValueError, match="Invalid RAID type"):
            RAIDFinding(
                type="invalid_type",
                description="Description",
                evidence_type=EvidenceType.FACT,
                source_citations=[sample_citation]
            )
    
    def test_raid_finding_invalid_evidence_type(self, sample_citation):
        """Test RAID finding validation with invalid evidence type."""
        with pytest.raises(ValueError, match="Invalid evidence type"):
            RAIDFinding(
                type=RAIDType.RISK,
                description="Description",
                evidence_type="invalid_evidence",
                source_citations=[sample_citation]
            )
    
    def test_raid_finding_invalid_status(self, sample_citation):
        """Test RAID finding validation with invalid status."""
        with pytest.raises(ValueError, match="Invalid status"):
            RAIDFinding(
                type=RAIDType.RISK,
                description="Description",
                evidence_type=EvidenceType.FACT,
                source_citations=[sample_citation],
                status="invalid_status"
            )
    
    def test_raid_finding_empty_description(self, sample_citation):
        """Test RAID finding validation with empty description."""
        with pytest.raises(ValueError, match="Description is required"):
            RAIDFinding(
                type=RAIDType.RISK,
                description="",
                evidence_type=EvidenceType.FACT,
                source_citations=[sample_citation]
            )
    
    def test_raid_finding_no_source_citations(self):
        """Test RAID finding validation with no source citations."""
        with pytest.raises(ValueError, match="At least one source citation is required"):
            RAIDFinding(
                type=RAIDType.RISK,
                description="Description",
                evidence_type=EvidenceType.FACT,
                source_citations=[]
            )
    
    def test_raid_finding_dict_source_citation_conversion(self):
        """Test RAID finding conversion of dict source citations."""
        finding = RAIDFinding(
            type=RAIDType.RISK,
            description="Description",
            evidence_type=EvidenceType.FACT,
            source_citations=[{
                'filename': 'test.pdf',
                'location': 'Page 1',
                'excerpt': 'Some text'
            }]
        )
        
        assert len(finding.source_citations) == 1
        assert isinstance(finding.source_citations[0], SourceCitation)
        assert finding.source_citations[0].filename == 'test.pdf'
    
    def test_update_description(self, sample_finding):
        """Test updating finding description."""
        original_updated_at = sample_finding.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        sample_finding.update_description("New description")
        
        assert sample_finding.description == "New description"
        assert sample_finding.updated_at > original_updated_at
    
    def test_update_description_empty(self, sample_finding):
        """Test updating finding with empty description."""
        with pytest.raises(ValueError, match="Description must be a non-empty string"):
            sample_finding.update_description("")
    
    def test_update_impact_assessment(self, sample_finding):
        """Test updating impact assessment."""
        original_updated_at = sample_finding.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        sample_finding.update_impact_assessment("New impact assessment")
        
        assert sample_finding.impact_assessment == "New impact assessment"
        assert sample_finding.updated_at > original_updated_at
    
    def test_update_recommendations(self, sample_finding):
        """Test updating recommendations."""
        original_updated_at = sample_finding.updated_at
        
        # Small delay to ensure timestamp difference  
        import time
        time.sleep(0.01)
        
        sample_finding.update_recommendations("New recommendations")
        
        assert sample_finding.recommendations == "New recommendations"
        assert sample_finding.updated_at > original_updated_at
    
    def test_approve_finding(self, sample_finding):
        """Test approving a finding."""
        assert sample_finding.status == Status.PENDING
        
        sample_finding.approve()
        
        assert sample_finding.status == Status.APPROVED
    
    def test_reject_finding(self, sample_finding):
        """Test rejecting a finding."""
        assert sample_finding.status == Status.PENDING
        
        sample_finding.reject()
        
        assert sample_finding.status == Status.REJECTED
    
    def test_reset_to_pending(self, sample_finding):
        """Test resetting finding to pending status."""
        sample_finding.approve()
        assert sample_finding.status == Status.APPROVED
        
        sample_finding.reset_to_pending()
        
        assert sample_finding.status == Status.PENDING
    
    def test_add_related_item(self, sample_finding):
        """Test adding related item ID."""
        related_id = "related-finding-123"
        
        sample_finding.add_related_item(related_id)
        
        assert related_id in sample_finding.related_items
    
    def test_add_duplicate_related_item(self, sample_finding):
        """Test adding duplicate related item ID."""
        related_id = "related-finding-123"
        
        sample_finding.add_related_item(related_id)
        sample_finding.add_related_item(related_id)  # duplicate
        
        assert sample_finding.related_items.count(related_id) == 1
    
    def test_remove_related_item(self, sample_finding):
        """Test removing related item ID."""
        related_id = "related-finding-123"
        
        sample_finding.add_related_item(related_id)
        assert related_id in sample_finding.related_items
        
        sample_finding.remove_related_item(related_id)
        assert related_id not in sample_finding.related_items
    
    def test_add_source_citation(self, sample_finding):
        """Test adding additional source citation."""
        new_citation = SourceCitation(
            filename="additional_doc.docx",
            location="Section 3",
            excerpt="Supporting evidence"
        )
        
        original_count = len(sample_finding.source_citations)
        sample_finding.add_source_citation(new_citation)
        
        assert len(sample_finding.source_citations) == original_count + 1
        assert new_citation in sample_finding.source_citations
    
    def test_add_invalid_source_citation(self, sample_finding):
        """Test adding invalid source citation."""
        with pytest.raises(ValueError, match="Citation must be a SourceCitation object"):
            sample_finding.add_source_citation("not a citation")
    
    def test_get_primary_source(self, sample_finding):
        """Test getting primary source citation."""
        primary = sample_finding.get_primary_source()
        
        assert primary == sample_finding.source_citations[0]
    
    def test_get_primary_source_empty(self):
        """Test getting primary source when no citations exist."""
        # Create finding without post_init validation for this test
        finding = RAIDFinding.__new__(RAIDFinding)
        finding.source_citations = []
        
        assert finding.get_primary_source() is None
    
    def test_get_summary(self, sample_finding):
        """Test getting finding summary."""
        summary = sample_finding.get_summary()
        
        expected_keys = {
            'id', 'type', 'description', 'evidence_type', 'status',
            'impact_assessment', 'recommendations', 'source_count',
            'related_items_count', 'created_at', 'updated_at'
        }
        
        assert set(summary.keys()) == expected_keys
        assert summary['type'] == 'risk'
        assert summary['source_count'] == 1
        assert summary['related_items_count'] == 0
    
    def test_to_dict(self, sample_finding):
        """Test converting finding to dictionary."""
        finding_dict = sample_finding.to_dict()
        
        expected_keys = {
            'id', 'type', 'description', 'evidence_type', 'source_citations',
            'impact_assessment', 'recommendations', 'status', 'related_items',
            'created_at', 'updated_at', 'metadata'
        }
        
        assert set(finding_dict.keys()) == expected_keys
        assert finding_dict['type'] == 'risk'
        assert isinstance(finding_dict['source_citations'], list)
        assert len(finding_dict['source_citations']) == 1
        assert isinstance(finding_dict['source_citations'][0], dict)
    
    def test_from_dict(self, sample_citation):
        """Test creating finding from dictionary."""
        finding_data = {
            'id': 'test-id-123',
            'type': 'risk',
            'description': 'Test description',
            'evidence_type': 'fact',
            'source_citations': [{
                'filename': 'test.pdf',
                'location': 'Page 1',
                'excerpt': 'Test excerpt'
            }],
            'impact_assessment': 'Test impact',
            'recommendations': 'Test recommendations',
            'status': 'pending',
            'related_items': ['related-1'],
            'created_at': '2023-01-01T12:00:00',
            'updated_at': '2023-01-01T12:30:00',
            'metadata': {'test': 'value'}
        }
        
        finding = RAIDFinding.from_dict(finding_data)
        
        assert finding.id == 'test-id-123'
        assert finding.type == RAIDType.RISK
        assert finding.description == 'Test description'
        assert finding.evidence_type == EvidenceType.FACT
        assert len(finding.source_citations) == 1
        assert finding.source_citations[0].filename == 'test.pdf'
        assert finding.status == Status.PENDING
        assert finding.related_items == ['related-1']
        assert finding.metadata == {'test': 'value'}
    
    def test_from_dict_minimal(self):
        """Test creating finding from minimal dictionary."""
        finding_data = {
            'type': 'issue',
            'description': 'Test issue',
            'evidence_type': 'inference',
            'source_citations': [{
                'filename': 'test.pdf',
                'location': 'Page 1', 
                'excerpt': 'Test excerpt'
            }]
        }
        
        finding = RAIDFinding.from_dict(finding_data)
        
        assert finding.type == RAIDType.ISSUE
        assert finding.evidence_type == EvidenceType.INFERENCE
        assert finding.status == Status.PENDING  # default
        assert finding.related_items == []  # default
        assert isinstance(finding.id, str)  # auto-generated
    
    def test_string_representation(self, sample_finding):
        """Test string representation of finding."""
        result = str(sample_finding)
        
        assert "RAIDFinding(RISK:" in result
        assert "Risk of budget overrun" in result
    
    def test_string_representation_long_description(self, sample_citation):
        """Test string representation with long description."""
        long_description = "This is a very long description that should be truncated in the string representation"
        finding = RAIDFinding(
            type=RAIDType.RISK,
            description=long_description,
            evidence_type=EvidenceType.FACT,
            source_citations=[sample_citation]
        )
        
        result = str(finding)
        assert "..." in result
        assert len(result) < len(long_description)
    
    def test_repr_representation(self, sample_finding):
        """Test detailed representation of finding."""
        result = repr(sample_finding)
        
        assert "RAIDFinding(id=" in result
        assert "type=RAIDType.RISK" in result
        assert "status=Status.PENDING" in result
        assert "evidence_type=EvidenceType.FACT" in result