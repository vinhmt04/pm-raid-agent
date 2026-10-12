"""
RAID finding data models for PM RAID Agent MVP.
Defines the core data structures for RAID items, evidence tracking, and source citations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import sys
from pathlib import Path

# Handle imports based on execution context
try:
    from logging_config import get_logger
except ImportError:
    # Fallback for direct execution or testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from logging_config import get_logger

logger = get_logger(__name__)


class RAIDType(Enum):
    """Types of RAID findings according to RAID methodology."""
    RISK = "risk"
    ASSUMPTION = "assumption"
    ISSUE = "issue"
    DEPENDENCY = "dependency"


class EvidenceType(Enum):
    """Classification of evidence supporting RAID findings."""
    FACT = "fact"  # Directly documented in source materials
    INFERENCE = "inference"  # Reasonably derived from available evidence
    UNCERTAINTY = "uncertainty"  # Insufficient evidence for confident classification


class Status(Enum):
    """Status of RAID findings in the review and approval workflow."""
    PENDING = "pending"  # Awaiting PM review
    APPROVED = "approved"  # Approved by PM for inclusion in final report
    REJECTED = "rejected"  # Rejected by PM, excluded from final report


@dataclass
class SourceCitation:
    """
    Citation information for RAID finding evidence.
    Links findings to specific locations in source documents.
    """
    filename: str
    location: str  # page/section/sheet reference
    excerpt: str  # relevant text excerpt from source
    
    def __post_init__(self):
        """Validate source citation data."""
        if not self.filename or not isinstance(self.filename, str):
            raise ValueError("Filename is required and must be a non-empty string")
        
        if not self.location or not isinstance(self.location, str):
            raise ValueError("Location is required and must be a non-empty string")
        
        if not isinstance(self.excerpt, str):
            self.excerpt = str(self.excerpt) if self.excerpt else ""
    
    def __str__(self) -> str:
        """String representation for display purposes."""
        excerpt_preview = self.excerpt[:50] + "..." if len(self.excerpt) > 50 else self.excerpt
        return f"{self.filename} ({self.location}): {excerpt_preview}"


@dataclass
class RAIDFinding:
    """
    Core RAID finding data model.
    Represents identified Risks, Assumptions, Issues, or Dependencies with evidence tracking.
    """
    type: RAIDType
    description: str
    evidence_type: EvidenceType
    source_citations: List[SourceCitation]
    impact_assessment: str = ""
    recommendations: str = ""
    status: Status = Status.PENDING
    related_items: List[str] = field(default_factory=list)  # IDs of related findings
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate RAID finding data and handle type conversions."""
        # Validate required fields
        if not isinstance(self.type, RAIDType):
            if isinstance(self.type, str):
                try:
                    self.type = RAIDType(self.type.lower())
                except ValueError:
                    raise ValueError(f"Invalid RAID type: {self.type}. Must be one of: {[t.value for t in RAIDType]}")
            else:
                raise ValueError("RAID type must be RAIDType enum or string")
        
        if not isinstance(self.evidence_type, EvidenceType):
            if isinstance(self.evidence_type, str):
                try:
                    self.evidence_type = EvidenceType(self.evidence_type.lower())
                except ValueError:
                    raise ValueError(f"Invalid evidence type: {self.evidence_type}. Must be one of: {[t.value for t in EvidenceType]}")
            else:
                raise ValueError("Evidence type must be EvidenceType enum or string")
        
        if not isinstance(self.status, Status):
            if isinstance(self.status, str):
                try:
                    self.status = Status(self.status.lower())
                except ValueError:
                    raise ValueError(f"Invalid status: {self.status}. Must be one of: {[s.value for s in Status]}")
            else:
                raise ValueError("Status must be Status enum or string")
        
        # Validate description
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Description is required and must be a non-empty string")
        
        # Validate source citations
        if not self.source_citations:
            raise ValueError("At least one source citation is required")
        
        if not isinstance(self.source_citations, list):
            raise ValueError("Source citations must be a list")
        
        # Convert string citations to SourceCitation objects if needed
        converted_citations = []
        for citation in self.source_citations:
            if isinstance(citation, dict):
                converted_citations.append(SourceCitation(**citation))
            elif isinstance(citation, SourceCitation):
                converted_citations.append(citation)
            else:
                raise ValueError("Source citations must be SourceCitation objects or dicts")
        
        self.source_citations = converted_citations
        
        # Validate related items
        if not isinstance(self.related_items, list):
            self.related_items = []
        
        # Ensure all related items are strings
        self.related_items = [str(item) for item in self.related_items]
        
        # Initialize metadata if not provided
        if self.metadata is None:
            self.metadata = {}
        
        # Update timestamp
        self.updated_at = datetime.now()
        
        logger.debug("RAIDFinding created/validated",
                    id=self.id,
                    type=self.type.value,
                    evidence_type=self.evidence_type.value,
                    status=self.status.value)
    
    def update_description(self, new_description: str) -> None:
        """
        Update the description and refresh timestamp.
        
        Args:
            new_description: New description text
        """
        if not new_description or not isinstance(new_description, str):
            raise ValueError("Description must be a non-empty string")
        
        self.description = new_description
        self.updated_at = datetime.now()
        
        logger.debug("RAIDFinding description updated", id=self.id)
    
    def update_impact_assessment(self, new_impact: str) -> None:
        """
        Update the impact assessment and refresh timestamp.
        
        Args:
            new_impact: New impact assessment text
        """
        self.impact_assessment = new_impact or ""
        self.updated_at = datetime.now()
        
        logger.debug("RAIDFinding impact assessment updated", id=self.id)
    
    def update_recommendations(self, new_recommendations: str) -> None:
        """
        Update the recommendations and refresh timestamp.
        
        Args:
            new_recommendations: New recommendations text
        """
        self.recommendations = new_recommendations or ""
        self.updated_at = datetime.now()
        
        logger.debug("RAIDFinding recommendations updated", id=self.id)
    
    def approve(self) -> None:
        """Mark finding as approved for inclusion in final report."""
        old_status = self.status
        self.status = Status.APPROVED
        self.updated_at = datetime.now()
        
        logger.info("RAIDFinding approved", 
                   id=self.id,
                   old_status=old_status.value,
                   new_status=self.status.value)
    
    def reject(self) -> None:
        """Mark finding as rejected, excluding from final report."""
        old_status = self.status
        self.status = Status.REJECTED
        self.updated_at = datetime.now()
        
        logger.info("RAIDFinding rejected",
                   id=self.id,
                   old_status=old_status.value,
                   new_status=self.status.value)
    
    def reset_to_pending(self) -> None:
        """Reset finding status to pending for re-review."""
        old_status = self.status
        self.status = Status.PENDING
        self.updated_at = datetime.now()
        
        logger.info("RAIDFinding reset to pending",
                   id=self.id,
                   old_status=old_status.value,
                   new_status=self.status.value)
    
    def add_related_item(self, item_id: str) -> None:
        """
        Add a related RAID finding ID.
        
        Args:
            item_id: ID of related RAID finding
        """
        item_id = str(item_id)
        if item_id not in self.related_items:
            self.related_items.append(item_id)
            self.updated_at = datetime.now()
            
            logger.debug("Related item added to RAIDFinding",
                        id=self.id,
                        related_item_id=item_id)
    
    def remove_related_item(self, item_id: str) -> None:
        """
        Remove a related RAID finding ID.
        
        Args:
            item_id: ID of related RAID finding to remove
        """
        item_id = str(item_id)
        if item_id in self.related_items:
            self.related_items.remove(item_id)
            self.updated_at = datetime.now()
            
            logger.debug("Related item removed from RAIDFinding",
                        id=self.id,
                        related_item_id=item_id)
    
    def add_source_citation(self, citation: SourceCitation) -> None:
        """
        Add additional source citation.
        
        Args:
            citation: Source citation to add
        """
        if not isinstance(citation, SourceCitation):
            raise ValueError("Citation must be a SourceCitation object")
        
        self.source_citations.append(citation)
        self.updated_at = datetime.now()
        
        logger.debug("Source citation added to RAIDFinding",
                    id=self.id,
                    citation_filename=citation.filename,
                    citation_location=citation.location)
    
    def get_primary_source(self) -> Optional[SourceCitation]:
        """
        Get the primary (first) source citation.
        
        Returns:
            Primary source citation or None if no citations exist
        """
        return self.source_citations[0] if self.source_citations else None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary dictionary suitable for display or export.
        
        Returns:
            Dictionary with key finding information
        """
        return {
            'id': self.id,
            'type': self.type.value,
            'description': self.description,
            'evidence_type': self.evidence_type.value,
            'status': self.status.value,
            'impact_assessment': self.impact_assessment,
            'recommendations': self.recommendations,
            'source_count': len(self.source_citations),
            'related_items_count': len(self.related_items),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RAIDFinding to dictionary for serialization.
        
        Returns:
            Dictionary representation of the finding
        """
        return {
            'id': self.id,
            'type': self.type.value,
            'description': self.description,
            'evidence_type': self.evidence_type.value,
            'source_citations': [
                {
                    'filename': citation.filename,
                    'location': citation.location,
                    'excerpt': citation.excerpt
                }
                for citation in self.source_citations
            ],
            'impact_assessment': self.impact_assessment,
            'recommendations': self.recommendations,
            'status': self.status.value,
            'related_items': self.related_items,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAIDFinding':
        """
        Create RAIDFinding from dictionary.
        
        Args:
            data: Dictionary containing finding data
            
        Returns:
            RAIDFinding instance
        """
        # Convert timestamps
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        # Convert source citations
        citations_data = data.get('source_citations', [])
        source_citations = [
            SourceCitation(**citation) if isinstance(citation, dict) else citation
            for citation in citations_data
        ]
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=data['type'],
            description=data['description'],
            evidence_type=data['evidence_type'],
            source_citations=source_citations,
            impact_assessment=data.get('impact_assessment', ''),
            recommendations=data.get('recommendations', ''),
            status=data.get('status', Status.PENDING),
            related_items=data.get('related_items', []),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation for display purposes."""
        return f"RAIDFinding({self.type.value.upper()}: {self.description[:50]}{'...' if len(self.description) > 50 else ''})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"RAIDFinding(id='{self.id}', type={self.type}, "
                f"status={self.status}, evidence_type={self.evidence_type})")