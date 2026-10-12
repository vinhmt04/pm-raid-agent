"""
RAID State Management for PM RAID Agent MVP.
Manages the lifecycle and relationships of RAID findings collections.
"""

from typing import List, Dict, Optional, Set, Any, Callable
from collections import defaultdict
from datetime import datetime
import sys
from pathlib import Path
from copy import deepcopy

# Handle imports based on execution context
try:
    from src.raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, Status, SourceCitation
    from src.logging_config import get_logger
except ImportError:
    # Fallback for direct execution or testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from raid_models.raid_finding import RAIDFinding, RAIDType, EvidenceType, Status, SourceCitation
    from logging_config import get_logger

logger = get_logger(__name__)


class RAIDStateManager:
    """
    Manages state and operations for RAID findings collections.
    
    Provides functionality for:
    - Adding, updating, and deleting RAID findings
    - Managing relationships between findings
    - Filtering and querying findings by various criteria
    - Tracking status transitions and validation
    - Session-scoped persistence (in-memory only)
    """
    
    def __init__(self):
        """Initialize the RAID state manager with empty collections."""
        self._findings: Dict[str, RAIDFinding] = {}
        self._relationships: Dict[str, Set[str]] = defaultdict(set)  # bidirectional relationships
        self._creation_order: List[str] = []  # track insertion order
        
        logger.info("RAIDStateManager initialized")
    
    def add_finding(self, finding: RAIDFinding) -> str:
        """
        Add a new RAID finding to the collection.
        
        Args:
            finding: RAIDFinding instance to add
            
        Returns:
            The ID of the added finding
            
        Raises:
            ValueError: If finding with same ID already exists
        """
        if not isinstance(finding, RAIDFinding):
            raise ValueError("finding must be a RAIDFinding instance")
        
        if finding.id in self._findings:
            raise ValueError(f"Finding with ID {finding.id} already exists")
        
        # Store the finding
        self._findings[finding.id] = finding
        self._creation_order.append(finding.id)
        
        # Initialize relationships set for this finding
        self._relationships[finding.id] = set()
        
        # Add any pre-existing related items
        for related_id in finding.related_items:
            self._add_relationship(finding.id, related_id)
        
        logger.info("RAID finding added", 
                   finding_id=finding.id,
                   finding_type=finding.type.value,
                   total_findings=len(self._findings))
        
        return finding.id
    
    def get_finding(self, finding_id: str) -> Optional[RAIDFinding]:
        """
        Retrieve a RAID finding by ID.
        
        Args:
            finding_id: ID of the finding to retrieve
            
        Returns:
            RAIDFinding instance or None if not found
        """
        return self._findings.get(finding_id)
    
    def update_finding(self, finding_id: str, **updates) -> bool:
        """
        Update specific fields of a RAID finding.
        
        Args:
            finding_id: ID of the finding to update
            **updates: Field updates (description, impact_assessment, recommendations, status)
            
        Returns:
            True if update was successful, False if finding not found
            
        Raises:
            ValueError: If trying to update read-only fields or invalid values
        """
        finding = self._findings.get(finding_id)
        if not finding:
            return False
        
        # Define updatable fields
        updatable_fields = {
            'description', 'impact_assessment', 'recommendations', 'status'
        }
        
        # Check for invalid field updates
        invalid_fields = set(updates.keys()) - updatable_fields
        if invalid_fields:
            raise ValueError(f"Cannot update fields: {invalid_fields}. "
                           f"Updatable fields are: {updatable_fields}")
        
        # Apply updates using the finding's methods
        if 'description' in updates:
            finding.update_description(updates['description'])
        
        if 'impact_assessment' in updates:
            finding.update_impact_assessment(updates['impact_assessment'])
        
        if 'recommendations' in updates:
            finding.update_recommendations(updates['recommendations'])
        
        if 'status' in updates:
            new_status = updates['status']
            if isinstance(new_status, str):
                new_status = Status(new_status.lower())
            elif not isinstance(new_status, Status):
                raise ValueError(f"Invalid status type: {type(new_status)}")
            
            # Use appropriate method based on status
            if new_status == Status.APPROVED:
                finding.approve()
            elif new_status == Status.REJECTED:
                finding.reject()
            elif new_status == Status.PENDING:
                finding.reset_to_pending()
        
        logger.info("RAID finding updated",
                   finding_id=finding_id,
                   updated_fields=list(updates.keys()))
        
        return True
    
    def delete_finding(self, finding_id: str) -> bool:
        """
        Delete a RAID finding and clean up its relationships.
        
        Args:
            finding_id: ID of the finding to delete
            
        Returns:
            True if deletion was successful, False if finding not found
        """
        if finding_id not in self._findings:
            return False
        
        # Remove from all relationships
        related_ids = self._relationships[finding_id].copy()
        for related_id in related_ids:
            self._remove_relationship(finding_id, related_id)
        
        # Remove the finding itself
        del self._findings[finding_id]
        del self._relationships[finding_id]
        
        # Remove from creation order
        if finding_id in self._creation_order:
            self._creation_order.remove(finding_id)
        
        logger.info("RAID finding deleted",
                   finding_id=finding_id,
                   total_findings=len(self._findings))
        
        return True
    
    def get_all_findings(self) -> List[RAIDFinding]:
        """
        Get all RAID findings in creation order.
        
        Returns:
            List of all RAIDFinding instances
        """
        return [self._findings[finding_id] for finding_id in self._creation_order 
                if finding_id in self._findings]
    
    def filter_findings(self, **criteria) -> List[RAIDFinding]:
        """
        Filter RAID findings by various criteria.
        
        Args:
            **criteria: Filter criteria (type, status, evidence_type, etc.)
            
        Returns:
            List of RAIDFinding instances matching criteria
        """
        findings = self.get_all_findings()
        
        for key, value in criteria.items():
            if key == 'type':
                if isinstance(value, str):
                    value = RAIDType(value.lower())
                findings = [f for f in findings if f.type == value]
            
            elif key == 'status':
                if isinstance(value, str):
                    value = Status(value.lower())
                findings = [f for f in findings if f.status == value]
            
            elif key == 'evidence_type':
                if isinstance(value, str):
                    value = EvidenceType(value.lower())
                findings = [f for f in findings if f.evidence_type == value]
            
            elif key == 'has_relationships':
                if value:
                    findings = [f for f in findings if f.related_items]
                else:
                    findings = [f for f in findings if not f.related_items]
            
            elif key == 'source_filename':
                findings = [f for f in findings 
                          if any(citation.filename == value for citation in f.source_citations)]
            
            elif key == 'created_after':
                findings = [f for f in findings if f.created_at > value]
            
            elif key == 'updated_after':
                findings = [f for f in findings if f.updated_at > value]
            
            else:
                logger.warning("Unknown filter criteria", key=key, value=value)
        
        logger.debug("Findings filtered",
                    criteria=criteria,
                    result_count=len(findings))
        
        return findings
    
    def get_approved_findings(self) -> List[RAIDFinding]:
        """
        Get all approved RAID findings for export.
        
        Returns:
            List of approved RAIDFinding instances
        """
        return self.filter_findings(status=Status.APPROVED)
    
    def get_pending_findings(self) -> List[RAIDFinding]:
        """
        Get all pending RAID findings requiring review.
        
        Returns:
            List of pending RAIDFinding instances
        """
        return self.filter_findings(status=Status.PENDING)
    
    def add_relationship(self, finding_id1: str, finding_id2: str) -> bool:
        """
        Add a bidirectional relationship between two findings.
        
        Args:
            finding_id1: ID of first finding
            finding_id2: ID of second finding
            
        Returns:
            True if relationship was added, False if findings don't exist
        """
        if finding_id1 not in self._findings or finding_id2 not in self._findings:
            return False
        
        if finding_id1 == finding_id2:
            logger.warning("Attempted to create self-relationship", finding_id=finding_id1)
            return False
        
        self._add_relationship(finding_id1, finding_id2)
        return True
    
    def remove_relationship(self, finding_id1: str, finding_id2: str) -> bool:
        """
        Remove a bidirectional relationship between two findings.
        
        Args:
            finding_id1: ID of first finding
            finding_id2: ID of second finding
            
        Returns:
            True if relationship was removed, False if not found
        """
        if finding_id1 not in self._findings or finding_id2 not in self._findings:
            return False
        
        self._remove_relationship(finding_id1, finding_id2)
        return True
    
    def get_related_findings(self, finding_id: str) -> List[RAIDFinding]:
        """
        Get all findings related to the specified finding.
        
        Args:
            finding_id: ID of the finding
            
        Returns:
            List of related RAIDFinding instances
        """
        if finding_id not in self._findings:
            return []
        
        related_ids = self._relationships[finding_id]
        return [self._findings[related_id] for related_id in related_ids 
                if related_id in self._findings]
    
    def query_findings(self, query_func: Callable[[RAIDFinding], bool]) -> List[RAIDFinding]:
        """
        Query findings using a custom function.
        
        Args:
            query_func: Function that takes a RAIDFinding and returns bool
            
        Returns:
            List of RAIDFinding instances matching the query
        """
        return [finding for finding in self.get_all_findings() if query_func(finding)]
    
    def get_findings_by_source(self, filename: str) -> List[RAIDFinding]:
        """
        Get all findings that cite a specific source document.
        
        Args:
            filename: Name of the source document
            
        Returns:
            List of RAIDFinding instances citing the source
        """
        return self.filter_findings(source_filename=filename)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current RAID findings collection.
        
        Returns:
            Dictionary with various statistics
        """
        findings = self.get_all_findings()
        
        stats = {
            'total_findings': len(findings),
            'by_type': defaultdict(int),
            'by_status': defaultdict(int),
            'by_evidence_type': defaultdict(int),
            'with_relationships': 0,
            'approved_count': 0,
            'pending_count': 0,
            'rejected_count': 0
        }
        
        for finding in findings:
            stats['by_type'][finding.type.value] += 1
            stats['by_status'][finding.status.value] += 1
            stats['by_evidence_type'][finding.evidence_type.value] += 1
            
            if finding.related_items:
                stats['with_relationships'] += 1
        
        stats['approved_count'] = stats['by_status']['approved']
        stats['pending_count'] = stats['by_status']['pending']
        stats['rejected_count'] = stats['by_status']['rejected']
        
        return dict(stats)  # Convert defaultdict to regular dict
    
    def bulk_approve_findings(self, finding_ids: List[str]) -> Dict[str, bool]:
        """
        Approve multiple findings in bulk.
        
        Args:
            finding_ids: List of finding IDs to approve
            
        Returns:
            Dictionary mapping finding IDs to success status
        """
        results = {}
        
        for finding_id in finding_ids:
            finding = self._findings.get(finding_id)
            if finding:
                finding.approve()
                results[finding_id] = True
            else:
                results[finding_id] = False
        
        approved_count = sum(results.values())
        logger.info("Bulk approval completed",
                   requested_count=len(finding_ids),
                   approved_count=approved_count)
        
        return results
    
    def bulk_reject_findings(self, finding_ids: List[str]) -> Dict[str, bool]:
        """
        Reject multiple findings in bulk.
        
        Args:
            finding_ids: List of finding IDs to reject
            
        Returns:
            Dictionary mapping finding IDs to success status
        """
        results = {}
        
        for finding_id in finding_ids:
            finding = self._findings.get(finding_id)
            if finding:
                finding.reject()
                results[finding_id] = True
            else:
                results[finding_id] = False
        
        rejected_count = sum(results.values())
        logger.info("Bulk rejection completed",
                   requested_count=len(finding_ids),
                   rejected_count=rejected_count)
        
        return results
    
    def clear_all_findings(self) -> int:
        """
        Clear all findings from the state manager.
        
        Returns:
            Number of findings that were cleared
        """
        count = len(self._findings)
        
        self._findings.clear()
        self._relationships.clear()
        self._creation_order.clear()
        
        logger.info("All findings cleared", cleared_count=count)
        
        return count
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export the current state for persistence or backup.
        
        Returns:
            Dictionary containing the complete state
        """
        return {
            'findings': {finding_id: finding.to_dict() 
                        for finding_id, finding in self._findings.items()},
            'relationships': {finding_id: list(related_ids) 
                           for finding_id, related_ids in self._relationships.items()},
            'creation_order': self._creation_order.copy(),
            'exported_at': datetime.now().isoformat()
        }
    
    def import_state(self, state_data: Dict[str, Any]) -> None:
        """
        Import state from exported data.
        
        Args:
            state_data: State data from export_state()
        """
        # Clear current state
        self.clear_all_findings()
        
        # Import findings
        findings_data = state_data.get('findings', {})
        for finding_id, finding_dict in findings_data.items():
            finding = RAIDFinding.from_dict(finding_dict)
            # Temporarily clear related items to avoid relationship conflicts
            related_items = finding.related_items.copy()
            finding.related_items = []
            
            self._findings[finding_id] = finding
        
        # Import relationships
        relationships_data = state_data.get('relationships', {})
        for finding_id, related_ids in relationships_data.items():
            if finding_id in self._findings:
                self._relationships[finding_id] = set(related_ids)
                # Update the finding's related_items list
                if finding_id in self._findings:
                    self._findings[finding_id].related_items = list(related_ids)
        
        # Import creation order
        self._creation_order = state_data.get('creation_order', list(self._findings.keys()))
        
        logger.info("State imported",
                   findings_count=len(self._findings),
                   relationships_count=sum(len(rels) for rels in self._relationships.values()) // 2)
    
    def _add_relationship(self, finding_id1: str, finding_id2: str) -> None:
        """Internal method to add bidirectional relationship."""
        # Add to relationships tracking
        self._relationships[finding_id1].add(finding_id2)
        self._relationships[finding_id2].add(finding_id1)
        
        # Update the findings' related_items lists
        if finding_id1 in self._findings:
            self._findings[finding_id1].add_related_item(finding_id2)
        
        if finding_id2 in self._findings:
            self._findings[finding_id2].add_related_item(finding_id1)
        
        logger.debug("Relationship added",
                    finding_id1=finding_id1,
                    finding_id2=finding_id2)
    
    def _remove_relationship(self, finding_id1: str, finding_id2: str) -> None:
        """Internal method to remove bidirectional relationship."""
        # Remove from relationships tracking
        self._relationships[finding_id1].discard(finding_id2)
        self._relationships[finding_id2].discard(finding_id1)
        
        # Update the findings' related_items lists
        if finding_id1 in self._findings:
            self._findings[finding_id1].remove_related_item(finding_id2)
        
        if finding_id2 in self._findings:
            self._findings[finding_id2].remove_related_item(finding_id1)
        
        logger.debug("Relationship removed",
                    finding_id1=finding_id1,
                    finding_id2=finding_id2)
    
    def __len__(self) -> int:
        """Return the number of findings in the collection."""
        return len(self._findings)
    
    def __contains__(self, finding_id: str) -> bool:
        """Check if a finding ID exists in the collection."""
        return finding_id in self._findings
    
    def __iter__(self):
        """Iterate over findings in creation order."""
        for finding_id in self._creation_order:
            if finding_id in self._findings:
                yield self._findings[finding_id]