# PM RAID Agent MVP - Design Document

## Architecture Overview

The PM RAID Agent MVP is a standalone Python application that uses LangGraph to orchestrate document analysis and RAID identification. The system processes multiple document formats through a workflow that extracts content, performs cross-document analysis, and generates evidence-based findings for PM review.

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   LangGraph     │    │   LLM Provider  │
│   Interface     │◄───┤   Workflow      │◄───┤   Integration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   RAID State    │    │   Evidence      │
│   Processing    │    │   Management    │    │   Synthesis     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Design

### Document Processing Engine

Handles multi-format document ingestion and content extraction:

- **PDF Processing**: Extract text with page number tracking
- **DOCX Processing**: Parse structured content with section identification  
- **XLSX Processing**: Handle multiple sheets with cell reference tracking
- **Content Normalization**: Standardize extracted text for analysis

*Validates: Requirements 1.1, 1.2*

### LangGraph Workflow Orchestration

Manages the analysis pipeline through discrete workflow nodes:

**Node 1: Individual Document Analysis**
- Extract content from each uploaded document
- Identify potential RAID items within single documents
- Tag findings with source document and location metadata

**Node 2: Cross-Document Synthesis** 
- Synthesize evidence across all uploaded documents before producing final findings
- Detect relationships and contradictions between documents
- Resolve conflicts and merge related findings
- Generate comprehensive RAID assessment

**Node 3: Evidence Classification**
- Classify findings using `evidence_type: fact | inference | uncertainty`
- Remove confidence scoring approach in favor of explicit evidence typing
- Link each finding to supporting document evidence

**Node 4: Review and Approval Management**
- Present findings with `pending | approved | rejected` status
- Track approval state independently of editing actions
- Prevent export of non-approved items
- Use `interrupt()` to pause for PM review and `Command(resume=...)` to resume with approval decisions, maintaining the same `thread_id` throughout the session

*Validates: Requirements 2.1, 2.2, 3.3, 5.1, 5.4*

### LLM Provider Integration

Implements loosely coupled integration for LLM reasoning:

- **Environment Configuration**: Select provider through environment variables
- **Single Provider Implementation**: Implement one real LLM provider for MVP
- **Abstraction Layer**: Design interface to support future provider additions
- **Error Handling**: Graceful fallback for API failures

*Validates: Requirements 2.2, 2.3*

### RAID State Management

Manages finding lifecycle and relationships:

```
RAID Item Structure:
- id: unique identifier
- type: risk | assumption | issue | dependency  
- description: PM-editable content
- evidence_type: fact | inference | uncertainty
- source_citations: [filename, location, excerpt]
- impact_assessment: PM-editable analysis
- recommendations: PM-editable actions
- status: pending | approved | rejected
- related_items: [cross-references]
```

*Validates: Requirements 3.1, 3.2, 4.1, 4.2, 4.4, 5.2, 5.3, 7.1, 7.2*

### Checkpoint Persistence

Uses local LangGraph checkpointer for workflow state management:

- **Local Storage**: LangGraph InMemorySaver (in-memory checkpointing) for interrupt/resume capability
- **Session Scope**: Document content and checkpoint data retained only during active session
- **Security Compliance**: No data persistence beyond application session
- **Restart Limitation**: No recovery across application restarts

*Validates: Security Requirements*

### Excel Export Engine

Generates consolidated Excel exports of approved findings:

- **Single Worksheet Design**: Consolidate all approved RAID items in one sheet
- **Comprehensive Data**: Include type, description, evidence, impact, recommendations, source citations, and related item references
- **Approval Filtering**: Export only explicitly approved findings
- **Structured Format**: Organize data for PM workflow integration

*Validates: Requirements 6.1, 6.2, 6.3, 6.4, 7.4*

## Data Models

### Document Metadata
```
{
  filename: string,
  file_type: "pdf" | "docx" | "xlsx",
  content_sections: [{
    location: string,  // page/section/sheet reference
    text: string,
    metadata: object
  }]
}
```

### RAID Finding
```
{
  id: string,
  type: "risk" | "assumption" | "issue" | "dependency",
  description: string,
  evidence_type: "fact" | "inference" | "uncertainty",
  source_citations: [{
    filename: string,
    location: string,
    excerpt: string
  }],
  impact_assessment: string,
  recommendations: string,
  status: "pending" | "approved" | "rejected",
  related_items: [string]  // IDs of related findings
}
```

## Interface Design

### Streamlit UI Flow

1. **Upload Interface**: Multi-file upload with format validation
2. **Processing Status**: Real-time workflow progress indicators
3. **Review Interface**: Finding display with approve/reject controls
4. **Edit Capability**: Inline editing of descriptions and recommendations
5. **Export Controls**: Download generation for approved items only

*Validates: Requirements 1.4, 5.1, 5.2, 6.3*

## Error Handling

### Document Processing Errors
- Invalid file format handling with clear messaging
- Corrupted file detection and user notification
- Extraction failure recovery with partial results

### LLM Integration Errors  
- API timeout handling with user feedback
- Rate limit management with retry logic
- Service unavailability graceful degradation

### Workflow State Errors
- Checkpoint corruption detection and recovery
- Invalid state transition prevention
- Data consistency validation

*Validates: Requirements 1.3, Reliability Requirements*

## Security Considerations

### Credential Management
- Environment variable configuration for API keys
- No hardcoded credentials in source code
- Secure handling of sensitive configuration

### Data Privacy
- No persistent document storage beyond session
- In-memory processing with session cleanup
- Local-only checkpoint storage with session scope

*Validates: Security Requirements*

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Cross-Document Evidence Synthesis

*For any* set of uploaded documents, the LLM must analyze evidence across all documents before producing final RAID findings, ensuring comprehensive cross-document reasoning rather than isolated document analysis.

**Validates: Requirements 3.3**

### Property 2: Evidence Type Classification

*For any* identified RAID finding, the system shall classify evidence as fact, inference, or uncertainty without using confidence scoring approaches.

**Validates: Requirements 3.2**

### Property 3: Single LLM Provider Integration

*For any* configured LLM provider, the system shall successfully invoke exactly one real provider while maintaining loose coupling for future provider additions.

**Validates: Requirements 2.2**

### Property 4: Session-Scoped Persistence

*For any* uploaded document content and checkpoint data, retention shall not extend beyond the active session, ensuring compliance with security requirements.

**Validates: Security Requirements**

### Property 5: Consolidated Excel Export

*For any* approved RAID items, the Excel export shall contain exactly one worksheet with all required data including source citations, recommendations, and related item references.

**Validates: Requirements 6.1, 6.2, 7.4**

### Property 6: Explicit Approval Requirement

*For any* RAID finding, the status shall remain pending until explicitly approved, and editing actions shall not automatically change approval state.

**Validates: Requirements 5.3, 5.4**