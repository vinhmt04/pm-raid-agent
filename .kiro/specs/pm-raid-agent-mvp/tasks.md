# Implementation Plan: PM RAID Agent MVP

## Overview

This implementation plan breaks down the PM RAID Agent MVP into sequential, testable coding tasks. The system uses LangGraph for workflow orchestration, configurable LLM integration for AI reasoning, and Streamlit for the user interface. The focus is on evidence-based RAID identification with cross-document synthesis and human approval workflow.

## Tasks

- [x] 1. Set up project structure and core dependencies
  - Create main project directory structure (src/, tests/, data/)
  - Set up pyproject.toml with all required dependencies (streamlit, langgraph, python-docx, PyPDF2, openpyxl, pandas)
  - Create environment configuration template (.env.example)
  - Set up basic logging configuration
  - _Requirements: 8.1, 8.4_

- [ ] 2. Implement document processing engine
  - [ ] 2.1 Create document parser interfaces and base classes
    - Define abstract DocumentParser base class
    - Create DocumentMetadata and ContentSection data models
    - Implement file format validation utilities
    - _Requirements: 1.1, 1.2_
  
  - [ ] 2.2 Implement PDF document processor
    - Create PDFProcessor class with PyPDF2 integration
    - Extract text content with page number tracking
    - Handle extraction errors gracefully with partial results
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ]* 2.3 Write unit tests for PDF processor
    - Test successful PDF text extraction
    - Test error handling for corrupted files
    - Test page number metadata tracking
    - _Requirements: 1.2, 1.3_
  
  - [ ] 2.4 Implement DOCX document processor
    - Create DOCXProcessor class with python-docx integration
    - Parse structured content with section identification
    - Extract text with document structure metadata
    - _Requirements: 1.1, 1.2_
  
  - [ ]* 2.5 Write unit tests for DOCX processor
    - Test structured content extraction
    - Test section identification accuracy
    - Test error handling for invalid files
    - _Requirements: 1.2, 1.3_
  
  - [ ] 2.6 Implement XLSX document processor
    - Create XLSXProcessor class with openpyxl integration
    - Handle multiple sheets with cell reference tracking
    - Extract textual content from spreadsheet cells
    - _Requirements: 1.1, 1.2_
  
  - [ ]* 2.7 Write unit tests for XLSX processor
    - Test multi-sheet processing
    - Test cell reference tracking
    - Test handling of various data types
    - _Requirements: 1.2, 1.3_

- [ ] 3. Checkpoint - Ensure document processing tests pass
  - Ensure all document processing tests pass, ask the user if questions arise.

- [ ] 4. Implement LLM provider integration layer
  - [ ] 4.1 Create LLM provider abstraction
    - Define abstract LLMProvider base class with standardized interface
    - Create LLMResponse and LLMConfig data models
    - Implement provider factory with environment-based selection
    - _Requirements: 2.2, 8.3_
  
  - [ ] 4.2 Implement selected LLM provider integration
    - Create provider-specific implementation for the configured LLM provider
    - Handle authentication, rate limiting, and error scenarios
    - Implement retry logic for transient failures
    - _Requirements: 2.2, 2.3_
  
  - [ ]* 4.3 Write unit tests for LLM provider integration
    - Test provider factory selection logic
    - Test OpenAI API integration with mocked responses
    - Test error handling and retry mechanisms
    - _Requirements: 2.2, 2.3_

- [ ] 5. Implement RAID data models and state management
  - [ ] 5.1 Create RAID finding data models
    - Define RAIDFinding dataclass with all required fields
    - Create RAIDType, EvidenceType, and Status enums
    - Implement SourceCitation model for evidence tracking
    - _Requirements: 3.1, 3.2, 4.1, 4.2, 5.2, 7.1_
  
  - [ ] 5.2 Implement RAID state management
    - Create RAIDStateManager class for finding lifecycle
    - Implement methods for adding, updating, and retrieving findings
    - Add relationship tracking between RAID items
    - _Requirements: 5.2, 5.3, 7.1, 7.2_
  
  - [ ]* 5.3 Write unit tests for RAID state management
    - Test finding creation and updates
    - Test relationship tracking functionality
    - Test status transitions and validation
    - _Requirements: 5.2, 5.3, 7.1, 7.2_

- [ ] 6. Implement LangGraph workflow orchestration
  - [ ] 6.1 Create workflow state and nodes structure
    - Define WorkflowState dataclass for LangGraph integration
    - Create workflow node functions for each analysis phase
    - Set up InMemorySaver checkpointer for session persistence
    - _Requirements: 2.1, 5.1_
  
  - [ ] 6.2 Implement individual document analysis node
    - Create analyze_individual_documents workflow node
    - Extract potential RAID items from single documents
    - Tag findings with source metadata and location tracking
    - _Requirements: 2.1, 2.4, 3.1_
  
  - [ ]* 6.3 Write property test for cross-document evidence synthesis
    - **Property 1: Cross-Document Evidence Synthesis**
    - **Validates: Requirements 3.3**
  
  - [ ] 6.4 Implement cross-document synthesis node
    - Create synthesize_cross_document workflow node
    - Analyze evidence across all documents before final findings
    - Detect relationships and contradictions between documents
    - _Requirements: 3.3, 7.1, 7.2_
  
  - [ ] 6.5 Implement evidence classification node
    - Create classify_evidence workflow node
    - Classify findings as fact, inference, or uncertainty
    - Link findings to supporting document evidence
    - _Requirements: 3.2, 3.4_
  
  - [ ]* 6.6 Write property test for evidence type classification
    - **Property 2: Evidence Type Classification**
    - **Validates: Requirements 3.2**
  
  - [ ] 6.7 Implement review and approval management node
    - Create manage_approval workflow node
    - Use interrupt() for PM review and Command(resume=...) for approval decisions
    - Track approval state independently of editing actions
    - _Requirements: 5.1, 5.3, 5.4_
  
  - [ ]* 6.8 Write property test for explicit approval requirement
    - **Property 6: Explicit Approval Requirement**
    - **Validates: Requirements 5.3, 5.4**

- [ ] 7. Checkpoint - Ensure LangGraph workflow tests pass
  - Ensure all LangGraph workflow tests pass, ask the user if questions arise.

- [ ] 8. Implement Excel export functionality
  - [ ] 8.1 Create Excel export engine
    - Implement ExcelExporter class with openpyxl integration
    - Generate single worksheet with all approved RAID items
    - Include comprehensive data: type, description, evidence, impact, recommendations, citations, relationships
    - _Requirements: 6.1, 6.2, 7.4_
  
  - [ ]* 8.2 Write property test for consolidated Excel export
    - **Property 5: Consolidated Excel Export**
    - **Validates: Requirements 6.1, 6.2, 7.4**
  
  - [ ] 8.3 Implement export filtering and validation
    - Filter to export only explicitly approved findings
    - Prevent export when no items are approved
    - Generate download links and confirmation messaging
    - _Requirements: 6.3, 6.4, 5.4_
  
  - [ ]* 8.4 Write unit tests for Excel export
    - Test approved item filtering
    - Test comprehensive data inclusion
    - Test export prevention with no approved items
    - _Requirements: 6.3, 6.4_

- [ ] 9. Implement Streamlit user interface
  - [ ] 9.1 Create main Streamlit application structure
    - Set up main app.py with page navigation
    - Implement session state management for workflow data
    - Create upload interface with multi-file support and format validation
    - _Requirements: 1.1, 1.4_
  
  - [ ] 9.2 Implement document processing interface
    - Create file upload handlers with error messaging
    - Display processing status and progress indicators
    - Show document summaries after successful processing
    - _Requirements: 1.3, 1.4_
  
  - [ ] 9.3 Implement RAID review and approval interface
    - Create finding display with approve/reject controls
    - Implement inline editing for descriptions and recommendations
    - Show relationship information and cross-references
    - _Requirements: 5.1, 5.2, 7.3_
  
  - [ ] 9.4 Implement export interface and controls
    - Create export button with approval validation
    - Generate download links for Excel files
    - Display export confirmation and status messaging
    - _Requirements: 6.3, 6.4_
  
  - [ ]* 9.5 Write integration tests for Streamlit UI
    - Test end-to-end workflow through UI components
    - Test error handling and user feedback
    - Test session state management
    - _Requirements: 1.3, 1.4, 5.1, 6.3_

- [ ] 10. Create sample data and end-to-end testing
  - [ ] 10.1 Create sample project documents
    - Generate sample PDF with project risks and issues
    - Create sample DOCX with project assumptions and dependencies
    - Build sample XLSX with project data and timeline conflicts
    - _Requirements: 8.4_
  
  - [ ] 10.2 Implement end-to-end test suite
    - Test complete workflow from upload to Excel export
    - Validate cross-document synthesis functionality
    - Test human approval interrupt/resume workflow
    - _Requirements: 8.4_
  
  - [ ]* 10.3 Write property test for single LLM provider integration
    - **Property 3: Single LLM Provider Integration**
    - **Validates: Requirements 2.2**
  
  - [ ]* 10.4 Write property test for session-scoped persistence
    - **Property 4: Session-Scoped Persistence**
    - **Validates: Security Requirements**

- [ ] 11. Integration and final validation
  - [ ] 11.1 Wire all components together in main application
    - Integrate document processing, LangGraph workflow, and Streamlit UI
    - Ensure proper error propagation and user feedback
    - Validate environment configuration and LLM provider selection
    - _Requirements: 2.1, 8.1, 8.3_
  
  - [ ] 11.2 Perform comprehensive system testing
    - Test complete RAID analysis workflow with real documents
    - Validate evidence-based findings and source citations
    - Test approval workflow and Excel export generation
    - _Requirements: 2.3, 3.1, 5.4, 6.2_
  
  - [ ]* 11.3 Write final integration tests
    - Test system with various document combinations
    - Validate LLM integration and error handling
    - Test export functionality with different approval scenarios
    - _Requirements: 2.2, 6.1, 6.4_

- [ ] 12. Final checkpoint - Ensure all tests pass and system is ready
  - Ensure all tests pass, validate sample documents work correctly, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout development
- Property tests validate universal correctness properties from the design
- Unit and integration tests validate specific examples and edge cases
- LLM integration is designed for loose coupling to support future provider additions
- InMemorySaver provides session-scoped checkpointing without persistent storage
- Focus remains on MVP scope with one real LLM provider implementation

```