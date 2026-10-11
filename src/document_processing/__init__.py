"""
Document processing package for PM RAID Agent MVP.
Provides unified document processing for PDF, DOCX, and XLSX formats.
"""

from .base import (
    DocumentParser,
    DocumentMetadata, 
    ContentSection,
    SupportedFormat,
    DocumentProcessingError,
    FileFormatValidator
)
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor  
from .xlsx_processor import XLSXProcessor
from .factory import DocumentProcessorFactory, document_factory

__all__ = [
    # Base classes and interfaces
    "DocumentParser",
    "DocumentMetadata",
    "ContentSection", 
    "SupportedFormat",
    "DocumentProcessingError",
    "FileFormatValidator",
    
    # Processor implementations
    "PDFProcessor",
    "DOCXProcessor",
    "XLSXProcessor",
    
    # Factory and utilities
    "DocumentProcessorFactory",
    "document_factory"
]

# Version info
__version__ = "0.1.0"
