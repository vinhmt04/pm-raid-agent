"""
Base classes and interfaces for document processing.
Defines abstract DocumentParser and data models for content extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import mimetypes
from enum import Enum
import sys

# Handle imports based on execution context
try:
    from logging_config import get_logger
except ImportError:
    # Fallback for direct execution or testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from logging_config import get_logger

logger = get_logger(__name__)


class SupportedFormat(Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx" 
    XLSX = "xlsx"


@dataclass
class ContentSection:
    """
    Represents a section of content extracted from a document.
    """
    location: str  # page/section/sheet reference
    text: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate content section data."""
        if not isinstance(self.text, str):
            self.text = str(self.text)
        if not isinstance(self.metadata, dict):
            self.metadata = {}


@dataclass
class DocumentMetadata:
    """
    Metadata for processed documents.
    """
    filename: str
    file_type: SupportedFormat
    content_sections: List[ContentSection]
    processing_status: str = "success"
    error_message: Optional[str] = None
    total_pages: Optional[int] = None
    total_sections: Optional[int] = None
    file_size_bytes: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived metadata."""
        if self.total_sections is None:
            self.total_sections = len(self.content_sections)


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    Defines the interface that all document processors must implement.
    """
    
    @property
    @abstractmethod
    def supported_format(self) -> SupportedFormat:
        """Return the format this parser supports."""
        pass
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this parser can handle the file
        """
        pass
    
    @abstractmethod
    def parse_document(self, file_path: Path) -> DocumentMetadata:
        """
        Parse a document and extract its content.
        
        Args:
            file_path: Path to the document to parse
            
        Returns:
            DocumentMetadata with extracted content and metadata
            
        Raises:
            DocumentProcessingError: If parsing fails
        """
        pass
    
    def _create_error_metadata(self, filename: str, error_message: str) -> DocumentMetadata:
        """
        Create metadata for failed parsing attempts.
        
        Args:
            filename: Name of the file that failed
            error_message: Error description
            
        Returns:
            DocumentMetadata with error information
        """
        return DocumentMetadata(
            filename=filename,
            file_type=self.supported_format,
            content_sections=[],
            processing_status="error",
            error_message=error_message
        )


class DocumentProcessingError(Exception):
    """Exception raised during document processing."""
    
    def __init__(self, message: str, file_path: Optional[Path] = None, original_error: Optional[Exception] = None):
        self.message = message
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(self.message)


class FileFormatValidator:
    """
    Utility class for validating file formats and types.
    """
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": SupportedFormat.PDF,
        ".docx": SupportedFormat.DOCX,
        ".xlsx": SupportedFormat.XLSX
    }
    
    SUPPORTED_MIME_TYPES = {
        "application/pdf": SupportedFormat.PDF,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": SupportedFormat.DOCX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": SupportedFormat.XLSX
    }
    
    @classmethod
    def validate_file_format(cls, file_path: Path) -> SupportedFormat:
        """
        Validate and determine the format of a file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            SupportedFormat for the file
            
        Raises:
            DocumentProcessingError: If format is not supported
        """
        if not file_path.exists():
            raise DocumentProcessingError(f"File does not exist: {file_path}")
        
        # Check by extension first
        extension = file_path.suffix.lower()
        if extension in cls.SUPPORTED_EXTENSIONS:
            logger.debug("File format detected by extension", file=str(file_path), format=extension)
            return cls.SUPPORTED_EXTENSIONS[extension]
        
        # Check by MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type in cls.SUPPORTED_MIME_TYPES:
            logger.debug("File format detected by MIME type", file=str(file_path), mime_type=mime_type)
            return cls.SUPPORTED_MIME_TYPES[mime_type]
        
        # Format not supported
        supported_formats = list(cls.SUPPORTED_EXTENSIONS.keys())
        raise DocumentProcessingError(
            f"Unsupported file format: {extension}. Supported formats: {supported_formats}",
            file_path=file_path
        )
    
    @classmethod
    def is_supported_format(cls, file_path: Path) -> bool:
        """
        Check if a file format is supported without raising exceptions.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if format is supported, False otherwise
        """
        try:
            cls.validate_file_format(file_path)
            return True
        except DocumentProcessingError:
            return False
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions (without dots)
        """
        return [ext[1:] for ext in cls.SUPPORTED_EXTENSIONS.keys()]
