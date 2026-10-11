"""Unit tests for base document processing classes and interfaces."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

# Import using sys.path manipulation to match main.py approach
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from document_processing.base import (
    DocumentParser, DocumentMetadata, ContentSection, SupportedFormat,
    DocumentProcessingError, FileFormatValidator
)


class TestContentSection:
    """Test ContentSection data class."""
    
    def test_content_section_creation(self):
        """Test basic ContentSection creation."""
        section = ContentSection(
            location="page_1",
            text="Test content",
            metadata={"page": 1}
        )
        
        assert section.location == "page_1"
        assert section.text == "Test content"
        assert section.metadata == {"page": 1}
    
    def test_content_section_text_conversion(self):
        """Test automatic text conversion."""
        section = ContentSection(
            location="test",
            text=123,  # Non-string value
            metadata={}
        )
        
        assert section.text == "123"
        assert isinstance(section.text, str)
    
    def test_content_section_metadata_default(self):
        """Test default metadata handling."""
        section = ContentSection(
            location="test",
            text="content",
            metadata=None  # Non-dict value
        )
        
        assert section.metadata == {}


class TestDocumentMetadata:
    """Test DocumentMetadata data class."""
    
    def test_document_metadata_creation(self):
        """Test basic DocumentMetadata creation."""
        sections = [
            ContentSection("page_1", "Content 1", {}),
            ContentSection("page_2", "Content 2", {})
        ]
        
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type=SupportedFormat.PDF,
            content_sections=sections
        )
        
        assert metadata.filename == "test.pdf"
        assert metadata.file_type == SupportedFormat.PDF
        assert len(metadata.content_sections) == 2
        assert metadata.processing_status == "success"
        assert metadata.total_sections == 2
    
    def test_document_metadata_error_case(self):
        """Test DocumentMetadata for error cases."""
        metadata = DocumentMetadata(
            filename="error.pdf",
            file_type=SupportedFormat.PDF,
            content_sections=[],
            processing_status="error",
            error_message="Processing failed"
        )
        
        assert metadata.processing_status == "error"
        assert metadata.error_message == "Processing failed"
        assert metadata.total_sections == 0


class TestSupportedFormat:
    """Test SupportedFormat enum."""
    
    def test_supported_formats(self):
        """Test all supported format values."""
        assert SupportedFormat.PDF.value == "pdf"
        assert SupportedFormat.DOCX.value == "docx"
        assert SupportedFormat.XLSX.value == "xlsx"


class TestDocumentProcessingError:
    """Test DocumentProcessingError exception class."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = DocumentProcessingError("Test error")
        assert str(error) == "Test error"
        assert error.file_path is None
        assert error.original_error is None
    
    def test_error_with_file_path(self):
        """Test error with file path."""
        file_path = Path("test.pdf")
        error = DocumentProcessingError("Test error", file_path=file_path)
        assert error.file_path == file_path
    
    def test_error_with_original_exception(self):
        """Test error wrapping another exception."""
        original = ValueError("Original error")
        error = DocumentProcessingError("Wrapper error", original_error=original)
        assert error.original_error == original


class TestFileFormatValidator:
    """Test FileFormatValidator utility class."""
    
    def test_get_supported_formats(self):
        """Test getting supported format list."""
        formats = FileFormatValidator.get_supported_formats()
        assert "pdf" in formats
        assert "docx" in formats
        assert "xlsx" in formats
        assert len(formats) == 3
    
    def test_validate_file_format_by_extension(self):
        """Test format validation by extension."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            format_type = FileFormatValidator.validate_file_format(tmp_path)
            assert format_type == SupportedFormat.PDF
        finally:
            tmp_path.unlink()
    
    def test_validate_unsupported_format(self):
        """Test validation of unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(DocumentProcessingError) as excinfo:
                FileFormatValidator.validate_file_format(tmp_path)
            assert "Unsupported file format" in str(excinfo.value)
        finally:
            tmp_path.unlink()
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        fake_path = Path("nonexistent.pdf")
        with pytest.raises(DocumentProcessingError) as excinfo:
            FileFormatValidator.validate_file_format(fake_path)
        assert "File does not exist" in str(excinfo.value)
    
    def test_is_supported_format(self):
        """Test format support checking without exceptions."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            assert FileFormatValidator.is_supported_format(tmp_path) is True
        finally:
            tmp_path.unlink()
        
        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            assert FileFormatValidator.is_supported_format(tmp_path) is False
        finally:
            tmp_path.unlink()


class MockDocumentParser(DocumentParser):
    """Mock implementation of DocumentParser for testing."""
    
    @property
    def supported_format(self) -> SupportedFormat:
        return SupportedFormat.PDF
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix == ".pdf"
    
    def parse_document(self, file_path: Path) -> DocumentMetadata:
        return DocumentMetadata(
            filename=file_path.name,
            file_type=SupportedFormat.PDF,
            content_sections=[
                ContentSection("page_1", "Mock content", {})
            ]
        )


class TestDocumentParser:
    """Test abstract DocumentParser base class."""
    
    def test_mock_parser_implementation(self):
        """Test that mock parser implements interface correctly."""
        parser = MockDocumentParser()
        
        assert parser.supported_format == SupportedFormat.PDF
        assert parser.can_parse(Path("test.pdf")) is True
        assert parser.can_parse(Path("test.docx")) is False
    
    def test_create_error_metadata(self):
        """Test error metadata creation helper."""
        parser = MockDocumentParser()
        error_metadata = parser._create_error_metadata("test.pdf", "Test error")
        
        assert error_metadata.filename == "test.pdf"
        assert error_metadata.processing_status == "error"
        assert error_metadata.error_message == "Test error"
        assert len(error_metadata.content_sections) == 0
