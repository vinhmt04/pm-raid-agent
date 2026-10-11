"""Unit tests for PDF document processor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
from io import BytesIO

# Import using sys.path manipulation to match main.py approach  
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from document_processing.pdf_processor import PDFProcessor
from document_processing.base import SupportedFormat, DocumentProcessingError


class TestPDFProcessor:
    """Test PDF document processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = PDFProcessor()
    
    def test_supported_format(self):
        """Test processor reports correct supported format."""
        assert self.processor.supported_format == SupportedFormat.PDF
    
    def test_can_parse_pdf_files(self):
        """Test can_parse method for PDF files."""
        assert self.processor.can_parse(Path("document.pdf")) is True
        assert self.processor.can_parse(Path("document.PDF")) is True
        assert self.processor.can_parse(Path("document.docx")) is False
        assert self.processor.can_parse(Path("document.xlsx")) is False
    
    @patch('document_processing.pdf_processor.PyPDF2.PdfReader')
    def test_successful_pdf_parsing(self, mock_pdf_reader):
        """Test successful PDF document parsing."""
        # Mock PDF reader and pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Content from page 1"
        mock_page2 = Mock()  
        mock_page2.extract_text.return_value = "Content from page 2"
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Create temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"fake pdf content")
        
        try:
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                result = self.processor.parse_document(tmp_path)
            
            # Verify results
            assert result.filename == tmp_path.name
            assert result.file_type == SupportedFormat.PDF
            assert result.processing_status == "success"
            assert result.error_message is None
            assert result.total_pages == 2
            assert len(result.content_sections) == 2
            
            # Check first page content
            page1_section = result.content_sections[0]
            assert page1_section.location == "page_1"
            assert page1_section.text == "Content from page 1"
            assert page1_section.metadata["page_number"] == 1
            assert page1_section.metadata["total_pages"] == 2
            
        finally:
            tmp_path.unlink()
    
    @patch('document_processing.pdf_processor.PyPDF2.PdfReader')
    def test_pdf_parsing_with_page_errors(self, mock_pdf_reader):
        """Test PDF parsing with some page extraction errors."""
        # Mock PDF reader with one successful and one failing page
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Content from page 1"
        mock_page2 = Mock()
        mock_page2.extract_text.side_effect = Exception("Page extraction failed")
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                result = self.processor.parse_document(tmp_path)
            
            # Should have partial success
            assert result.processing_status == "partial_success"
            assert "Failed to extract" in result.error_message
            assert result.total_pages == 2
            assert len(result.content_sections) == 2  # One success + one error section
            
            # Check that we got one successful section and one error section
            successful_section = result.content_sections[0] 
            assert successful_section.text == "Content from page 1"
            
            error_section = result.content_sections[1]
            assert error_section.location == "page_2_error"
            assert error_section.text == ""
            assert "extraction_error" in error_section.metadata
            
        finally:
            tmp_path.unlink()
    
    @patch('document_processing.pdf_processor.PyPDF2.PdfReader')
    def test_pdf_parsing_complete_failure(self, mock_pdf_reader):
        """Test PDF parsing when all pages fail to extract."""
        # Mock PDF reader where all pages fail
        mock_page1 = Mock()
        mock_page1.extract_text.side_effect = Exception("Page 1 failed")
        mock_page2 = Mock() 
        mock_page2.extract_text.side_effect = Exception("Page 2 failed")
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                with pytest.raises(DocumentProcessingError) as excinfo:
                    self.processor.parse_document(tmp_path)
                
                assert "Failed to extract any pages" in str(excinfo.value)
        finally:
            tmp_path.unlink()
    
    @patch('document_processing.pdf_processor.PyPDF2.PdfReader')
    def test_pdf_read_error_handling(self, mock_pdf_reader):
        """Test handling of PyPDF2 PdfReadError."""
        from PyPDF2.errors import PdfReadError
        mock_pdf_reader.side_effect = PdfReadError("Invalid PDF file")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                with pytest.raises(DocumentProcessingError) as excinfo:
                    self.processor.parse_document(tmp_path)
                
                assert "PDF read error" in str(excinfo.value)
                assert excinfo.value.file_path == tmp_path
        finally:
            tmp_path.unlink()
    
    def test_file_not_found_error(self):
        """Test handling of non-existent files."""
        non_existent_path = Path("nonexistent.pdf")
        
        with pytest.raises(DocumentProcessingError) as excinfo:
            self.processor.parse_document(non_existent_path)
        
        assert "PDF file not found" in str(excinfo.value)
    
    @patch('document_processing.pdf_processor.PyPDF2.PdfReader')
    def test_extract_metadata_only(self, mock_pdf_reader):
        """Test metadata-only extraction method."""
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [Mock(), Mock(), Mock()]  # 3 pages
        mock_reader_instance.metadata = {
            "/Title": "Test Document",
            "/Author": "Test Author"
        }
        mock_pdf_reader.return_value = mock_reader_instance
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"fake content")
        
        try:
            with patch("builtins.open", mock_open(read_data=b"fake pdf content")):
                metadata = self.processor.extract_metadata_only(tmp_path)
            
            assert metadata["num_pages"] == 3
            assert metadata["format"] == "PDF"
            assert metadata["title"] == "Test Document" 
            assert metadata["author"] == "Test Author"
            assert "file_size_bytes" in metadata
        finally:
            tmp_path.unlink()
