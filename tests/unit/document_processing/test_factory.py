"""Unit tests for document processor factory."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Import using sys.path manipulation to match main.py approach  
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from document_processing.factory import DocumentProcessorFactory
from document_processing.base import SupportedFormat, DocumentProcessingError
from document_processing.pdf_processor import PDFProcessor
from document_processing.docx_processor import DOCXProcessor
from document_processing.xlsx_processor import XLSXProcessor


class TestDocumentProcessorFactory:
    """Test document processor factory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = DocumentProcessorFactory()
    
    def test_factory_initialization(self):
        """Test factory initializes with all processors."""
        assert len(self.factory._processors) == 3
        assert SupportedFormat.PDF in self.factory._processors
        assert SupportedFormat.DOCX in self.factory._processors
        assert SupportedFormat.XLSX in self.factory._processors
        
        # Check processor types
        assert isinstance(self.factory._processors[SupportedFormat.PDF], PDFProcessor)
        assert isinstance(self.factory._processors[SupportedFormat.DOCX], DOCXProcessor)
        assert isinstance(self.factory._processors[SupportedFormat.XLSX], XLSXProcessor)
    
    def test_get_processor_for_pdf(self):
        """Test getting PDF processor."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            processor = self.factory.get_processor(tmp_path)
            assert isinstance(processor, PDFProcessor)
            assert processor.supported_format == SupportedFormat.PDF
        finally:
            tmp_path.unlink()
    
    def test_get_processor_for_docx(self):
        """Test getting DOCX processor."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            processor = self.factory.get_processor(tmp_path)
            assert isinstance(processor, DOCXProcessor)
            assert processor.supported_format == SupportedFormat.DOCX
        finally:
            tmp_path.unlink()
    
    def test_get_processor_for_xlsx(self):
        """Test getting XLSX processor."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            processor = self.factory.get_processor(tmp_path)
            assert isinstance(processor, XLSXProcessor)
            assert processor.supported_format == SupportedFormat.XLSX
        finally:
            tmp_path.unlink()
    
    def test_get_processor_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            with pytest.raises(DocumentProcessingError) as excinfo:
                self.factory.get_processor(tmp_path)
            
            assert "Unsupported file format" in str(excinfo.value)
        finally:
            tmp_path.unlink()
    
    def test_get_processor_nonexistent_file(self):
        """Test error handling for non-existent files."""
        fake_path = Path("nonexistent.pdf")
        
        with pytest.raises(DocumentProcessingError) as excinfo:
            self.factory.get_processor(fake_path)
        
        assert "File does not exist" in str(excinfo.value)
    
    def test_can_process_supported_formats(self):
        """Test can_process method for supported formats."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)
        
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            docx_path = Path(tmp_file.name)
        
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            xlsx_path = Path(tmp_file.name)
        
        try:
            assert self.factory.can_process(pdf_path) is True
            assert self.factory.can_process(docx_path) is True
            assert self.factory.can_process(xlsx_path) is True
        finally:
            pdf_path.unlink()
            docx_path.unlink()
            xlsx_path.unlink()
    
    def test_can_process_unsupported_format(self):
        """Test can_process method for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            txt_path = Path(tmp_file.name)
        
        try:
            assert self.factory.can_process(txt_path) is False
        finally:
            txt_path.unlink()
    
    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        formats = self.factory.get_supported_formats()
        assert isinstance(formats, list)
        assert "pdf" in formats
        assert "docx" in formats
        assert "xlsx" in formats
        assert len(formats) == 3
    
    @patch('document_processing.pdf_processor.PDFProcessor.parse_document')
    def test_process_document_success(self, mock_parse):
        """Test successful document processing through factory."""
        # Mock successful processing
        mock_metadata = Mock()
        mock_metadata.filename = "test.pdf"
        mock_metadata.processing_status = "success"
        mock_parse.return_value = mock_metadata
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            result = self.factory.process_document(tmp_path)
            assert result.filename == "test.pdf"
            assert result.processing_status == "success"
            mock_parse.assert_called_once_with(tmp_path)
        finally:
            tmp_path.unlink()
    
    @patch('document_processing.pdf_processor.PDFProcessor.parse_document')
    def test_process_multiple_documents(self, mock_parse):
        """Test processing multiple documents."""
        # Mock successful processing
        mock_metadata1 = Mock()
        mock_metadata1.filename = "doc1.pdf"
        mock_metadata1.processing_status = "success"
        mock_metadata1.content_sections = []
        
        mock_metadata2 = Mock()
        mock_metadata2.filename = "doc2.pdf"  
        mock_metadata2.processing_status = "success"
        mock_metadata2.content_sections = []
        
        mock_parse.side_effect = [mock_metadata1, mock_metadata2]
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
            path1 = Path(tmp1.name)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
            path2 = Path(tmp2.name)
        
        try:
            results = self.factory.process_multiple_documents([path1, path2])
            
            assert len(results) == 2
            assert path1.name in results
            assert path2.name in results
            assert results[path1.name].processing_status == "success"
            assert results[path2.name].processing_status == "success"
            
        finally:
            path1.unlink()
            path2.unlink()
    
    @patch('document_processing.pdf_processor.PDFProcessor.parse_document')
    def test_process_multiple_documents_with_errors(self, mock_parse):
        """Test processing multiple documents with some errors."""
        # First document succeeds, second fails
        mock_metadata1 = Mock()
        mock_metadata1.filename = "good.pdf"
        mock_metadata1.processing_status = "success"
        mock_metadata1.content_sections = []
        
        mock_parse.side_effect = [
            mock_metadata1,
            DocumentProcessingError("Processing failed", None)
        ]
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
            path1 = Path(tmp1.name)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
            path2 = Path(tmp2.name)
        
        try:
            results = self.factory.process_multiple_documents([path1, path2])
            
            assert len(results) == 2
            
            # First document should succeed
            assert results[path1.name].processing_status == "success"
            
            # Second document should have error metadata
            assert results[path2.name].processing_status == "error"
            assert "Processing failed" in results[path2.name].error_message
            
        finally:
            path1.unlink()
            path2.unlink()
