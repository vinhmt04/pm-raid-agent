"""
Document processor factory for handling different file formats.
Provides unified interface for document processing across all supported formats.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import DocumentParser, DocumentMetadata, SupportedFormat, DocumentProcessingError, FileFormatValidator
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .xlsx_processor import XLSXProcessor
from logging_config import get_logger

logger = get_logger(__name__)


class DocumentProcessorFactory:
    """
    Factory class for creating and managing document processors.
    Automatically selects the appropriate processor based on file format.
    """
    
    def __init__(self):
        """Initialize factory with all available processors."""
        self._processors: Dict[SupportedFormat, DocumentParser] = {
            SupportedFormat.PDF: PDFProcessor(),
            SupportedFormat.DOCX: DOCXProcessor(),
            SupportedFormat.XLSX: XLSXProcessor()
        }
        
        logger.debug("Document processor factory initialized", 
                    processors=list(self._processors.keys()))
    
    def get_processor(self, file_path: Path) -> DocumentParser:
        """
        Get the appropriate processor for a file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            DocumentParser instance for the file format
            
        Raises:
            DocumentProcessingError: If format is not supported
        """
        try:
            file_format = FileFormatValidator.validate_file_format(file_path)
            processor = self._processors.get(file_format)
            
            if processor is None:
                raise DocumentProcessingError(
                    f"No processor available for format: {file_format}",
                    file_path
                )
            
            logger.debug("Processor selected", 
                        file=str(file_path),
                        format=file_format,
                        processor=type(processor).__name__)
            
            return processor
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Failed to select processor: {str(e)}", file_path, e)
    
    def process_document(self, file_path: Path) -> DocumentMetadata:
        """
        Process a document using the appropriate processor.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            DocumentMetadata with extracted content
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        processor = self.get_processor(file_path)
        return processor.parse_document(file_path)
    
    def can_process(self, file_path: Path) -> bool:
        """
        Check if a file can be processed by any available processor.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file can be processed
        """
        return FileFormatValidator.is_supported_format(file_path)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return FileFormatValidator.get_supported_formats()
    
    def process_multiple_documents(self, file_paths: List[Path]) -> Dict[str, DocumentMetadata]:
        """
        Process multiple documents and return results.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary mapping file names to DocumentMetadata
        """
        results = {}
        
        for file_path in file_paths:
            try:
                logger.info("Processing document", file=str(file_path))
                metadata = self.process_document(file_path)
                results[file_path.name] = metadata
                
                logger.info("Document processed successfully", 
                          file=str(file_path),
                          sections=len(metadata.content_sections),
                          status=metadata.processing_status)
                
            except DocumentProcessingError as e:
                logger.error("Document processing failed", 
                           file=str(file_path),
                           error=str(e))
                
                # Create error metadata
                error_metadata = DocumentMetadata(
                    filename=file_path.name,
                    file_type=SupportedFormat.PDF,  # Default, will be overridden if format detected
                    content_sections=[],
                    processing_status="error",
                    error_message=str(e)
                )
                results[file_path.name] = error_metadata
        
        return results


# Global factory instance
document_factory = DocumentProcessorFactory()
