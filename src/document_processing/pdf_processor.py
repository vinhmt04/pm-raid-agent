"""
PDF document processor using PyPDF2.
Extracts text content with page number tracking and error handling.
"""

from pathlib import Path
from typing import List, Optional
import PyPDF2
from PyPDF2.errors import PdfReadError

from .base import DocumentParser, DocumentMetadata, ContentSection, SupportedFormat, DocumentProcessingError
from logging_config import get_logger

logger = get_logger(__name__)


class PDFProcessor(DocumentParser):
    """
    PDF document processor using PyPDF2.
    Handles text extraction with page tracking and graceful error recovery.
    """
    
    @property
    def supported_format(self) -> SupportedFormat:
        """Return PDF format support."""
        return SupportedFormat.PDF
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle PDF files.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file has .pdf extension
        """
        return file_path.suffix.lower() == ".pdf"
    
    def parse_document(self, file_path: Path) -> DocumentMetadata:
        """
        Parse PDF document and extract text content with page tracking.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentMetadata with extracted text and page information
            
        Raises:
            DocumentProcessingError: If parsing fails completely
        """
        logger.info("Starting PDF parsing", file=str(file_path))
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get basic document info
                num_pages = len(pdf_reader.pages)
                file_size = file_path.stat().st_size
                
                logger.debug("PDF metadata extracted", pages=num_pages, size_bytes=file_size)
                
                # Extract content from each page
                content_sections = []
                successful_pages = 0
                errors = []
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        # Create content section for this page
                        section = ContentSection(
                            location=f"page_{page_num + 1}",
                            text=text.strip(),
                            metadata={
                                "page_number": page_num + 1,
                                "total_pages": num_pages,
                                "character_count": len(text),
                                "extraction_method": "PyPDF2"
                            }
                        )
                        
                        content_sections.append(section)
                        successful_pages += 1
                        
                        logger.debug("Page extracted successfully", 
                                   page=page_num + 1, 
                                   chars=len(text))
                        
                    except Exception as e:
                        error_msg = f"Failed to extract page {page_num + 1}: {str(e)}"
                        errors.append(error_msg)
                        logger.warning("Page extraction failed", 
                                     page=page_num + 1, 
                                     error=str(e))
                        
                        # Create error section for partial results
                        error_section = ContentSection(
                            location=f"page_{page_num + 1}_error",
                            text="",
                            metadata={
                                "page_number": page_num + 1,
                                "total_pages": num_pages,
                                "extraction_error": error_msg,
                                "extraction_method": "PyPDF2"
                            }
                        )
                        content_sections.append(error_section)
                
                # Determine processing status
                if successful_pages == 0:
                    # Complete failure
                    error_message = f"Failed to extract any pages. Errors: {'; '.join(errors)}"
                    logger.error("PDF parsing completely failed", file=str(file_path), error=error_message)
                    raise DocumentProcessingError(error_message, file_path)
                
                elif successful_pages < num_pages:
                    # Partial success
                    status = "partial_success"
                    error_message = f"Extracted {successful_pages}/{num_pages} pages. Errors: {'; '.join(errors)}"
                    logger.warning("PDF parsing partially succeeded", 
                                 file=str(file_path),
                                 successful_pages=successful_pages,
                                 total_pages=num_pages)
                else:
                    # Complete success
                    status = "success"
                    error_message = None
                    logger.info("PDF parsing completed successfully", 
                              file=str(file_path),
                              pages=num_pages)
                
                return DocumentMetadata(
                    filename=file_path.name,
                    file_type=SupportedFormat.PDF,
                    content_sections=content_sections,
                    processing_status=status,
                    error_message=error_message,
                    total_pages=num_pages,
                    file_size_bytes=file_size
                )
                
        except PdfReadError as e:
            error_msg = f"PDF read error: {str(e)}"
            logger.error("PDF read error", file=str(file_path), error=error_msg)
            raise DocumentProcessingError(error_msg, file_path, e)
            
        except FileNotFoundError as e:
            error_msg = f"PDF file not found: {str(e)}"
            logger.error("PDF file not found", file=str(file_path), error=error_msg)
            raise DocumentProcessingError(error_msg, file_path, e)
            
        except Exception as e:
            error_msg = f"Unexpected error during PDF parsing: {str(e)}"
            logger.error("Unexpected PDF parsing error", file=str(file_path), error=error_msg)
            raise DocumentProcessingError(error_msg, file_path, e)
    
    def extract_metadata_only(self, file_path: Path) -> dict:
        """
        Extract only metadata without full text content (for quick validation).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with basic PDF metadata
            
        Raises:
            DocumentProcessingError: If metadata extraction fails
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    "num_pages": len(pdf_reader.pages),
                    "file_size_bytes": file_path.stat().st_size,
                    "format": "PDF"
                }
                
                # Try to extract document info if available
                if pdf_reader.metadata:
                    doc_info = pdf_reader.metadata
                    metadata.update({
                        "title": doc_info.get("/Title", ""),
                        "author": doc_info.get("/Author", ""),
                        "subject": doc_info.get("/Subject", ""),
                        "creator": doc_info.get("/Creator", "")
                    })
                
                return metadata
                
        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract PDF metadata: {str(e)}", file_path, e)
