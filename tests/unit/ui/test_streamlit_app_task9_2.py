"""Task 9.2 — document upload parsing and Streamlit integration."""
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.ui.app import process_uploaded_documents, validate_uploads
from src.document_processing.base import DocumentMetadata, ContentSection, SupportedFormat


class Uploaded:
    def __init__(self, name, content=b"file bytes"):
        self.name = name
        self.content = content
    def getvalue(self):
        return self.content


def metadata(status="success", error=None):
    return DocumentMetadata(filename="temp.pdf", file_type=SupportedFormat.PDF,
        content_sections=[ContentSection(location="Page 1", text="Risk of delay", metadata={})],
        processing_status=status, error_message=error)


def test_successful_processing_and_original_filename():
    factory = Mock()
    factory.process_document.return_value = metadata()
    docs, results = process_uploaded_documents([Uploaded("Project Plan.PDF")], factory)
    assert len(docs) == 1
    assert docs[0].filename == "Project Plan.PDF"
    assert results[0].status == "success"
    assert results[0].sections == 1
    assert results[0].characters == len("Risk of delay")
    assert factory.process_document.call_args.args[0].suffix == ".pdf"


def test_failure_does_not_block_next_document():
    factory = Mock()
    factory.process_document.side_effect = [ValueError("corrupt"), metadata()]
    docs, results = process_uploaded_documents([Uploaded("bad.pdf"), Uploaded("good.pdf")], factory)
    assert len(docs) == 1
    assert [r.status for r in results] == ["error", "success"]
    assert "corrupt" in results[0].error


def test_partial_success_keeps_extracted_document():
    factory = Mock()
    factory.process_document.return_value = metadata("partial_success", "page 2 failed")
    docs, results = process_uploaded_documents([Uploaded("report.pdf")], factory)
    assert len(docs) == 1
    assert results[0].error == "page 2 failed"


def test_invalid_upload_rejected_before_parsing():
    factory = Mock()
    with pytest.raises(ValueError, match="Unsupported format"):
        process_uploaded_documents([Uploaded("script.exe")], factory)
    factory.process_document.assert_not_called()


def test_empty_file_reports_error():
    factory = Mock()
    docs, results = process_uploaded_documents([Uploaded("empty.docx", b"")], factory)
    assert docs == []
    assert results[0].status == "error"
    factory.process_document.assert_not_called()


def test_upload_paths_are_generated_not_user_controlled():
    factory = Mock()
    factory.process_document.return_value = metadata()
    docs, results = process_uploaded_documents([Uploaded("../../private.pdf")], factory)
    assert len(docs) == 1
    path = factory.process_document.call_args.args[0]
    assert path.name == "document_0.pdf"
