"""Task 8.3: approval-gated Excel download preparation.

Streamlit integration (Task 9.4) can pass the returned fields to
st.download_button(label=..., data=result.data, file_name=result.filename,
                   mime=result.mime_type).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Union

from .excel_exporter import ExcelExporter

try:
    from src.raid_models.raid_finding import RAIDFinding, Status
    from src.workflow.raid_state_manager import RAIDStateManager
except ImportError:
    from raid_models.raid_finding import RAIDFinding, Status
    from workflow.raid_state_manager import RAIDStateManager


EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class NoApprovedFindingsError(ValueError):
    """Raised when export is requested without any explicitly approved findings."""


@dataclass(frozen=True)
class ExcelDownload:
    data: bytes
    filename: str
    mime_type: str
    message: str
    approved_count: int


class RAIDExportService:
    """Validate export eligibility and prepare Streamlit-compatible download data."""

    def __init__(self, exporter: ExcelExporter | None = None):
        self.exporter = exporter if exporter is not None else ExcelExporter()

    def prepare_download(
        self,
        findings: Union[RAIDStateManager, Iterable[RAIDFinding]],
        *,
        filename: str | None = None,
    ) -> ExcelDownload:
        # Snapshot the approved findings once; never trust a UI-provided count.
        if isinstance(findings, RAIDStateManager):
            approved = list(findings.get_approved_findings())
        else:
            approved = []
            for finding in findings:
                if not isinstance(finding, RAIDFinding):
                    raise TypeError("All export items must be RAIDFinding instances")
                if finding.status == Status.APPROVED:
                    approved.append(finding)

        if not approved:
            raise NoApprovedFindingsError(
                "Cannot export: no RAID findings have been explicitly approved."
            )

        if filename is None:
            filename = f"approved_raid_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        if not isinstance(filename, str) or not filename or filename != filename.split('/')[-1] or '\\' in filename or not filename.lower().endswith('.xlsx'):
            raise ValueError("filename must be a simple .xlsx filename")

        data = self.exporter.export(approved)
        return ExcelDownload(
            data=data,
            filename=filename,
            mime_type=EXCEL_MIME_TYPE,
            message=f"Excel export ready: {len(approved)} approved RAID finding(s).",
            approved_count=len(approved),
        )
