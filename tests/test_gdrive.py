"""Unit tests for the Drive file-selection logic (no network access)."""

from app.sources.gdrive import plan_download


def test_google_doc_is_exported_as_text():
    plan = plan_download("id1", "Meeting notes", "application/vnd.google-apps.document")
    assert plan is not None
    assert plan.export_mime == "text/plain"
    assert plan.extension == ".txt"


def test_pdf_is_downloaded_directly():
    plan = plan_download("id2", "report.pdf", "application/pdf")
    assert plan is not None
    assert plan.export_mime is None
    assert plan.extension == ".pdf"


def test_unknown_mime_with_known_extension_falls_back():
    plan = plan_download("id3", "notes.md", "application/octet-stream")
    assert plan is not None
    assert plan.extension == ".md"


def test_unsupported_file_is_skipped():
    assert plan_download("id4", "photo.jpg", "image/jpeg") is None
    assert plan_download("id5", "sheet", "application/vnd.google-apps.spreadsheet") is None
