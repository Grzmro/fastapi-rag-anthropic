"""Document ingestion endpoints."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app import vectorstore
from app.ingestion import SUPPORTED_EXTENSIONS, ingest_file
from app.schemas import (
    DocumentsResponse,
    DriveIngestRequest,
    DriveIngestResponse,
    IngestResponse,
)

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    """Upload a document (PDF/TXT/MD), chunk it and index it in Chroma."""
    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        tmp_path.write_bytes(content)
        try:
            chunks = ingest_file(tmp_path, source_name=filename)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to parse document: {exc}") from exc

    added = vectorstore.add_documents(chunks)
    return IngestResponse(filename=filename, chunks_added=added)


@router.post("/ingest/drive", response_model=DriveIngestResponse)
def ingest_drive(request: DriveIngestRequest) -> DriveIngestResponse:
    """Index all supported files from a shared Google Drive folder.

    Requires GOOGLE_SERVICE_ACCOUNT_FILE and the folder shared with the
    service account's e-mail (see README).
    """
    from app.sources.gdrive import ingest_drive_folder

    try:
        ingested, skipped = ingest_drive_folder(request.folder_id)
    except RuntimeError as exc:  # missing/invalid configuration
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # Google API errors (bad folder id, no access, ...)
        raise HTTPException(status_code=502, detail=f"Google Drive error: {exc}") from exc

    return DriveIngestResponse(
        ingested=[IngestResponse(filename=name, chunks_added=n) for name, n in ingested],
        skipped=skipped,
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    """List indexed source documents."""
    return DocumentsResponse(
        sources=vectorstore.list_sources(),
        total_chunks=vectorstore.count_chunks(),
    )
