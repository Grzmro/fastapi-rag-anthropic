"""Google Drive ingestion source (service-account auth).

Setup:
1. In Google Cloud Console create a project, enable the Google Drive API and
   create a *service account*; download its key as JSON.
2. Point GOOGLE_SERVICE_ACCOUNT_FILE at that JSON file.
3. In Google Drive, share the folder you want to index with the service
   account's e-mail address (Viewer role is enough).

The service account only sees what has been explicitly shared with it.
"""

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from app.config import get_settings

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Native Google formats must be *exported*; regular files are downloaded as-is.
EXPORTABLE = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
}
DOWNLOADABLE = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


@dataclass
class DrivePlan:
    """How to fetch a single Drive file."""

    file_id: str
    name: str
    extension: str
    export_mime: str | None  # None -> plain download


def plan_download(file_id: str, name: str, mime_type: str) -> DrivePlan | None:
    """Decide whether/how a Drive file can be ingested. None -> skip."""
    if mime_type in EXPORTABLE:
        export_mime, extension = EXPORTABLE[mime_type]
        return DrivePlan(file_id=file_id, name=name, extension=extension, export_mime=export_mime)
    if mime_type in DOWNLOADABLE:
        return DrivePlan(
            file_id=file_id, name=name, extension=DOWNLOADABLE[mime_type], export_mime=None
        )
    # Fallback for files served with a generic mime type but a known extension.
    suffix = Path(name).suffix.lower()
    if suffix in {".pdf", ".txt", ".md"}:
        return DrivePlan(file_id=file_id, name=name, extension=suffix, export_mime=None)
    return None


def get_drive_service():
    settings = get_settings()
    if not settings.google_service_account_file:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_FILE is not configured")
    if not Path(settings.google_service_account_file).exists():
        raise RuntimeError(
            f"Service account key not found: {settings.google_service_account_file}"
        )
    credentials = service_account.Credentials.from_service_account_file(
        settings.google_service_account_file, scopes=SCOPES
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_folder(service, folder_id: str) -> list[dict]:
    """List non-trashed files directly inside a Drive folder."""
    files: list[dict] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return files


def download_to(service, plan: DrivePlan, target_dir: Path) -> Path:
    """Download or export one file into `target_dir`, returning the local path."""
    if plan.export_mime:
        request = service.files().export_media(fileId=plan.file_id, mimeType=plan.export_mime)
    else:
        request = service.files().get_media(fileId=plan.file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    local_name = plan.name if plan.name.lower().endswith(plan.extension) else plan.name + plan.extension
    # Drive names may contain path separators — keep only the base name.
    local_path = target_dir / Path(local_name).name
    local_path.write_bytes(buffer.getvalue())
    return local_path


def ingest_drive_folder(folder_id: str) -> tuple[list[tuple[str, int]], list[str]]:
    """Index all supported files from a Drive folder.

    Returns (ingested, skipped): ingested as (filename, chunks_added) pairs,
    skipped as filenames with unsupported types.
    """
    from app import vectorstore
    from app.ingestion import ingest_file

    service = get_drive_service()
    entries = list_folder(service, folder_id)

    ingested: list[tuple[str, int]] = []
    skipped: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for entry in entries:
            plan = plan_download(entry["id"], entry["name"], entry.get("mimeType", ""))
            if plan is None:
                skipped.append(entry["name"])
                continue
            local_path = download_to(service, plan, tmp_path)
            chunks = ingest_file(local_path, source_name=local_path.name)
            added = vectorstore.add_documents(chunks)
            ingested.append((local_path.name, added))

    return ingested, skipped
