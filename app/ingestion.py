"""Document loading and chunking.

Each chunk carries `source`, `page` and `chunk_id` metadata — these are the
basis for citations returned by the RAG chain.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import get_settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(path: str | Path, source_name: str | None = None) -> list[Document]:
    """Load a single file into LangChain Documents.

    `source_name` overrides the `source` metadata (useful when the file was
    saved under a temporary path but should be cited by its original name).
    """
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    source = source_name or path.name
    if extension == ".pdf":
        reader = PdfReader(str(path))
        docs = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": source, "page": page_number},
            )
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    else:
        docs = [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": source, "page": 1},
            )
        ]
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks and assign stable chunk ids."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}:{i}"
    return chunks


def ingest_file(path: str | Path, source_name: str | None = None) -> list[Document]:
    """Load + chunk a file, ready to be added to the vector store."""
    return chunk_documents(load_document(path, source_name=source_name))
