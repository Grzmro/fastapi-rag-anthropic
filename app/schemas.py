"""API request/response models."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int


class DocumentsResponse(BaseModel):
    sources: list[str]
    total_chunks: int


class DriveIngestRequest(BaseModel):
    folder_id: str = Field(
        min_length=1,
        description="Google Drive folder ID (the part after /folders/ in the folder URL)",
    )


class DriveIngestResponse(BaseModel):
    ingested: list[IngestResponse]
    skipped: list[str] = Field(description="Files with unsupported types, left out")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Question to answer from the indexed documents")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override the number of retrieved chunks")
    include_chunks: bool = Field(default=False, description="Return the retrieved chunks for debugging")


class Citation(BaseModel):
    marker: int = Field(description="Context fragment number referenced in the answer, e.g. [1]")
    source: str
    page: int
    quote: str = Field(description="Short verbatim fragment supporting the claim")


class RetrievedChunk(BaseModel):
    marker: int
    source: str
    page: int
    content: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk] | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    indexed_chunks: int
