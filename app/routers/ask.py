"""Question answering endpoint."""

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.rag_chain import answer_question
from app.schemas import AskRequest, AskResponse, Citation, RetrievedChunk

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question grounded in the indexed documents, with citations."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")

    try:
        result, docs = answer_question(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    # Map citation markers back to chunk metadata — the model never produces
    # filenames or page numbers itself.
    citations = [
        Citation(
            marker=c.marker,
            source=docs[c.marker - 1].metadata.get("source", "unknown"),
            page=docs[c.marker - 1].metadata.get("page", 1),
            quote=c.quote,
        )
        for c in result.citations
    ]

    retrieved = None
    if request.include_chunks:
        retrieved = [
            RetrievedChunk(
                marker=i,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 1),
                content=doc.page_content,
            )
            for i, doc in enumerate(docs, start=1)
        ]

    return AskResponse(answer=result.answer, citations=citations, retrieved_chunks=retrieved)
