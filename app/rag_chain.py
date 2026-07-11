"""RAG chain: retrieval -> grounded answer with citations via Anthropic API.

The model only ever sees numbered context fragments and returns citation
*markers* (plus a short quote). Mapping markers back to source/page metadata
is done in code, so the model cannot invent filenames or page numbers.
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app import vectorstore
from app.config import get_settings

SYSTEM_PROMPT = """\
You are a document assistant. Answer the user's question using ONLY the \
numbered context fragments provided below. Rules:

1. Base every claim strictly on the context fragments. Do not use outside knowledge.
2. Reference the fragments you used with their markers, e.g. [1], [2].
3. For each claim, include a citation with the fragment marker and a short \
verbatim quote from that fragment that supports the claim.
4. If the context does not contain the information needed to answer, say so \
explicitly in the answer and return an empty citations list. Do not guess.
5. Answer in the same language as the question.

Context fragments:
{context}"""


class CitationOutput(BaseModel):
    """A single citation produced by the model."""

    marker: int = Field(description="Number of the context fragment supporting the claim, e.g. 1 for [1]")
    quote: str = Field(description="Short verbatim quote from that fragment")


class GroundedAnswer(BaseModel):
    """Structured answer grounded in the retrieved context."""

    answer: str = Field(description="The answer, referencing fragment markers like [1]")
    citations: list[CitationOutput] = Field(default_factory=list)


@lru_cache
def get_llm() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_model,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key or None,
    )


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)
        parts.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, top_k: int | None = None) -> tuple[GroundedAnswer, list[Document]]:
    """Retrieve context and generate a grounded answer.

    Returns the structured answer and the retrieved documents (index 0 == marker [1]).
    """
    docs = vectorstore.search(question, k=top_k)
    if not docs:
        return (
            GroundedAnswer(
                answer="The document index is empty — ingest documents via POST /ingest first.",
                citations=[],
            ),
            [],
        )

    structured_llm = get_llm().with_structured_output(GroundedAnswer)
    messages = [
        ("system", SYSTEM_PROMPT.format(context=format_context(docs))),
        ("human", question),
    ]
    result = structured_llm.invoke(messages)

    # Drop citations pointing at markers outside the retrieved range.
    result.citations = [c for c in result.citations if 1 <= c.marker <= len(docs)]
    return result, docs
