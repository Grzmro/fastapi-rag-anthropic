"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app import vectorstore
from app.config import get_settings
from app.routers import ask, ingest
from app.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the local embedding model and the Chroma client at startup so
    # the first request doesn't pay the model-loading cost.
    vectorstore.get_vectorstore()
    yield


app = FastAPI(
    title="RAG Document Assistant",
    description="Grounded LLM Q&A with citations — LangChain + Chroma + Anthropic API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(ask.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model=settings.anthropic_model,
        embedding_model=settings.embedding_model,
        indexed_chunks=vectorstore.count_chunks(),
    )
