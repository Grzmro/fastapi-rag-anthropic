"""LLM-as-judge: is the generated answer faithful to the retrieved context?

The judge sees ONLY the context fragments and the answer — it must not use
outside knowledge. It splits the answer into claims and marks each as
supported or not; faithfulness = supported / total.

Note: the default judge model equals the generator model (claude-haiku-4-5),
which is cheap but effectively self-evaluation. For a final validation pass
a stronger judge, e.g. `--judge-model claude-sonnet-5`.
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from app.config import get_settings

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluation judge for a retrieval-augmented QA system. You \
are given numbered context fragments and an answer produced by another model. \
Rules:

1. Split the answer into its individual factual claims. Meta-statements like \
"the context does not mention X" are not factual claims — skip them.
2. For each claim, decide whether it is supported by the context fragments. \
Use ONLY the fragments — no outside knowledge. A claim that is true in the \
real world but not stated in the fragments is NOT supported.
3. Set is_refusal to true if the answer states that the requested information \
is not present in the provided context.

Context fragments:
{context}"""


class ClaimVerdict(BaseModel):
    """One factual claim extracted from the answer, with a verdict."""

    claim: str = Field(description="The factual claim, quoted or paraphrased from the answer")
    supported: bool = Field(description="Is the claim supported by the context fragments?")


class JudgeVerdict(BaseModel):
    """Structured judgement of one answer."""

    claims: list[ClaimVerdict] = Field(default_factory=list)
    is_refusal: bool = Field(
        description="Does the answer state that the information is not in the context?"
    )


@lru_cache
def _judge_llm(model: str):
    settings = get_settings()
    llm = ChatAnthropic(
        model=model,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key or None,
    )
    return llm.with_structured_output(JudgeVerdict)


def judge_answer(
    question: str, context: str, answer: str, model: str = DEFAULT_JUDGE_MODEL
) -> JudgeVerdict:
    """Judge one answer against the context it was generated from."""
    messages = [
        ("system", JUDGE_SYSTEM_PROMPT.format(context=context)),
        ("human", f"Question: {question}\n\nAnswer to evaluate:\n{answer}"),
    ]
    return _judge_llm(model).invoke(messages)


def faithfulness(verdict: JudgeVerdict) -> float | None:
    """Fraction of claims supported by the context; None if there are no claims."""
    if not verdict.claims:
        return None
    return sum(c.supported for c in verdict.claims) / len(verdict.claims)
