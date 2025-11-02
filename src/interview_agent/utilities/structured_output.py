from collections.abc import AsyncIterable
from typing import Annotated, Callable, Literal, Optional

from pydantic import Field
from pydantic_core import from_json
from typing_extensions import TypedDict

# =======================
# TYPE DEFINITIONS
# =======================


class InterviewTurnJSON(TypedDict, total=False):
    voice_instructions: Annotated[
        str,
        Field(description="Specific TTS directive for tone, pace, and emphasis"),
    ]
    system_response: Annotated[
        str,
        Field(description="The officer's spoken response"),
    ]
    internal_assessment: Annotated[
        str,
        Field(description="Private evaluation of interaction"),
    ]
    analysis: Annotated[
        list[str],
        Field(description="Behavioral and content analysis points"),
    ]
    interview_stage: Annotated[
        Literal[
            "document_check",
            "background_inquiry",
            "academic_assessment",
            "financial_review",
            "intent_evaluation",
            "final_decision",
        ],
        Field(description="Current phase of the interview process"),
    ]
    credibility_score: Annotated[
        int,
        Field(description="Running assessment score (1–10)", ge=1, le=10),
    ]
    red_flags: Annotated[
        list[str],
        Field(description="Specific concerns identified"),
    ]
    next_focus_area: Annotated[
        str,
        Field(description="Recommended next area to probe"),
    ]


# =======================
# SEPARATE HELPERS
# =======================


async def detect_output_mode(
    text_stream: AsyncIterable[str],
    json_indicators: tuple[str, ...] = ('"system_response"', '"voice_instructions"'),
) -> tuple[bool, list[str]]:
    """
    Peek into the first few chunks to determine whether
    the stream contains structured JSON or plain text.
    Returns (is_structured, buffered_chunks).
    """
    chunks = []
    async for chunk in text_stream:
        chunks.append(chunk)
        chunk_stripped = chunk.strip()
        if chunk_stripped.startswith(("{", '{"')) or any(
            ind in chunk for ind in json_indicators
        ):
            return True, chunks
        # If first meaningful chunk looks like plain text, assume plain mode
        if len(chunk_stripped) > 0:
            return False, chunks
    # Empty stream (edge case)
    return False, chunks


async def process_plain_text(
    text_stream: AsyncIterable[str],
    buffered_chunks: list[str],
) -> AsyncIterable[str]:
    """Yield plain text directly."""
    for chunk in buffered_chunks:
        yield chunk
    async for chunk in text_stream:
        yield chunk


async def process_structured_json(
    text_stream: AsyncIterable[str],
    buffered_chunks: list[str],
    callback: Optional[Callable[[InterviewTurnJSON], None]] = None,
) -> AsyncIterable[str]:
    """
    Incrementally parse structured JSON and yield new system_response deltas.
    No fallback to plain text — assumes structured JSON stream.

    Args:
        text_stream: Async generator yielding chunks of JSON text.
        callback: Optional callback invoked with the parsed JSON after each successful parse.

    Yields:
        system_response deltas as they become available.

    """
    cb = callback or (lambda _: None)
    acc_text = "".join(buffered_chunks)
    system_response = ""

    async for chunk in text_stream:
        acc_text += chunk
        try:
            resp: InterviewTurnJSON = from_json(
                acc_text, allow_partial="trailing-strings"
            )
        except ValueError:
            # Simply wait for more chunks — do not fallback
            continue

        current_response = resp.get("system_response", "")
        if current_response and len(current_response) > len(system_response):
            new_delta = current_response[len(system_response) :]
            yield new_delta
            system_response = current_response

        cb(resp)


# =======================
# MAIN ORCHESTRATOR
# =======================


async def process_structured_output(
    text: AsyncIterable[str],
    callback: Optional[Callable[[InterviewTurnJSON], None]] = None,
    force_structured: bool = False,
) -> AsyncIterable[str]:
    """
    Detects whether the input stream is structured or plain text,
    and routes it to the correct processing function.
    """
    if force_structured:
        is_structured, buffered = True, []
    else:
        is_structured, buffered = await detect_output_mode(text)

    if is_structured:
        async for chunk in process_structured_json(text, buffered, callback):
            yield chunk
    else:
        async for chunk in process_plain_text(text, buffered):
            yield chunk
