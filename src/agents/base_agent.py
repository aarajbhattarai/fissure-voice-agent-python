"""
Base agent class with full dynamic configurability.
"""

import asyncio
import logging
from collections.abc import AsyncIterable
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Callable, Optional, Type, cast

from livekit import rtc
from livekit.agents import (
    Agent,
    ChatContext,
    FunctionTool,
    ModelSettings,
    stt,
)
from livekit.plugins import openai
from opentelemetry import trace
from typing_extensions import TypedDict

from interview_agent.utilities.rpc import StructuredDataStreamer
from interview_agent.utilities.structured_output import process_structured_output

logger = logging.getLogger("base-agent")


class BaseAgent(Agent):
    """
    Abstract base agent with full configurability.
    All agents inherit from this and get dynamic behavior.
    """

    def __init__(
        self,
        room: rtc.Room,
        user_id: str,
        session_id: str,
        llm: Any,
        tts: Any,
        stt: Any,
        turn_detection: Any,
        prompt: str,
        schema_class: Type[TypedDict],
        tracer: Optional[trace.Tracer],
        config: dict[str, Any],
        user_details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize base agent.

        Args:
            room: LiveKit room instance
            user_id: User identifier
            session_id: Session identifier
            llm: LLM instance
            tts: TTS instance
            stt: STT instance
            turn_detection: Turn detection instance
            prompt: System prompt
            schema_class: Dynamic TypedDict schema class
            tracer: Optional OpenTelemetry tracer
            config: Full agent configuration
            user_details: User-specific metadata
        """
        super().__init__(
            stt=stt,
            llm=llm,
            tts=tts,
            turn_detection=turn_detection,
        )

        # Core identifiers
        self.user_id = user_id
        self.session_id = session_id
        self.room = room

        # Dynamic configuration
        self.prompt = prompt
        self.schema_class = schema_class
        self.config = config
        self.user_details = user_details or {}

        # Optional tracing
        self.tracer = tracer
        self.tracing_enabled = config.get("tracing_config", {}).get("enabled", False)

        # Data streamer for RPC
        self.data_streamer = StructuredDataStreamer(room)

        # Internal state
        self._is_llm_response = False
        self._background_tasks: set[asyncio.Task] = set()

    def trace_span(self, name: str):
        """
        Context manager for conditional tracing.

        Args:
            name: Span name

        Returns:
            Trace span or no-op context manager
        """
        if self.tracing_enabled and self.tracer:
            return self.tracer.start_as_current_span(name)
        else:
            # No-op context manager
            return nullcontext()

    def set_user_attributes(self, span) -> None:
        """
        Helper method to set user attributes on spans.

        Args:
            span: OpenTelemetry span
        """
        if span is not None:
            span.set_attribute("langfuse.user.id", self.user_id)
            span.set_attribute("langfuse.session.id", self.session_id)

            # Set user details as attributes
            if self.user_details:
                for key, value in self.user_details.items():
                    span.set_attribute(f"user.{key}", str(value))

    async def on_enter(self) -> None:
        """Called when agent session starts."""
        with self.trace_span("agent-session-start") as span:
            self.set_user_attributes(span)

            # Create personalized welcome message
            user_name = self.user_details.get("name", "")
            welcome_message = (
                f"Welcome to your session{', ' + user_name if user_name else ''}. "
                f"I'm here to assist you. Let's begin!"
            )

            if span:
                span.set_attribute("input.value", welcome_message)

            try:
                await self.session.say(text=welcome_message)
                logger.info(f"Session started for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to send welcome message: {e}", exc_info=True)
                if span:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ):
        """LLM node with dynamic schema support."""
        self._is_llm_response = True

        with self.trace_span("llm_node") as span:
            self.set_user_attributes(span)

            if span:
                span.set_attribute(
                    "gen_ai.request.model", self.config["llm_config"].get("model", "")
                )

            try:
                # Check if LLM supports structured output
                supports_structured = self.config["llm_config"].get(
                    "supports_structured_output", False
                )

                if supports_structured and hasattr(self.llm, "chat"):
                    llm = cast(openai.LLM, self.llm)

                    async with llm.chat(
                        chat_ctx=chat_ctx,
                        tools=tools,
                        response_format=self.schema_class,
                    ) as stream:
                        async for chunk in stream:
                            yield chunk
                else:
                    # Fallback to default
                    async for chunk in Agent.default.llm_node(
                        self, chat_ctx, tools, model_settings
                    ):
                        yield chunk
            except Exception as e:
                logger.error(f"LLM node error: {e}", exc_info=True)
                if span:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                self._is_llm_response = False

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """TTS node with dynamic instruction field."""
        with self.trace_span("tts_node") as span:
            self.set_user_attributes(span)

            if span:
                span.set_attribute(
                    "gen_ai.request.model", self.config["tts_config"].get("provider", "")
                )

            instruction_field = self.config["tts_config"].get("instruction_field")
            instruction_updated = False

            def on_output_processed(resp: dict):
                nonlocal instruction_updated
                if (
                    instruction_field
                    and instruction_field in resp
                    and not instruction_updated
                ):
                    instruction_updated = True
                    instructions = resp[instruction_field]

                    # Apply instructions to TTS
                    if hasattr(self.tts, "update_options"):
                        try:
                            self.tts.update_options(instructions=instructions)
                            logger.info(f"Applied TTS instructions: {instructions}")
                        except Exception as e:
                            logger.warning(f"Failed to apply TTS instructions: {e}")

            # Process with dynamic schema
            processed_text = process_structured_output(
                text, callback=on_output_processed, force_structured=self._is_llm_response
            )

            async for frame in Agent.default.tts_node(
                self, processed_text, model_settings
            ):
                yield frame

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """Speech-to-text processing with tracing."""
        with self.trace_span("stt_node") as span:
            self.set_user_attributes(span)

            if span:
                span.set_attribute(
                    "gen_ai.request.model", self.config.get("stt_config", {}).get("provider", "")
                )

            try:
                transcribed_text = ""
                async for event in Agent.default.stt_node(self, audio, model_settings):
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        transcribed_text = event.alternatives[0].text
                        logger.info(
                            f"Speech recognized for {self.user_id}: {transcribed_text[:50]}..."
                        )

                        # Set output attributes
                        if span:
                            span.set_attribute("output.value", transcribed_text)
                            if hasattr(event.alternatives[0], "confidence"):
                                span.set_attribute(
                                    "stt.confidence", event.alternatives[0].confidence
                                )
                    yield event

            except Exception as e:
                if span:
                    span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR, f"STT error for user {self.user_id}: {e}"
                        )
                    )
                logger.error(f"STT error: {e}")
                raise

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ):
        """Transcription node with smart processing and structured data streaming."""
        chunk_count = 0
        start_time = datetime.utcnow()
        chunk_buffer = []
        last_send_time = start_time
        send_interval = 1  # Send every 1 second
        acc_text = ""

        async def text_processor():
            """Process text and send structured data."""
            nonlocal chunk_count, acc_text, chunk_buffer, last_send_time

            try:
                async for delta in text:
                    chunk_count += 1
                    acc_text += delta

                    # Create payload
                    chunk_data = {
                        "type": "llm_chunk",
                        "source": "llm_transcription",
                        "content": delta,
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "is_structured": self._is_llm_response,
                        "chunk_count": chunk_count,
                        "total_length": len(acc_text),
                    }
                    chunk_buffer.append(chunk_data)

                    # Send buffered chunks periodically
                    current_time = datetime.utcnow()
                    time_elapsed = (current_time - last_send_time).total_seconds()

                    if time_elapsed >= send_interval and chunk_buffer:
                        asyncio.create_task(
                            self.data_streamer.stream_structured_data(
                                topic="llm-transcription", data_chunks=chunk_buffer.copy()
                            )
                        )
                        chunk_buffer.clear()
                        last_send_time = current_time

                    yield delta

                # Send remaining buffered chunks
                if chunk_buffer:
                    asyncio.create_task(
                        self.data_streamer.stream_structured_data(
                            topic="llm-transcription", data_chunks=chunk_buffer.copy()
                        )
                    )

                # Send final complete payload
                duration = (datetime.utcnow() - start_time).total_seconds()
                final_payload = {
                    "type": "llm_response_complete",
                    "source": "llm_transcription",
                    "content": acc_text,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_structured": self._is_llm_response,
                    "chunk_count": chunk_count,
                    "total_length": len(acc_text),
                    "response_time_seconds": duration,
                    "completion_signal": "transcription_complete",
                }

                asyncio.create_task(
                    self.data_streamer.send_structured_data(
                        topic="llm-response-complete", data=final_payload
                    )
                )

                logger.info(
                    f"Completed streaming {chunk_count} chunks ({len(acc_text)} chars) in {duration:.2f}s"
                )

            except Exception as e:
                logger.error(f"Error in structured data streaming: {e}", exc_info=True)
                raise

        # Process the text with custom processor
        processed_text = process_structured_output(
            text_processor(), force_structured=self._is_llm_response
        )

        # Run default transcription processing
        result = Agent.default.transcription_node(self, processed_text, model_settings)

        async for chunk in result:
            yield chunk

    async def close(self) -> None:
        """Clean up resources on session end."""
        if getattr(self, "_background_tasks", None) and self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Create final span
        with self.trace_span("session_end") as span:
            self.set_user_attributes(span)
            if span:
                span.set_attribute("session.ended", True)

        logger.info(f"Agent session ended: {self.session_id}")
