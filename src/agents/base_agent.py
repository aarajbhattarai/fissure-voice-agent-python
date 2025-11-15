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

from .conversation_summary import ConversationSummarizer, SummaryStorage
from .user_data import (
    ConversationHistory,
    SessionData,
    UserData,
    create_user_data_from_dict,
)

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

        # User data management
        self.user_data = create_user_data_from_dict(
            {**user_details, "user_id": user_id} if user_details else {"user_id": user_id}
        )

        # Session data
        self.session_data = SessionData(
            session_id=session_id,
            user_data=self.user_data,
            tenant_id=config.get("tenant_config", {}).get("tenant_id", "default"),
            agent_id=config.get("agent_id", "unknown"),
        )

        # Conversation tracking
        self.conversation_history = ConversationHistory(
            session_id=session_id, user_id=user_id
        )

        # Conversation summary
        self.summarizer: Optional[ConversationSummarizer] = None
        if config.get("summary_config", {}).get("enabled", True):
            self.summarizer = ConversationSummarizer(
                llm_provider=config.get("llm_config", {}).get("provider", "openai"),
                model=config.get("summary_config", {}).get("model")
                or config.get("llm_config", {}).get("model", "gpt-5-nano"),
            )

        # Optional tracing
        self.tracer = tracer
        self.tracing_enabled = config.get("tracing_config", {}).get("enabled", False)

        # Data streamer for RPC
        self.data_streamer = StructuredDataStreamer(room)

        # Internal state
        self._is_llm_response = False
        self._background_tasks: set[asyncio.Task] = set()
        self._last_user_message = ""
        self._last_agent_message = ""

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
            user_name = self.user_data.name if hasattr(self.user_data, "name") else None
            welcome_message = (
                f"Welcome to your session{', ' + user_name if user_name else ''}. "
                f"I'm here to assist you. Let's begin!"
            )

            if span:
                span.set_attribute("input.value", welcome_message)

            try:
                await self.session.say(text=welcome_message)

                # Track welcome message in conversation history
                self.conversation_history.add_turn(
                    speaker="agent", message=welcome_message
                )

                logger.info(f"Session started for user {self.user_id}")
                logger.info(
                    f"User data: name={self.user_data.name if hasattr(self.user_data, 'name') else 'N/A'}, "
                    f"institution={self.user_data.institution if hasattr(self.user_data, 'institution') else 'N/A'}"
                )
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

                        # Track user message in conversation history
                        self._last_user_message = transcribed_text
                        self.conversation_history.add_turn(
                            speaker="user", message=transcribed_text
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

                # Track agent message in conversation history
                self._last_agent_message = acc_text
                self.conversation_history.add_turn(
                    speaker="agent", message=acc_text
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

    async def generate_summary(self) -> Optional[dict]:
        """
        Generate conversation summary.

        Returns:
            Summary data dictionary or None if summarizer not configured
        """
        if not self.summarizer:
            logger.info("Summary generation disabled")
            return None

        try:
            logger.info("Generating conversation summary...")

            # Get full transcript
            transcript = self.conversation_history.get_full_transcript()

            # Generate summary
            summary_data = await self.summarizer.generate_summary(
                transcript=transcript,
                user_data=self.user_data.to_dict(),
                session_metadata={
                    "session_id": self.session_id,
                    "agent_id": self.config.get("agent_id"),
                    "total_turns": self.conversation_history.total_turns,
                    "duration_seconds": self.session_data.duration_seconds,
                },
            )

            # Set summary in conversation history
            self.conversation_history.set_summary(summary_data["summary"])

            logger.info("Summary generated successfully")

            # Send summary to frontend
            await self.data_streamer.send_structured_data(
                topic="conversation-summary",
                data={
                    "session_id": self.session_id,
                    "summary": summary_data["summary"],
                    "structured_summary": summary_data.get("structured_summary", {}),
                    "total_turns": self.conversation_history.total_turns,
                },
            )

            return summary_data

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}", exc_info=True)
            return None

    async def close(self) -> None:
        """Clean up resources on session end."""
        logger.info(f"Closing agent session: {self.session_id}")

        # Mark session as ended
        self.session_data.end_session(status="completed")

        # Generate conversation summary
        summary_data = await self.generate_summary()

        # Wait for background tasks
        if getattr(self, "_background_tasks", None) and self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Send final session data to frontend
        await self.data_streamer.send_structured_data(
            topic="session-complete",
            data={
                "session_id": self.session_id,
                "user_id": self.user_id,
                "total_turns": self.conversation_history.total_turns,
                "duration_seconds": self.session_data.duration_seconds,
                "status": self.session_data.status,
                "summary_available": summary_data is not None,
            },
        )

        # Create final span
        with self.trace_span("session_end") as span:
            self.set_user_attributes(span)
            if span:
                span.set_attribute("session.ended", True)
                span.set_attribute("total_turns", self.conversation_history.total_turns)
                span.set_attribute("duration_seconds", self.session_data.duration_seconds or 0)

        logger.info(
            f"Agent session ended: {self.session_id} "
            f"({self.conversation_history.total_turns} turns, "
            f"{self.session_data.duration_seconds:.2f}s)"
        )
