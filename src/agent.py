import asyncio
import logging
from collections.abc import AsyncIterable
from datetime import datetime
from typing import Any, Optional, cast

# from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dotenv import load_dotenv
from langfuse import Langfuse
from livekit import agents, rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    ChatContext,
    # ChatMessage,
    FunctionTool,
    JobContext,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    UserStateChangedEvent,
    WorkerOptions,
    cli,
    stt,
)
from livekit.plugins import deepgram, noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from whispey import LivekitObserve

from interview_agent.utilities.prewarm import prewarm
from interview_agent.utilities.prompt_loader import load_prompt
from interview_agent.utilities.rpc import StructuredDataStreamer
from interview_agent.utilities.setup_langfuse import setup_langfuse
from interview_agent.utilities.structured_output import (
    InterviewTurnJSON,
    process_structured_output,
)
from interview_agent.utilities.user_details import get_default_user_details

# Configure logging
logger = logging.getLogger("interview-agent")

# Load environment variables
load_dotenv(".env")

# Initialize Langfuse
_langfuse = Langfuse()


class PTEInterviewAgent(Agent):
    """AI-powered US VISA Interview Officer for F-1 student visa assessments."""

    def __init__(
        self,
        room: rtc.Room,
        user_id: str,
        user_details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            instructions=load_prompt("pte_interview.yaml"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-5-nano"),
            tts=deepgram.TTS(),
            turn_detection=MultilingualModel(),
        )

        # User information
        self.user_id = user_id or f"user-{str(uuid4())[:8]}"
        self.user_details = user_details or {}

        # Livekit Room related data
        self.room = room
        self.session_id = str(uuid4())

        # Get OpenTelemetry tracer
        self.tracer = trace.get_tracer(__name__)

        # Add this flag
        self._is_llm_response = False
        # Track previously seen result IDs
        self._seen_results = set()
        self._background_tasks: set[asyncio.Task] = set()

        # Initialize structured data streamer
        self.data_streamer = StructuredDataStreamer(room)

    def set_user_attributes(self, span) -> None:
        """Helper method to set user attributes on spans."""
        span.set_attribute("langfuse.user.id", self.user_id)
        span.set_attribute("langfuse.session.id", self.session_id)

        # Set user details as attributes
        if self.user_details:
            for key, value in self.user_details.items():
                span.set_attribute(f"user.{key}", str(value))

    async def on_enter(self) -> None:
        """Initialize the interview session with welcome message."""
        with self.tracer.start_as_current_span("pte-interview-session-start") as span:
            self.set_user_attributes(span)

            # Create personalized welcome message
            user_name = self.user_details.get("name", "")
            welcome_message = (
                f"Welcome to your PTE Interview Practice Session"
                f"{', ' + user_name if user_name else ''}. "
                f"I'm here to help you practice and improve your interview skills. "
                f"Let's begin!"
            )

            span.set_attribute("input.value", welcome_message)

            try:
                await self.session.say(text=welcome_message)
                logger.info(f"Session started for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to send welcome message: {e}", exc_info=True)
                span.set_status(Status(StatusCode.ERROR, str(e)))

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """Speech-to-text processing with tracing."""
        with self.tracer.start_as_current_span("stt_node") as span:
            # Set user and model attributes
            self.set_user_attributes(span)
            span.set_attribute("gen_ai.request.model", "deepgram")

            try:
                transcribed_text = ""
                async for event in Agent.default.stt_node(self, audio, model_settings):
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        transcribed_text = event.alternatives[0].text
                        logger.info(
                            f"Speech recognized for {self.user_id}: {transcribed_text[:50]}..."
                        )
                        logger.info(f"Full transcript: {transcribed_text}")

                        # Set output attributes
                        span.set_attribute("output.value", transcribed_text)
                        if hasattr(event.alternatives[0], "confidence"):
                            span.set_attribute(
                                "stt.confidence", event.alternatives[0].confidence
                            )
                    yield event

            except Exception as e:
                span.set_status(
                    Status(StatusCode.ERROR, f"STT error for user {self.user_id}: {e}")
                )
                logger.error(f"STT error: {e}")
                raise

    async def send_interview_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Send interview metrics to frontend in real-time.

        Example usage:
            await self.send_interview_metrics({
                "fluency_score": 85,
                "pronunciation_score": 78,
                "content_score": 92,
                "timestamp": datetime.utcnow().isoformat()
            })
        """
        await self.data_streamer.send_structured_data(
            topic="interview-metrics", data=metrics
        )

    async def send_structured_feedback(self, feedback_chunks: list[dict]) -> None:
        """
        Stream structured feedback incrementally.

        Example usage:
            feedback_chunks = [
                {"type": "pronunciation", "score": 78, "feedback": "Good clarity"},
                {"type": "fluency", "score": 85, "feedback": "Natural pace"},
                {"type": "content", "score": 92, "feedback": "Comprehensive answer"}
            ]
            await self.send_structured_feedback(feedback_chunks)
        """
        await self.data_streamer.stream_structured_data(
            topic="interview-feedback", data_chunks=feedback_chunks
        )

    async def store_structured_response(self, response_data: InterviewTurnJSON):
        """Store structured response data and optionally send to frontend."""
        try:
            # Add metadata
            enhanced_data = {
                **response_data,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "question_index": self.question_index,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # ALSO send to frontend in real-time
            await self.data_streamer.send_structured_data(
                topic="llm-structured-response", data=enhanced_data
            )

        except Exception as e:
            logger.error(f"Error storing/sending structured response: {e}")

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ):
        """LLM processing with structured output support."""
        self._is_llm_response = True
        try:
            #  not all LLMs support structured output, so we need to cast to the specific LLM type
            llm = cast(openai.LLM, self.llm)
            tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN

            async with llm.chat(
                chat_ctx=chat_ctx,
                tools=tools,
                tool_choice=tool_choice,
                response_format=InterviewTurnJSON,
            ) as stream:
                async for chunk in stream:
                    yield chunk
        finally:
            self._is_llm_response = False

    async def _on_response_complete(self):
        """Called when LLM response is complete."""
        if not self._current_response or not self._current_response["structured_data"]:
            return

        response_time = (datetime.utcnow() - self._response_start_time).total_seconds()

        logger.info(
            f"Response completed in {response_time:.2f}s, sending complete data"
        )

        await self.data_streamer.send_structured_data(
            topic="llm-structured-output",
            data={
                "type": "response_complete",
                "content": self._current_response["structured_data"],
                "chunk_count": len(self._current_response["chunks"]),
                "response_time_seconds": response_time,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
        )

        # Reset for next response
        self._current_response = None
        self._response_start_time = None

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Text-to-speech processing with intelligent text processing."""
        with self.tracer.start_as_current_span("tts_node") as span:
            # Set user and model attributes
            self.set_user_attributes(span)
            span.set_attribute("gen_ai.request.model", "deepgram-tts")

            instruction_updated = False
            collected_text = ""
            audio_frames_count = 0

            def output_processed(resp: InterviewTurnJSON):
                nonlocal instruction_updated
                if (
                    resp.get("voice_instructions")
                    and resp.get("system_response")
                    and not instruction_updated
                ):
                    instruction_updated = True
                    logger.info(
                        f"Applying TTS instructions before generating response audio: "
                        f'"{resp["voice_instructions"]}"'
                    )

                    tts = cast(openai.TTS, self.tts)
                    try:
                        tts.update_options(instructions=resp["voice_instructions"])
                    except TypeError:
                        logger.warning(
                            "TTS.update_options() doesn't accept 'instructions' in this backend. Instruction ignored: %s",
                            resp["voice_instructions"],
                        )

            # Use smart processing based on context
            processed_text = process_structured_output(
                text, callback=output_processed, force_structured=self._is_llm_response
            )

            async for frame in Agent.default.tts_node(
                self,
                processed_text,
                model_settings,
            ):
                audio_frames_count += 1
                yield frame

            span.set_attribute("input.value", collected_text)
            span.set_attribute(
                "output.value", f"Generated {audio_frames_count} audio frames"
            )

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ):
        """Transcription node with smart processing and incremental structured data streaming."""

        chunk_count = 0
        start_time = datetime.utcnow()
        chunk_buffer = []
        last_send_time = start_time
        send_interval = 1  # Send every 500ms to avoid overwhelming frontend
        acc_text = ""

        async def text_processor():
            """Process text and send structured data without re-iterating."""
            nonlocal chunk_count, acc_text, chunk_buffer, last_send_time

            try:
                async for delta in text:
                    chunk_count += 1
                    acc_text += delta

                    # Debug logging
                    logger.debug(f"Received chunk {chunk_count}: {len(delta)} chars")

                    # Create payload matching frontend StructuredResponsePayload interface
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

                    # Send buffered chunks periodically to 'llm-transcription' topic
                    current_time = datetime.utcnow()
                    time_elapsed = (current_time - last_send_time).total_seconds()

                    if time_elapsed >= send_interval and chunk_buffer:
                        # Send without blocking
                        asyncio.create_task(
                            self.data_streamer.stream_structured_data(
                                topic="llm-transcription",  # Frontend listens to this
                                data_chunks=chunk_buffer.copy(),
                            )
                        )
                        chunk_buffer.clear()
                        last_send_time = current_time

                    # Yield the delta to continue the pipeline
                    yield delta

                # Send any remaining buffered chunks to 'llm-transcription'
                if chunk_buffer:
                    asyncio.create_task(
                        self.data_streamer.stream_structured_data(
                            topic="llm-transcription", data_chunks=chunk_buffer.copy()
                        )
                    )

                # Send final complete payload to 'llm-response-complete' topic
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
                        topic="llm-response-complete",  # Frontend listens to this
                        data=final_payload,
                    )
                )

                logger.info(
                    f"Completed streaming {chunk_count} chunks "
                    f"({len(acc_text)} chars) in {duration:.2f}s"
                )

            except Exception as e:
                logger.error(f"Error in structured data streaming: {e}", exc_info=True)
                raise

        # Process the text with our custom processor that also sends structured data
        processed_text = process_structured_output(
            text_processor(), force_structured=self._is_llm_response
        )

        # Run the default transcription processing and yield chunks immediately
        result = Agent.default.transcription_node(self, processed_text, model_settings)

        async for chunk in result:
            yield chunk

    def on_user_state_change(self, event: UserStateChangedEvent) -> None:
        """Handle user state changes with tracing."""
        old_state = event.old_state
        new_state = event.new_state
        logger.info(f"User {self.user_id} state changed: {old_state} -> {new_state}")

        # Create a span for state change
        with self.tracer.start_as_current_span("user_state_change") as span:
            self.set_user_attributes(span)
            span.set_attribute("state.old", old_state)
            span.set_attribute("state.new", new_state)

    async def close(self) -> None:
        """Clean up resources on session end."""
        await self.close_video_stream()

        if getattr(self, "_background_tasks", None) and self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Create final span
        with self.tracer.start_as_current_span("session_end") as span:
            self.set_user_attributes(span)
            span.set_attribute("session.ended", True)

    async def close_video_stream(self) -> None:
        """Close video stream if active."""
        if hasattr(self, "video_stream") and self.video_stream:
            await self.video_stream.aclose()
            self.video_stream = None


async def setup_resource_cleanup(ctx: JobContext) -> None:
    """Setup general resource cleanup to prevent unclosed connections."""

    async def cleanup_resources():
        try:
            # Give time for any pending operations to complete
            await asyncio.sleep(0.1)

            # Force garbage collection to help cleanup unclosed resources
            import gc

            gc.collect()

            # Close any remaining aiohttp sessions
            import aiohttp

            # Get all active connector instances and close them
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.TCPConnector):
                    try:
                        if not obj.closed:
                            await obj.close()
                    except Exception as e:
                        logger.debug(f"Connector cleanup warning: {e}")
                elif isinstance(obj, aiohttp.ClientSession):
                    try:
                        if not obj.closed:
                            await obj.close()
                    except Exception as e:
                        logger.debug(f"Session cleanup warning: {e}")

            logger.info("Resource cleanup completed")

        except Exception as e:
            logger.debug(f"Resource cleanup warning: {e}")

    ctx.add_shutdown_callback(cleanup_resources)


async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entry point for the LiveKit agent with S3 recording."""
    # Readable timestamp is to organize the recordings in a single directory
    # readable_timestamp = datetime.now(timezone(timedelta(hours=5, minutes=45))).strftime("%Y-%m-%d_%H-%M-%S")
    # Setup tracing
    trace_provider = setup_langfuse(metadata={"langfuse.session.id": ctx.room.name})

    # Add shutdown callback to flush traces
    async def flush_trace():
        trace_provider.force_flush()

    ctx.add_shutdown_callback(flush_trace)

    # Get user details
    user_details = get_default_user_details()

    # # Setup resource cleanup
    # await setup_resource_cleanup(ctx)

    # # Setup egress recording
    # egress_id = await setup_egress_recording(ctx, user_details,readable_timestamp)

    # if egress_id:
    #     await setup_egress_cleanup(ctx, egress_id)

    # Connect to the room
    await ctx.connect()

    # Log room connection details
    logger.info(f"F-1 Student VISA Interview room connected: {ctx.room.name}")
    logger.info(f"AI Interview Officer identity: {ctx.room.local_participant.identity}")
    logger.info(f"Found {len(ctx.room.remote_participants)} remote participants")

    # Configure agent session
    session = AgentSession(
        vad=ctx.proc.userdata["vad"], use_tts_aligned_transcript=True
    )

    # Setup transcript callback with proper session reference
    # await setup_transcript_callback(ctx, session, user_details,readable_timestamp)

    # Configure room input/output options
    room_input = RoomInputOptions(
        video_enabled=True,
        audio_enabled=True,
        noise_cancellation=noise_cancellation.BVC(),
    )

    room_output = RoomOutputOptions(audio_enabled=True, transcription_enabled=True)

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=PTEInterviewAgent(
            room=ctx.room,
            user_details=user_details,
            user_id="user-Aaraj",
        ),
        room_input_options=room_input,
        room_output_options=room_output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Agent entrypoint started, attaching to room...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
