from dotenv import load_dotenv
import logging
import os
from typing import Union, AsyncIterable, Optional, List, Dict, Any
from uuid import uuid4
from livekit.agents.voice import MetricsCollectedEvent
from langfuse import Langfuse
from setup_langfuse import setup_langfuse
from livekit import rtc, api
from livekit import agents
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from livekit.agents import (
    Agent,
    AgentSession,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    UserStateChangedEvent,
    JobContext,
    JobProcess,
    stt,
    cli,
    metrics,
    WorkerOptions,
)
from livekit.agents import UserStateChangedEvent, AgentStateChangedEvent
from livekit.plugins import (
    openai,
    # cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel


class BaseAgent(Agent):
    def __init__(
        self,
        instructions: str,
        context_vars: Optional[Dict[str, Any]] = None,
        pronunciations: Optional[Dict[str, str]] = None,
    ) -> None:
        self.instructions_template = instructions
        self.context_vars = context_vars or {}
        self.pronunciations = pronunciations or {}
        formatted_instructions = self._format_instructions()

        # Other initialization...
        self.frames: List[rtc.VideoFrame] = []
        self.last_frame_time: float = 0
        self.video_stream: Optional[rtc.VideoStream] = None
        self.interview_started = False
        self.interview_stage = "greeting"
        self.responses = []
        self.red_flags = []

        super().__init__(
            instructions=formatted_instructions,
        )

    def _format_instructions(self) -> str:
        """Format instructions with current context variables"""
        try:
            return self.instructions_template.format(**self.context_vars)
        except KeyError as e:
            # Handle missing context variables gracefully
            print(f"Warning: Missing context variable {e}")
            return self.instructions_template

    def update_context_vars(
        self, new_vars: Dict[str, Any], update_instructions: bool = True
    ) -> None:
        """
        Update context variables and optionally refresh instructions

        Args:
            new_vars: Dictionary of new context variables
            update_instructions: Whether to immediately update agent instructions
        """
        self.context_vars.update(new_vars)
        if update_instructions:
            self.instructions = self._format_instructions()

    def get_context_var(self, key: str, default: Any = None) -> Any:
        """Get a specific context variable"""
        return self.context_vars.get(key, default)

    def update_pronunciations(self, new_pronunciations: Dict[str, str]) -> None:
        """Update pronunciation dictionary at runtime"""
        self.pronunciations.update(new_pronunciations)

    async def on_enter(self) -> None:
        """Enhanced initialization for F-1 student visa interview session"""
        with self.tracer.start_as_current_span("pte-interview-session") as span:
            # Set user attributes
            self.set_user_attributes(span)
            span.set_attribute("interview_stage", self.interview_stage)

            welcome_message = (
                f"Welcome to your F-1 Student Visa Interview Practice Session"
            )
            if self.user_details.get("name"):
                welcome_message += f", {self.user_details['name']}"
            welcome_message += """. I am your AI Interview Officer, 
            and I'll be conducting a comprehensive practice interview similar to what you'll experience at the US Consulate."""

            span.set_attribute("input.value", "User Entered,Session initialization")
            span.set_attribute("output.value", welcome_message)

            await self.session.generate_reply(instructions=welcome_message)
    
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        print(f"User input transcribed: {event.transcript}, "
              f"language: {event.language}, "
              f"final: {event.is_final}, "
              f"speaker id: {event.speaker_id}")

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        async def filtered_audio():
            async for frame in audio:
                # insert custom audio preprocessing here
                yield frame
        
        async for event in Agent.default.stt_node(self, filtered_audio(), model_settings):
            # insert custom text postprocessing here 
            yield event
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        # Insert custom preprocessing here
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            # Insert custom postprocessing here
            yield chunk
    
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """
        TTS node with pronunciation adjustments using provided dictionary
        """
        async def adjust_pronunciation(input_text: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in input_text:
                modified_chunk = chunk
                # Apply pronunciation rules from the provided dictionary
                for term, pronunciation in self.pronunciations.items():
                    # Use word boundaries to avoid partial replacements
                    modified_chunk = re.sub(
                        rf'\b{re.escape(term)}\b',
                        pronunciation,
                        modified_chunk,
                        flags=re.IGNORECASE
                    )
                yield modified_chunk

        # Process with modified text through base TTS implementation
        async for frame in Agent.default.tts_node(
            self,
            adjust_pronunciation(text),
            
            model_settings
        ):
            yield frame
    
    async def realtime_audio_output_node(
    self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
) -> AsyncIterable[rtc.AudioFrame]:
        # Insert custom audio preprocessing here
        async for frame in Agent.default.realtime_audio_output_node(self, audio, model_settings):
            # Insert custom audio postprocessing here
            yield frame
    
    # async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[str]: 
    #     async for delta in text:
    #         yield delta.replace("😘", "")
    #     from livekit.agents import UserStateChangedEvent, AgentStateChangedEvent

    # @session.on("user_state_changed")
    # def on_user_state_changed(ev: UserStateChangedEvent):
    #     if ev.new_state == "speaking":
    #         print("User started speaking")
    #     elif ev.new_state == "listening":
    #         print("User stopped speaking")
    #     elif ev.new_state == "away":
    #         print("User is not present (e.g. disconnected)")

    # @session.on("agent_state_changed")
    # def on_agent_state_changed(ev: AgentStateChangedEvent):
    #     if ev.new_state == "initializing":
    #         print("Agent is starting up")
    #     elif ev.new_state == "idle":
    #         print("Agent is ready but not processing")
    #     elif ev.new_state == "listening":
    #         print("Agent is listening for user input")
    #     elif ev.new_state == "thinking":
    #         print("Agent is processing user input and generating a response")
    #     elif ev.new_state == "speaking":
    #         print("Agent started speaking")
#     async def on_user_turn_completed(
#     self, turn_ctx: ChatContext, new_message: ChatMessage,
# ) -> None:
#     # RAG function definition omitted for brevity
#     rag_content = await my_rag_lookup(new_message.text_content())
#     turn_ctx.add_message(
#         role="assistant", 
#         content=f"Additional information relevant to the user's next message: {rag_content}"
#     )
    
    async def close(self) -> None:
        await self.close_video_stream()

        # Create final span
        with self.tracer.start_as_current_span("session_end") as span:
            self.set_user_attributes(span)
            span.set_attribute("session.ended", True)

    async def close_video_stream(self) -> None:
        if self.video_stream:
            await self.video_stream.aclose()
            self.video_stream = None
# to hang up the call as part of a function call
    @function_tool
    async def end_call(self, ctx: RunContext):
        """Use this tool when the user has signaled they wish to end the current call. The session ends automatically after invoking this tool."""
        await ctx.wait_for_playout() # let the agent finish speaking
        # call API to delete_room
        ...