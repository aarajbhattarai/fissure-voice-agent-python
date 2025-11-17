"""
Updated entrypoint for dynamic agent system.

This replaces the hardcoded agent instantiation in src/agent.py
with dynamic configuration-based agent creation.

Usage:
    uv run python examples/dynamic_agent_entrypoint.py dev
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions, RoomOutputOptions, WorkerOptions, cli
from livekit.plugins import noise_cancellation

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interview_agent.utilities.prewarm import prewarm
from interview_agent.utilities.setup_langfuse import setup_langfuse
from orchestrator import AgentFactory, ConfigurationStore, SessionManager

logger = logging.getLogger("dynamic-agent")
load_dotenv()


async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entry point with dynamic agent creation."""

    # 1. Extract metadata from room
    room_metadata = json.loads(ctx.room.metadata or "{}")
    agent_id = room_metadata.get("agent_id", "pte-interview-agent-v1")
    user_id = room_metadata.get("user_id", "anonymous")
    tenant_id = room_metadata.get("tenant_id", "default")
    user_details = room_metadata.get("user_details", {})

    logger.info(f"Starting dynamic agent session: {agent_id} for user {user_id}")

    # 2. Initialize configuration store
    config_store = ConfigurationStore(
        connection_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        database="agent_configs",
    )

    await config_store.initialize()

    # 3. Initialize agent factory
    factory = AgentFactory(config_store)

    # 4. Initialize session manager
    session_manager = SessionManager(factory, config_store)

    # 5. Connect to room
    await ctx.connect()

    logger.info(f"Connected to room: {ctx.room.name}")
    logger.info(f"Local participant: {ctx.room.local_participant.identity}")
    logger.info(f"Remote participants: {len(ctx.room.remote_participants)}")

    # 6. Create dynamic agent session
    try:
        session_id = await session_manager.create_session(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            room=ctx.room,
            user_details=user_details,
        )

        logger.info(f"Created session: {session_id}")

    except ValueError as e:
        logger.error(f"Failed to create session: {e}")
        return

    # 7. Get agent config to check if tracing is enabled
    config = await config_store.get_agent_config(agent_id)

    # 8. Conditionally setup tracing
    if config.get("tracing_config", {}).get("enabled", False):
        trace_provider = setup_langfuse(metadata={"langfuse.session.id": session_id})

        async def flush_trace():
            trace_provider.force_flush()

        ctx.add_shutdown_callback(flush_trace)
        logger.info("Langfuse tracing enabled")

    # 9. Configure agent session
    agent = session_manager.active_sessions[session_id]

    session = AgentSession(
        vad=ctx.proc.userdata["vad"], use_tts_aligned_transcript=True
    )

    # 10. Start session
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            audio_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=True, transcription_enabled=True
        ),
    )

    logger.info("Agent session started successfully")

    # 11. Cleanup on shutdown
    async def cleanup():
        logger.info("Cleaning up session...")
        await session_manager.end_session(session_id)
        await config_store.close()

    ctx.add_shutdown_callback(cleanup)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting dynamic agent worker...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
