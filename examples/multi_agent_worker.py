"""
Multi-agent worker that can handle different agent types.

This example shows how to:
1. Route users to different agents based on metadata
2. Run multiple agent types from a single worker
3. Dynamically select agent configuration

Usage:
    # Setup configurations first
    uv run python examples/setup_agents.py

    # Then run this worker
    uv run python examples/multi_agent_worker.py dev
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
from agents import create_session_storage

logger = logging.getLogger("multi-agent-worker")
load_dotenv()


# =======================
# AGENT ROUTING LOGIC
# =======================

def determine_agent_type(room_metadata: dict) -> str:
    """
    Determine which agent to use based on room metadata.

    Args:
        room_metadata: Room metadata from frontend

    Returns:
        agent_id to use

    Routing strategies:
    1. Explicit: Frontend specifies agent_id
    2. Purpose-based: Frontend specifies purpose (support, sales, etc.)
    3. User-based: Route based on user attributes
    4. Default: Use default agent
    """

    # Strategy 1: Explicit agent_id
    if "agent_id" in room_metadata:
        logger.info(f"Using explicit agent: {room_metadata['agent_id']}")
        return room_metadata["agent_id"]

    # Strategy 2: Purpose-based routing
    purpose = room_metadata.get("purpose", "").lower()
    agent_map = {
        "support": "support-agent-v1",
        "sales": "sales-agent-v1",
        "onboarding": "onboarding-agent-v1",
        "interview": "pte-interview-agent-v1",
    }

    if purpose in agent_map:
        logger.info(f"Routing {purpose} session to {agent_map[purpose]}")
        return agent_map[purpose]

    # Strategy 3: User attribute-based routing
    user_details = room_metadata.get("user_details", {})

    # Example: New users get onboarding
    if user_details.get("is_new_user", False):
        logger.info("New user detected, using onboarding agent")
        return "onboarding-agent-v1"

    # Example: Premium tier gets dedicated support
    if user_details.get("tier") == "premium":
        logger.info("Premium user, using priority support agent")
        return "support-agent-v1"

    # Strategy 4: Default fallback
    logger.info("No specific routing, using default support agent")
    return "support-agent-v1"


# =======================
# ENTRYPOINT
# =======================

async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Multi-agent entrypoint that routes to appropriate agent type.
    """

    # 1. Extract metadata from room
    try:
        room_metadata = json.loads(ctx.room.metadata or "{}")
    except json.JSONDecodeError:
        logger.error("Invalid room metadata JSON")
        room_metadata = {}

    user_id = room_metadata.get("user_id", "anonymous")
    user_details = room_metadata.get("user_details", {})
    tenant_id = room_metadata.get("tenant_id", "default")

    logger.info(f"New session request - User: {user_id}, Tenant: {tenant_id}")

    # 2. Determine which agent to use
    agent_id = determine_agent_type(room_metadata)

    logger.info(f"Selected agent: {agent_id}")

    # 3. Initialize configuration store
    config_store = ConfigurationStore(
        connection_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        database="agent_configs",
    )

    await config_store.initialize()

    # 4. Initialize storage (optional)
    storage = None
    if os.getenv("MONGO_URI"):
        try:
            storage = await create_session_storage(
                connection_string=os.getenv("MONGO_URI"),
                database="agent_data",
            )
            logger.info("Session storage enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize storage: {e}")

    # 5. Initialize agent factory
    factory = AgentFactory(config_store)

    # 6. Initialize session manager
    session_manager = SessionManager(factory, config_store, storage)

    # 7. Connect to room
    await ctx.connect()

    logger.info(f"Connected to room: {ctx.room.name}")
    logger.info(f"Local participant: {ctx.room.local_participant.identity}")

    # 8. Create dynamic agent session
    try:
        session_id = await session_manager.create_session(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            room=ctx.room,
            user_details=user_details,
        )

        logger.info(f"Created session: {session_id} with agent: {agent_id}")

    except ValueError as e:
        logger.error(f"Failed to create session: {e}")
        return

    # 9. Get agent config to check if tracing is enabled
    config = await config_store.get_agent_config(agent_id)

    # 10. Conditionally setup tracing
    if config.get("tracing_config", {}).get("enabled", False):
        trace_provider = setup_langfuse(metadata={"langfuse.session.id": session_id})

        async def flush_trace():
            trace_provider.force_flush()

        ctx.add_shutdown_callback(flush_trace)
        logger.info("Langfuse tracing enabled")

    # 11. Configure agent session
    agent = session_manager.active_sessions[session_id]

    session = AgentSession(
        vad=ctx.proc.userdata["vad"], use_tts_aligned_transcript=True
    )

    # 12. Start session
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

    logger.info(f"Agent session started successfully: {agent_id}")

    # 13. Cleanup on shutdown
    async def cleanup():
        logger.info(f"Cleaning up session: {session_id}")
        await session_manager.end_session(session_id)
        await config_store.close()

    ctx.add_shutdown_callback(cleanup)


# =======================
# MAIN
# =======================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("🚀 Starting multi-agent worker...")
    logger.info("Supported agents: support, sales, onboarding, interview")
    logger.info("Routing based on room metadata 'purpose' or 'agent_id'")

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
