"""
Session manager for multi-tenant agent sessions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from livekit import rtc

from agents.session_storage import SessionStorage

from .agent_factory import AgentFactory
from .configuration_store import ConfigurationStore

logger = logging.getLogger("session-manager")


class SessionManager:
    """Manages multi-tenant agent sessions with lifecycle control."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        config_store: ConfigurationStore,
        session_storage: Optional[SessionStorage] = None,
        max_concurrent_sessions: int = 100,
    ):
        """
        Initialize session manager.

        Args:
            agent_factory: Agent factory instance
            config_store: Configuration store instance
            session_storage: Optional session storage for persistence
            max_concurrent_sessions: Maximum concurrent sessions allowed
        """
        self.factory = agent_factory
        self.config_store = config_store
        self.session_storage = session_storage
        self.active_sessions: dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def create_session(
        self,
        agent_id: str,
        user_id: str,
        tenant_id: str,
        room: rtc.Room,
        user_details: Optional[dict] = None,
        overrides: Optional[dict] = None,
    ) -> str:
        """
        Create a new agent session.

        Args:
            agent_id: Agent configuration identifier
            user_id: User identifier
            tenant_id: Tenant identifier
            room: LiveKit room instance
            user_details: User-specific metadata
            overrides: Session-specific config overrides

        Returns:
            session_id
        """
        async with self._semaphore:
            session_id = str(uuid4())

            # Store session config
            session_config = {
                "session_id": session_id,
                "agent_id": agent_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "user_details": user_details or {},
                "overrides": overrides or {},
                "created_at": datetime.utcnow(),
            }

            await self.config_store.create_session(session_config)

            # Create agent instance
            agent = await self.factory.create_agent(
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                room=room,
                user_details=user_details,
                overrides=overrides,
            )

            # Track active session
            self.active_sessions[session_id] = agent
            self._session_locks[session_id] = asyncio.Lock()

            logger.info(
                f"Created session {session_id} for user {user_id} (tenant: {tenant_id})"
            )

            return session_id

    async def get_session(self, session_id: str) -> Optional[Any]:
        """
        Get active session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Agent instance or None if not found
        """
        return self.active_sessions.get(session_id)

    async def end_session(self, session_id: str):
        """
        End an agent session and cleanup.

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            agent = self.active_sessions[session_id]

            # Acquire lock to prevent concurrent access
            async with self._session_locks[session_id]:
                # Close agent (generates summary)
                await agent.close()

                # Save to storage if configured
                if self.session_storage:
                    try:
                        # Save session data
                        await self.session_storage.save_session(agent.session_data)

                        # Save conversation history
                        await self.session_storage.save_conversation(
                            agent.conversation_history
                        )

                        # Save summary if available
                        if agent.conversation_history.summary:
                            await self.session_storage.save_summary(
                                session_id=session_id,
                                user_id=agent.user_id,
                                summary_data={
                                    "summary": agent.conversation_history.summary,
                                    "structured_summary": {},
                                    "model": agent.config.get("llm_config", {}).get(
                                        "model"
                                    ),
                                },
                                conversation_history=agent.conversation_history.to_dict(),
                            )

                        logger.info(f"Saved session data to storage: {session_id}")
                    except Exception as e:
                        logger.error(f"Failed to save session data: {e}", exc_info=True)

                del self.active_sessions[session_id]
                del self._session_locks[session_id]

            logger.info(f"Ended session {session_id}")

    async def send_to_session(self, session_id: str, message: dict):
        """
        Send message to a specific session.

        Args:
            session_id: Session identifier
            message: Message payload

        Raises:
            ValueError: If session not found
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")

        async with self._session_locks[session_id]:
            agent = self.active_sessions[session_id]
            await agent.data_streamer.send_structured_data(
                topic="agent-message", data=message
            )

    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)

    def get_sessions_by_tenant(self, tenant_id: str) -> list[str]:
        """
        Get all active session IDs for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of session IDs
        """
        # This requires storing tenant_id with agent
        # For now, return all sessions (to be implemented with tenant tracking)
        return list(self.active_sessions.keys())

    async def cleanup_inactive_sessions(self, timeout_seconds: int = 3600):
        """
        Cleanup sessions that have been inactive for too long.

        Args:
            timeout_seconds: Session timeout in seconds
        """
        current_time = datetime.utcnow()
        sessions_to_cleanup = []

        for session_id in list(self.active_sessions.keys()):
            # Get session config from store
            session_config = await self.config_store.get_session(session_id)

            if session_config:
                created_at = session_config["created_at"]
                age = (current_time - created_at).total_seconds()

                if age > timeout_seconds:
                    sessions_to_cleanup.append(session_id)

        # Cleanup identified sessions
        for session_id in sessions_to_cleanup:
            logger.info(f"Cleaning up inactive session: {session_id}")
            await self.end_session(session_id)

        logger.info(f"Cleaned up {len(sessions_to_cleanup)} inactive sessions")
