"""
Session and conversation storage in MongoDB.

Stores session data, conversation history, and summaries for retrieval and analysis.
"""

import logging
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .user_data import ConversationHistory, SessionData

logger = logging.getLogger("session-storage")


class SessionStorage:
    """
    Stores and retrieves session data, conversation history, and summaries.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize session storage.

        Args:
            db: MongoDB database instance
        """
        self.sessions = db.sessions
        self.conversations = db.conversations
        self.summaries = db.conversation_summaries

    async def initialize(self):
        """Create indexes for faster queries."""
        # Session indexes
        await self.sessions.create_index([("session_id", 1)])
        await self.sessions.create_index([("user_id", 1)])
        await self.sessions.create_index([("tenant_id", 1)])
        await self.sessions.create_index([("started_at", -1)])

        # Conversation indexes
        await self.conversations.create_index([("session_id", 1)])
        await self.conversations.create_index([("user_id", 1)])

        # Summary indexes
        await self.summaries.create_index([("session_id", 1)])
        await self.summaries.create_index([("user_id", 1)])
        await self.summaries.create_index([("created_at", -1)])

        logger.info("Session storage initialized")

    # =======================
    # SESSION STORAGE
    # =======================

    async def save_session(self, session_data: SessionData) -> str:
        """
        Save session data.

        Args:
            session_data: SessionData instance

        Returns:
            Inserted document ID
        """
        document = session_data.to_dict()

        # Convert datetime objects to ISO format
        if document.get("started_at"):
            document["started_at"] = document["started_at"].isoformat()
        if document.get("ended_at"):
            document["ended_at"] = document["ended_at"].isoformat()

        result = await self.sessions.insert_one(document)
        logger.info(f"Saved session: {session_data.session_id}")
        return str(result.inserted_id)

    async def update_session(
        self, session_id: str, updates: dict
    ) -> bool:
        """
        Update session data.

        Args:
            session_id: Session identifier
            updates: Fields to update

        Returns:
            True if updated
        """
        # Convert datetime objects
        if "ended_at" in updates and updates["ended_at"]:
            updates["ended_at"] = updates["ended_at"].isoformat()

        result = await self.sessions.update_one(
            {"session_id": session_id}, {"$set": updates}
        )

        return result.modified_count > 0

    async def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session document or None
        """
        session = await self.sessions.find_one({"session_id": session_id})

        if session:
            session.pop("_id", None)

        return session

    async def get_user_sessions(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        """
        Get all sessions for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of sessions

        Returns:
            List of session documents
        """
        sessions = (
            await self.sessions.find({"user_id": user_id})
            .sort("started_at", -1)
            .limit(limit)
            .to_list(limit)
        )

        for session in sessions:
            session.pop("_id", None)

        return sessions

    # =======================
    # CONVERSATION STORAGE
    # =======================

    async def save_conversation(
        self, conversation_history: ConversationHistory
    ) -> str:
        """
        Save conversation history.

        Args:
            conversation_history: ConversationHistory instance

        Returns:
            Inserted document ID
        """
        document = conversation_history.to_dict()

        result = await self.conversations.insert_one(document)
        logger.info(
            f"Saved conversation: {conversation_history.session_id} "
            f"({conversation_history.total_turns} turns)"
        )
        return str(result.inserted_id)

    async def get_conversation(self, session_id: str) -> Optional[dict]:
        """
        Get conversation history by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Conversation document or None
        """
        conversation = await self.conversations.find_one({"session_id": session_id})

        if conversation:
            conversation.pop("_id", None)

        return conversation

    async def get_user_conversations(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        """
        Get all conversations for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of conversations

        Returns:
            List of conversation documents
        """
        conversations = (
            await self.conversations.find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
            .to_list(limit)
        )

        for conversation in conversations:
            conversation.pop("_id", None)

        return conversations

    # =======================
    # SUMMARY STORAGE
    # =======================

    async def save_summary(
        self,
        session_id: str,
        user_id: str,
        summary_data: dict,
        conversation_history: Optional[dict] = None,
    ) -> str:
        """
        Save conversation summary.

        Args:
            session_id: Session identifier
            user_id: User identifier
            summary_data: Summary data from summarizer
            conversation_history: Optional full conversation history

        Returns:
            Inserted document ID
        """
        document = {
            "session_id": session_id,
            "user_id": user_id,
            "summary": summary_data["summary"],
            "structured_summary": summary_data.get("structured_summary", {}),
            "model": summary_data.get("model"),
            "tokens_used": summary_data.get("tokens_used"),
            "created_at": datetime.utcnow(),
        }

        if conversation_history:
            document["conversation_history"] = conversation_history

        result = await self.summaries.insert_one(document)
        logger.info(f"Saved summary for session: {session_id}")
        return str(result.inserted_id)

    async def get_summary(self, session_id: str) -> Optional[dict]:
        """
        Get summary by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Summary document or None
        """
        summary = await self.summaries.find_one({"session_id": session_id})

        if summary:
            summary.pop("_id", None)

        return summary

    async def get_user_summaries(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        """
        Get all summaries for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of summaries

        Returns:
            List of summary documents
        """
        summaries = (
            await self.summaries.find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
            .to_list(limit)
        )

        for summary in summaries:
            summary.pop("_id", None)

        return summaries

    # =======================
    # ANALYTICS
    # =======================

    async def get_session_stats(self, user_id: Optional[str] = None) -> dict:
        """
        Get session statistics.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Statistics dictionary
        """
        query = {"user_id": user_id} if user_id else {}

        total_sessions = await self.sessions.count_documents(query)

        # Average duration
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "avg_duration": {"$avg": "$duration_seconds"},
                    "total_turns": {"$sum": "$session_metadata.total_turns"},
                }
            },
        ]

        result = await self.sessions.aggregate(pipeline).to_list(1)

        if result:
            stats = result[0]
            return {
                "total_sessions": total_sessions,
                "avg_duration_seconds": stats.get("avg_duration", 0),
                "total_turns": stats.get("total_turns", 0),
            }
        else:
            return {
                "total_sessions": total_sessions,
                "avg_duration_seconds": 0,
                "total_turns": 0,
            }

    async def search_conversations(
        self, search_text: str, user_id: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        """
        Search conversations by text.

        Args:
            search_text: Text to search for
            user_id: Optional user ID to filter by
            limit: Maximum results

        Returns:
            List of matching conversation documents
        """
        query = {"$text": {"$search": search_text}}

        if user_id:
            query["user_id"] = user_id

        conversations = (
            await self.conversations.find(query)
            .limit(limit)
            .to_list(limit)
        )

        for conversation in conversations:
            conversation.pop("_id", None)

        return conversations


# =======================
# HELPER FUNCTIONS
# =======================


async def create_session_storage(
    connection_string: str, database: str = "agent_data"
) -> SessionStorage:
    """
    Create and initialize session storage.

    Args:
        connection_string: MongoDB connection string
        database: Database name

    Returns:
        Initialized SessionStorage instance
    """
    client = AsyncIOMotorClient(connection_string)
    db = client[database]

    storage = SessionStorage(db)
    await storage.initialize()

    return storage
