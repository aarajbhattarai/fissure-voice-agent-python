"""
Configuration store for agent configurations with caching and versioning.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger("configuration-store")


class ConfigurationStore:
    """
    Manages agent configurations with versioning and caching.
    Supports MongoDB backend with in-memory LRU cache.
    """

    def __init__(
        self,
        connection_string: str,
        database: str = "agent_configs",
        cache_ttl: int = 300,
    ):
        """
        Initialize configuration store.

        Args:
            connection_string: MongoDB connection string
            database: Database name
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database]
        self.configs = self.db.agent_configurations
        self.sessions = self.db.session_configurations

        # In-memory cache with TTL
        self._cache: dict[str, tuple[dict, float]] = {}
        self._cache_ttl = cache_ttl

    async def initialize(self):
        """Initialize database indexes."""
        # Create indexes for faster queries
        await self.configs.create_index([("agent_id", 1), ("version", -1)])
        await self.configs.create_index([("enabled", 1)])
        await self.sessions.create_index([("session_id", 1)])
        await self.sessions.create_index([("user_id", 1)])
        await self.sessions.create_index([("tenant_id", 1)])

        logger.info("Configuration store initialized")

    async def get_agent_config(
        self, agent_id: str, version: Optional[str] = None
    ) -> dict:
        """
        Retrieve agent configuration by ID with caching.

        Args:
            agent_id: Unique agent identifier
            version: Optional version string (defaults to latest)

        Returns:
            Agent configuration dictionary

        Raises:
            ValueError: If configuration not found
        """
        cache_key = f"{agent_id}:{version or 'latest'}"

        # Check cache
        if cache_key in self._cache:
            config, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return config

        # Query database
        query = {"agent_id": agent_id, "enabled": True}
        if version:
            query["version"] = version

        config = await self.configs.find_one(
            query, sort=[("version", -1)]  # Get latest version
        )

        if not config:
            raise ValueError(f"Agent configuration not found: {agent_id}")

        # Remove MongoDB _id from response
        config.pop("_id", None)

        # Update cache
        self._cache[cache_key] = (config, time.time())

        logger.info(f"Loaded configuration for {agent_id}")
        return config

    async def create_agent_config(self, config: dict) -> str:
        """
        Create new agent configuration.

        Args:
            config: Agent configuration dictionary

        Returns:
            Configuration ID

        Raises:
            ValueError: If configuration is invalid
        """
        # Add timestamps
        config["created_at"] = datetime.utcnow()
        config["updated_at"] = datetime.utcnow()

        # Validate schema
        self._validate_config(config)

        # Check if agent_id already exists
        existing = await self.configs.find_one({"agent_id": config["agent_id"]})
        if existing:
            raise ValueError(
                f"Agent configuration already exists: {config['agent_id']}"
            )

        # Insert into database
        result = await self.configs.insert_one(config)

        # Invalidate cache
        self._invalidate_cache(config["agent_id"])

        logger.info(f"Created configuration for {config['agent_id']}")
        return str(result.inserted_id)

    async def update_agent_config(self, agent_id: str, updates: dict) -> bool:
        """
        Update existing agent configuration.

        Args:
            agent_id: Agent identifier
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully

        Raises:
            ValueError: If agent not found
        """
        # Add update timestamp
        updates["updated_at"] = datetime.utcnow()

        # Perform update
        result = await self.configs.update_one(
            {"agent_id": agent_id}, {"$set": updates}
        )

        if result.matched_count == 0:
            raise ValueError(f"Agent configuration not found: {agent_id}")

        # Invalidate cache
        self._invalidate_cache(agent_id)

        logger.info(f"Updated configuration for {agent_id}")
        return result.modified_count > 0

    async def delete_agent_config(self, agent_id: str) -> bool:
        """
        Soft delete agent configuration (set enabled=False).

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted successfully
        """
        result = await self.configs.update_one(
            {"agent_id": agent_id}, {"$set": {"enabled": False}}
        )

        # Invalidate cache
        self._invalidate_cache(agent_id)

        logger.info(f"Deleted configuration for {agent_id}")
        return result.modified_count > 0

    async def create_session(self, session_config: dict) -> str:
        """
        Create a new session configuration.

        Args:
            session_config: Session configuration dictionary

        Returns:
            Session ID
        """
        session_config["created_at"] = datetime.utcnow()

        result = await self.sessions.insert_one(session_config)

        logger.info(f"Created session {session_config['session_id']}")
        return str(result.inserted_id)

    async def get_session(self, session_id: str) -> Optional[dict]:
        """
        Retrieve session configuration.

        Args:
            session_id: Session identifier

        Returns:
            Session configuration or None if not found
        """
        session = await self.sessions.find_one({"session_id": session_id})

        if session:
            session.pop("_id", None)

        return session

    async def list_agent_configs(
        self, enabled_only: bool = True, limit: int = 100
    ) -> list[dict]:
        """
        List all agent configurations.

        Args:
            enabled_only: Only return enabled configurations
            limit: Maximum number of results

        Returns:
            List of agent configurations
        """
        query = {"enabled": True} if enabled_only else {}

        configs = await self.configs.find(query).limit(limit).to_list(limit)

        # Remove MongoDB _id
        for config in configs:
            config.pop("_id", None)

        return configs

    def _validate_config(self, config: dict):
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            "agent_id",
            "agent_type",
            "prompt_config",
            "schema_config",
            "llm_config",
            "tts_config",
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        # Validate schema_config structure
        if "fields" not in config["schema_config"]:
            raise ValueError("schema_config must contain 'fields' array")

        # Validate each field definition
        for field_def in config["schema_config"]["fields"]:
            required_field_attrs = ["field_name", "field_type", "description"]
            for attr in required_field_attrs:
                if attr not in field_def:
                    raise ValueError(f"Schema field missing '{attr}': {field_def}")

    def _invalidate_cache(self, agent_id: str):
        """
        Invalidate cache entries for an agent.

        Args:
            agent_id: Agent identifier
        """
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(agent_id)]
        for key in keys_to_remove:
            del self._cache[key]

        logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for {agent_id}")

    async def close(self):
        """Close database connection."""
        self.client.close()
        logger.info("Configuration store closed")
