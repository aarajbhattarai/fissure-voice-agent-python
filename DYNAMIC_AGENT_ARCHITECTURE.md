# Dynamic Agent Architecture Design

## Executive Summary

This document defines a **modular, runtime-configurable agent system** that transforms the existing hardcoded PTEInterviewAgent into a fully dynamic, multi-tenant architecture where agents, prompts, structured schemas, TTS instructions, and tracing are configured through APIs or dashboards without code changes.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Configuration Layer                             │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────────┐  │
│  │  Admin API     │  │  Dashboard UI   │  │  Configuration Store   │  │
│  │  (FastAPI)     │  │  (React/Next)   │  │  (MongoDB/PostgreSQL)  │  │
│  └────────────────┘  └─────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Orchestration Layer                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Agent Orchestrator                          │  │
│  │  - Agent Factory (dynamic instantiation)                         │  │
│  │  - Session Manager (multi-tenant)                                │  │
│  │  - Configuration Resolver (runtime config loading)               │  │
│  │  - Schema Registry (dynamic TypedDict generation)                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Agent Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Generic   │  │  Interview  │  │   Sales     │  │   Custom     │  │
│  │   Agent     │  │   Agent     │  │   Agent     │  │   Agent N    │  │
│  │  (Base)     │  │ (Configured)│  │ (Configured)│  │ (Configured) │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │
│                                                                          │
│  Each agent has:                                                         │
│  - Dynamic Prompt Engine                                                 │
│  - Dynamic Structured Output Schema                                      │
│  - Configurable TTS Instructions                                         │
│  - Optional Tracing (Langfuse/OTLP)                                      │
│  - Custom Pipeline Nodes                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Integration Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │   LiveKit    │  │  Deepgram/   │  │  Langfuse   │  │  MongoDB   │  │
│  │   (RTC/RPC)  │  │  OpenAI      │  │  (Optional) │  │  (Storage) │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Frontend Layer                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Real-time UI (React/Next.js)                                    │  │
│  │  - LiveKit Client SDK                                            │  │
│  │  - Structured Data Receiver (RPC topics)                         │  │
│  │  - Dynamic Schema Renderer                                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Configuration Store Schema

```python
# MongoDB/PostgreSQL Schema

AgentConfiguration = {
    "agent_id": str,              # Unique identifier
    "agent_type": str,            # "interview", "sales", "support", etc.
    "version": str,               # "1.0.0" (for versioning)
    "created_at": datetime,
    "updated_at": datetime,
    "enabled": bool,              # Feature flag

    # Prompt Configuration
    "prompt_config": {
        "template_id": str,       # Reference to prompt template
        "instructions": str,      # System instructions
        "dynamic_vars": dict,     # Variables to inject {key: default_value}
        "examples": list[dict],   # Few-shot examples
    },

    # Structured Output Schema
    "schema_config": {
        "schema_id": str,
        "schema_version": str,
        "fields": [
            {
                "field_name": str,        # e.g., "voice_instructions"
                "field_type": str,        # "str", "int", "list[str]", "Literal['a','b']"
                "description": str,
                "required": bool,
                "default": Any,
                "constraints": dict,      # e.g., {"ge": 1, "le": 10}
                "metadata": {
                    "use_in_tts": bool,   # Flag for TTS processing
                    "hidden": bool,       # Don't send to frontend
                }
            }
        ],
        "custom_validators": list[str]   # References to validator functions
    },

    # TTS Configuration
    "tts_config": {
        "provider": str,           # "deepgram", "openai", "cartesia"
        "model": str,
        "voice": str,
        "default_instructions": str,
        "instruction_field": str,  # Which schema field contains TTS instructions
        "streaming": bool,
    },

    # LLM Configuration
    "llm_config": {
        "provider": str,           # "openai", "anthropic", "together"
        "model": str,              # "gpt-5-nano"
        "temperature": float,
        "max_tokens": int,
        "tool_choice": str,
        "response_format": dict,   # For structured output
    },

    # Tracing Configuration
    "tracing_config": {
        "enabled": bool,
        "provider": str,           # "langfuse", "datadog", "custom"
        "endpoints": dict,
        "sample_rate": float,      # 0.0 to 1.0
        "metadata_fields": list[str],  # Which fields to trace
    },

    # Pipeline Configuration
    "pipeline_config": {
        "vad_enabled": bool,
        "noise_cancellation": bool,
        "transcription_enabled": bool,
        "custom_nodes": list[str], # References to custom node handlers
    },

    # Multi-tenant Settings
    "tenant_config": {
        "tenant_id": str,
        "isolation_level": str,    # "session", "user", "organization"
        "rate_limits": dict,
        "quotas": dict,
    }
}

# Session Configuration (Runtime)
SessionConfiguration = {
    "session_id": str,
    "agent_id": str,              # Links to AgentConfiguration
    "user_id": str,
    "tenant_id": str,
    "user_details": dict,         # Custom user metadata
    "overrides": dict,            # Session-specific config overrides
    "created_at": datetime,
    "expires_at": datetime,
}
```

### 2.2 Agent Factory (Dynamic Instantiation)

```python
# src/orchestrator/agent_factory.py

from typing import Type, Any
import importlib
from pydantic import create_model
from typing_extensions import TypedDict

class AgentFactory:
    """
    Dynamically creates agent instances based on configuration.
    Implements the Factory Pattern for agent creation.
    """

    def __init__(self, config_store: ConfigurationStore):
        self.config_store = config_store
        self.schema_registry = SchemaRegistry()
        self.agent_registry: dict[str, Type[BaseAgent]] = {}

    async def create_agent(
        self,
        agent_id: str,
        session_id: str,
        user_id: str,
        room: rtc.Room,
        overrides: dict[str, Any] = None
    ) -> BaseAgent:
        """
        Creates and configures an agent instance dynamically.

        Args:
            agent_id: Configuration identifier
            session_id: Unique session identifier
            user_id: User identifier
            room: LiveKit room instance
            overrides: Session-specific config overrides

        Returns:
            Configured agent instance
        """
        # 1. Load configuration from store
        config = await self.config_store.get_agent_config(agent_id)

        # 2. Apply session overrides
        if overrides:
            config = self._merge_config(config, overrides)

        # 3. Generate dynamic structured output schema
        schema_class = self._create_schema_from_config(config["schema_config"])

        # 4. Load prompt template
        prompt = await self._load_prompt(config["prompt_config"])

        # 5. Configure LLM
        llm = self._create_llm(config["llm_config"])

        # 6. Configure TTS
        tts = self._create_tts(config["tts_config"])

        # 7. Configure tracing
        tracer = self._create_tracer(config["tracing_config"]) if config["tracing_config"]["enabled"] else None

        # 8. Get agent class (generic or specialized)
        agent_class = self._get_agent_class(config["agent_type"])

        # 9. Instantiate agent with all configurations
        agent = agent_class(
            room=room,
            user_id=user_id,
            session_id=session_id,
            llm=llm,
            tts=tts,
            stt=self._create_stt(config.get("stt_config", {})),
            turn_detection=self._create_turn_detector(config["pipeline_config"]),
            prompt=prompt,
            schema_class=schema_class,
            tracer=tracer,
            config=config,
        )

        return agent

    def _create_schema_from_config(self, schema_config: dict) -> Type[TypedDict]:
        """
        Dynamically creates a TypedDict class from configuration.
        Uses Pydantic's create_model for runtime type generation.
        """
        fields = {}
        annotations = {}

        for field_def in schema_config["fields"]:
            field_name = field_def["field_name"]
            field_type = self._parse_type(field_def["field_type"])

            # Build Pydantic Field with constraints
            field_kwargs = {
                "description": field_def["description"],
            }

            if "constraints" in field_def:
                field_kwargs.update(field_def["constraints"])

            if not field_def.get("required", True):
                field_kwargs["default"] = field_def.get("default")

            # Create annotated field
            from pydantic import Field
            annotations[field_name] = (
                field_type,
                Field(**field_kwargs)
            )

        # Generate dynamic TypedDict
        schema_name = f"DynamicSchema_{schema_config['schema_id']}"
        DynamicSchema = type(
            schema_name,
            (TypedDict,),
            {
                "__annotations__": annotations,
                "__total__": False,  # Allow partial
            }
        )

        # Register in schema registry
        self.schema_registry.register(schema_config["schema_id"], DynamicSchema)

        return DynamicSchema

    def _parse_type(self, type_str: str) -> Type:
        """Parse string type representation to Python type."""
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list[str]": list[str],
            "list[int]": list[int],
            "dict": dict,
        }

        # Handle Literal types
        if type_str.startswith("Literal["):
            from typing import Literal
            # Extract literal values
            import ast
            values = ast.literal_eval(type_str.replace("Literal", ""))
            return Literal[values]

        return type_map.get(type_str, str)

    def _merge_config(self, base: dict, overrides: dict) -> dict:
        """Deep merge configuration with overrides."""
        import copy
        result = copy.deepcopy(base)

        def deep_merge(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value

        deep_merge(result, overrides)
        return result
```

### 2.3 Base Generic Agent

```python
# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, Optional, Type
from livekit.agents import Agent, ModelSettings, ChatContext, FunctionTool
from opentelemetry import trace

class BaseAgent(Agent, ABC):
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
    ):
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

        # Optional tracing
        self.tracer = tracer
        self.tracing_enabled = config["tracing_config"]["enabled"]

        # Data streamer
        self.data_streamer = StructuredDataStreamer(room)

        # Internal state
        self._is_llm_response = False
        self._background_tasks: set[asyncio.Task] = set()

    def trace_span(self, name: str):
        """Context manager for conditional tracing."""
        if self.tracing_enabled and self.tracer:
            return self.tracer.start_as_current_span(name)
        else:
            # No-op context manager
            from contextlib import nullcontext
            return nullcontext()

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ):
        """LLM node with dynamic schema."""
        self._is_llm_response = True

        with self.trace_span("llm_node") as span:
            if span:
                span.set_attribute("langfuse.user.id", self.user_id)
                span.set_attribute("langfuse.session.id", self.session_id)

            try:
                # Check if LLM supports structured output
                if hasattr(self.llm, "chat") and self.config["llm_config"].get("supports_structured_output"):
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
                    async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                        yield chunk
            finally:
                self._is_llm_response = False

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """TTS node with dynamic instruction field."""
        with self.trace_span("tts_node") as span:
            instruction_field = self.config["tts_config"].get("instruction_field")
            instruction_updated = False

            def on_output_processed(resp: dict):
                nonlocal instruction_updated
                if instruction_field and instruction_field in resp and not instruction_updated:
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
            processed_text = self._process_structured_output(
                text,
                callback=on_output_processed,
                force_structured=self._is_llm_response
            )

            async for frame in Agent.default.tts_node(self, processed_text, model_settings):
                yield frame

    def _process_structured_output(
        self,
        text: AsyncIterable[str],
        callback: Optional[Callable] = None,
        force_structured: bool = False,
    ) -> AsyncIterable[str]:
        """
        Process structured output using dynamic schema.
        Similar to existing process_structured_output but schema-aware.
        """
        from interview_agent.utilities.structured_output import process_structured_output
        return process_structured_output(text, callback, force_structured)

    @abstractmethod
    async def on_enter(self) -> None:
        """Called when agent session starts. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def on_exit(self) -> None:
        """Called when agent session ends. Must be implemented by subclasses."""
        pass
```

### 2.4 Configuration Store

```python
# src/orchestrator/configuration_store.py

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import json

class ConfigurationStore:
    """
    Manages agent configurations with versioning and caching.
    Supports MongoDB, PostgreSQL, or Redis backends.
    """

    def __init__(self, connection_string: str, database: str = "agent_configs"):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database]
        self.configs = self.db.agent_configurations
        self.sessions = self.db.session_configurations

        # In-memory cache with TTL
        self._cache: dict[str, tuple[dict, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_agent_config(self, agent_id: str, version: Optional[str] = None) -> dict:
        """Retrieve agent configuration by ID with caching."""
        import time

        cache_key = f"{agent_id}:{version or 'latest'}"

        # Check cache
        if cache_key in self._cache:
            config, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return config

        # Query database
        query = {"agent_id": agent_id, "enabled": True}
        if version:
            query["version"] = version

        config = await self.configs.find_one(
            query,
            sort=[("version", -1)]  # Get latest version
        )

        if not config:
            raise ValueError(f"Agent configuration not found: {agent_id}")

        # Update cache
        self._cache[cache_key] = (config, time.time())

        return config

    async def create_agent_config(self, config: dict) -> str:
        """Create new agent configuration."""
        from datetime import datetime

        config["created_at"] = datetime.utcnow()
        config["updated_at"] = datetime.utcnow()

        # Validate schema
        self._validate_config(config)

        result = await self.configs.insert_one(config)

        # Invalidate cache
        self._invalidate_cache(config["agent_id"])

        return str(result.inserted_id)

    async def update_agent_config(self, agent_id: str, updates: dict) -> bool:
        """Update existing agent configuration."""
        from datetime import datetime

        updates["updated_at"] = datetime.utcnow()

        result = await self.configs.update_one(
            {"agent_id": agent_id},
            {"$set": updates}
        )

        # Invalidate cache
        self._invalidate_cache(agent_id)

        return result.modified_count > 0

    async def create_session(self, session_config: dict) -> str:
        """Create a new session configuration."""
        result = await self.sessions.insert_one(session_config)
        return str(result.inserted_id)

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session configuration."""
        return await self.sessions.find_one({"session_id": session_id})

    def _validate_config(self, config: dict):
        """Validate configuration structure."""
        required_fields = ["agent_id", "agent_type", "prompt_config", "schema_config"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

    def _invalidate_cache(self, agent_id: str):
        """Invalidate cache entries for an agent."""
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(agent_id)]
        for key in keys_to_remove:
            del self._cache[key]
```

### 2.5 Admin API (Configuration Management)

```python
# src/api/admin_api.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import logging

app = FastAPI(title="Agent Configuration API")
logger = logging.getLogger("admin-api")

# Dependency injection
async def get_config_store() -> ConfigurationStore:
    return ConfigurationStore(
        connection_string=os.getenv("MONGO_URI"),
        database="agent_configs"
    )

class AgentConfigRequest(BaseModel):
    agent_id: str
    agent_type: str
    prompt_config: dict
    schema_config: dict
    llm_config: dict
    tts_config: dict
    tracing_config: dict
    pipeline_config: dict
    tenant_config: Optional[dict] = None

class SchemaFieldRequest(BaseModel):
    field_name: str
    field_type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[dict] = None
    metadata: Optional[dict] = None

@app.post("/api/v1/agents/config")
async def create_agent_config(
    config: AgentConfigRequest,
    store: ConfigurationStore = Depends(get_config_store)
):
    """Create a new agent configuration."""
    try:
        config_id = await store.create_agent_config(config.dict())
        return {"config_id": config_id, "agent_id": config.agent_id}
    except Exception as e:
        logger.error(f"Failed to create config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents/config/{agent_id}")
async def get_agent_config(
    agent_id: str,
    version: Optional[str] = None,
    store: ConfigurationStore = Depends(get_config_store)
):
    """Retrieve agent configuration."""
    try:
        config = await store.get_agent_config(agent_id, version)
        return config
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/api/v1/agents/config/{agent_id}")
async def update_agent_config(
    agent_id: str,
    updates: dict,
    store: ConfigurationStore = Depends(get_config_store)
):
    """Update agent configuration."""
    success = await store.update_agent_config(agent_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "updated", "agent_id": agent_id}

@app.post("/api/v1/agents/config/{agent_id}/schema/fields")
async def add_schema_field(
    agent_id: str,
    field: SchemaFieldRequest,
    store: ConfigurationStore = Depends(get_config_store)
):
    """Add a new field to agent's structured output schema."""
    config = await store.get_agent_config(agent_id)

    # Add field to schema
    config["schema_config"]["fields"].append(field.dict())

    # Update config
    await store.update_agent_config(agent_id, {"schema_config": config["schema_config"]})

    return {"status": "field_added", "field_name": field.field_name}

@app.delete("/api/v1/agents/config/{agent_id}/schema/fields/{field_name}")
async def remove_schema_field(
    agent_id: str,
    field_name: str,
    store: ConfigurationStore = Depends(get_config_store)
):
    """Remove a field from agent's structured output schema."""
    config = await store.get_agent_config(agent_id)

    # Remove field
    config["schema_config"]["fields"] = [
        f for f in config["schema_config"]["fields"]
        if f["field_name"] != field_name
    ]

    await store.update_agent_config(agent_id, {"schema_config": config["schema_config"]})

    return {"status": "field_removed", "field_name": field_name}

@app.post("/api/v1/agents/config/{agent_id}/tracing/toggle")
async def toggle_tracing(
    agent_id: str,
    enabled: bool,
    store: ConfigurationStore = Depends(get_config_store)
):
    """Toggle Langfuse tracing for an agent."""
    await store.update_agent_config(
        agent_id,
        {"tracing_config.enabled": enabled}
    )
    return {"status": "tracing_updated", "enabled": enabled}

@app.get("/api/v1/agents/list")
async def list_agents(
    store: ConfigurationStore = Depends(get_config_store)
):
    """List all agent configurations."""
    configs = await store.configs.find({"enabled": True}).to_list(100)
    return {"agents": configs}
```

---

## 3. Data Flow Architecture

### 3.1 Agent Creation Flow

```
1. API Request / Dashboard Action
   ↓
2. Admin API receives create_session request
   ↓
3. Orchestrator calls AgentFactory.create_agent()
   ↓
4. Factory loads config from ConfigurationStore
   ↓
5. Factory generates dynamic schema class
   ↓
6. Factory instantiates LLM, TTS, STT with config
   ↓
7. Factory creates BaseAgent instance
   ↓
8. Agent connects to LiveKit room
   ↓
9. Session starts, agent.on_enter() called
   ↓
10. Ready to process user input
```

### 3.2 Runtime Processing Flow

```
User Audio Input
   ↓
STT Node (speech → text)
   ↓
LLM Node (text → structured JSON with dynamic schema)
   ↓
Transcription Node (extract system_response field)
   │
   ├─→ Send structured data to frontend via RPC
   │   (topic: "llm-structured-response")
   │
   └─→ TTS Node (apply voice_instructions if configured)
       ↓
   Audio Output to User
```

### 3.3 Configuration Update Flow

```
Dashboard/API Update
   ↓
ConfigurationStore.update_agent_config()
   ↓
Cache Invalidation
   ↓
Active Sessions:
   Option 1: Continue with old config (session-level isolation)
   Option 2: Hot-reload (send config update event via RPC)
   ↓
New Sessions:
   Load updated config from store
```

---

## 4. Code Patterns & Implementation Examples

### 4.1 Dynamic Session Initialization

```python
# src/orchestrator/session_manager.py

class SessionManager:
    """Manages multi-tenant agent sessions."""

    def __init__(self, agent_factory: AgentFactory, config_store: ConfigurationStore):
        self.factory = agent_factory
        self.config_store = config_store
        self.active_sessions: dict[str, BaseAgent] = {}

    async def create_session(
        self,
        agent_id: str,
        user_id: str,
        tenant_id: str,
        room: rtc.Room,
        user_details: dict = None,
        overrides: dict = None,
    ) -> str:
        """
        Create a new agent session.

        Returns:
            session_id
        """
        from uuid import uuid4

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
            overrides=overrides,
        )

        # Track active session
        self.active_sessions[session_id] = agent

        logger.info(f"Created session {session_id} for user {user_id}")

        return session_id

    async def end_session(self, session_id: str):
        """End an agent session and cleanup."""
        if session_id in self.active_sessions:
            agent = self.active_sessions[session_id]
            await agent.close()
            del self.active_sessions[session_id]

            logger.info(f"Ended session {session_id}")
```

### 4.2 Modified Entrypoint with Dynamic Agent

```python
# src/agent.py (updated)

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entry point with dynamic agent creation."""

    # 1. Extract metadata from room
    room_metadata = json.loads(ctx.room.metadata or "{}")
    agent_id = room_metadata.get("agent_id", "default-interview-agent")
    user_id = room_metadata.get("user_id", "anonymous")
    tenant_id = room_metadata.get("tenant_id", "default")
    user_details = room_metadata.get("user_details", {})

    # 2. Initialize configuration store
    config_store = ConfigurationStore(
        connection_string=os.getenv("MONGO_URI"),
        database="agent_configs"
    )

    # 3. Initialize agent factory
    factory = AgentFactory(config_store)

    # 4. Initialize session manager
    session_manager = SessionManager(factory, config_store)

    # 5. Connect to room
    await ctx.connect()

    # 6. Create dynamic agent session
    session_id = await session_manager.create_session(
        agent_id=agent_id,
        user_id=user_id,
        tenant_id=tenant_id,
        room=ctx.room,
        user_details=user_details,
    )

    # 7. Get agent config to check if tracing is enabled
    config = await config_store.get_agent_config(agent_id)

    # 8. Conditionally setup tracing
    if config["tracing_config"]["enabled"]:
        trace_provider = setup_langfuse(metadata={"langfuse.session.id": session_id})
        ctx.add_shutdown_callback(lambda: trace_provider.force_flush())

    # 9. Configure agent session
    agent = session_manager.active_sessions[session_id]

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        use_tts_aligned_transcript=True
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
            audio_enabled=True,
            transcription_enabled=True
        ),
    )

    # 11. Cleanup on shutdown
    async def cleanup():
        await session_manager.end_session(session_id)

    ctx.add_shutdown_callback(cleanup)
```

### 4.3 Adding/Removing Schema Fields at Runtime

```python
# Example API usage to modify schema

import httpx

async def add_pronunciation_score_field():
    """Add a new field to the agent schema without code changes."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agents/config/interview-agent-v1/schema/fields",
            json={
                "field_name": "pronunciation_score",
                "field_type": "int",
                "description": "Pronunciation accuracy score (0-100)",
                "required": False,
                "default": 0,
                "constraints": {"ge": 0, "le": 100},
                "metadata": {
                    "use_in_tts": False,
                    "hidden": False,
                }
            }
        )

    print(f"Field added: {response.json()}")

async def remove_field():
    """Remove a field from schema."""

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            "http://localhost:8000/api/v1/agents/config/interview-agent-v1/schema/fields/pronunciation_score"
        )

    print(f"Field removed: {response.json()}")
```

### 4.4 Toggle Tracing Per Session

```python
# Toggle tracing for specific agent

async def toggle_langfuse_tracing(agent_id: str, enabled: bool):
    """Enable/disable Langfuse tracing for an agent."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/v1/agents/config/{agent_id}/tracing/toggle",
            params={"enabled": enabled}
        )

    print(f"Tracing {'enabled' if enabled else 'disabled'}: {response.json()}")

# Usage
await toggle_langfuse_tracing("interview-agent-v1", enabled=True)
```

---

## 5. Advanced Considerations

### 5.1 Async Operations & Concurrency

```python
# High-concurrency session handling

class ConcurrentSessionManager(SessionManager):
    """Session manager with asyncio concurrency controls."""

    def __init__(self, *args, max_concurrent_sessions: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def create_session(self, *args, **kwargs) -> str:
        """Create session with concurrency control."""
        async with self._semaphore:
            session_id = await super().create_session(*args, **kwargs)
            self._session_locks[session_id] = asyncio.Lock()
            return session_id

    async def send_to_session(self, session_id: str, message: dict):
        """Thread-safe message sending."""
        async with self._session_locks[session_id]:
            agent = self.active_sessions[session_id]
            await agent.data_streamer.send_structured_data(
                topic="agent-message",
                data=message
            )
```

### 5.2 Data Validation & Type Safety

```python
# Runtime schema validation with Pydantic

from pydantic import create_model, ValidationError

class SchemaValidator:
    """Validates structured output against dynamic schemas."""

    def __init__(self, schema_config: dict):
        self.pydantic_model = self._build_pydantic_model(schema_config)

    def _build_pydantic_model(self, schema_config: dict):
        """Build Pydantic model from schema config."""
        fields = {}

        for field_def in schema_config["fields"]:
            field_type = self._parse_type(field_def["field_type"])
            field_kwargs = {}

            if not field_def.get("required", True):
                field_kwargs["default"] = field_def.get("default")

            if "constraints" in field_def:
                field_kwargs.update(field_def["constraints"])

            fields[field_def["field_name"]] = (field_type, Field(**field_kwargs))

        return create_model(
            f"Schema_{schema_config['schema_id']}",
            **fields
        )

    def validate(self, data: dict) -> dict:
        """Validate data and return validated dict."""
        try:
            validated = self.pydantic_model(**data)
            return validated.dict()
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise
```

### 5.3 Logging & Error Handling

```python
# Structured logging with correlation IDs

import structlog

logger = structlog.get_logger()

class StructuredLogger:
    """Context-aware logger for multi-tenant environments."""

    def __init__(self, session_id: str, user_id: str, tenant_id: str):
        self.logger = logger.bind(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    def log_agent_event(self, event_type: str, **kwargs):
        """Log agent events with full context."""
        self.logger.info(
            "agent_event",
            event_type=event_type,
            **kwargs
        )

    def log_error(self, error: Exception, context: dict):
        """Log errors with context."""
        self.logger.error(
            "agent_error",
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )

# Usage in BaseAgent
class BaseAgent(Agent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = StructuredLogger(
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.config["tenant_config"]["tenant_id"]
        )
```

### 5.4 Multi-Tenant Isolation

```python
# Tenant-level resource isolation

class TenantIsolationManager:
    """Enforces tenant isolation and quotas."""

    def __init__(self):
        self.tenant_quotas: dict[str, dict] = {}
        self.tenant_usage: dict[str, dict] = {}

    async def check_quota(self, tenant_id: str, resource: str) -> bool:
        """Check if tenant has available quota for resource."""
        quota = self.tenant_quotas.get(tenant_id, {}).get(resource, float('inf'))
        usage = self.tenant_usage.get(tenant_id, {}).get(resource, 0)

        return usage < quota

    async def increment_usage(self, tenant_id: str, resource: str, amount: int = 1):
        """Increment resource usage for tenant."""
        if tenant_id not in self.tenant_usage:
            self.tenant_usage[tenant_id] = {}

        self.tenant_usage[tenant_id][resource] = \
            self.tenant_usage[tenant_id].get(resource, 0) + amount

    async def reset_usage(self, tenant_id: str):
        """Reset usage counters (e.g., monthly reset)."""
        self.tenant_usage[tenant_id] = {}

# Integration in SessionManager
class SessionManager:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isolation_manager = TenantIsolationManager()

    async def create_session(self, tenant_id: str, *args, **kwargs):
        """Check quotas before creating session."""
        if not await self.isolation_manager.check_quota(tenant_id, "sessions"):
            raise QuotaExceededError(f"Tenant {tenant_id} has exceeded session quota")

        session_id = await super().create_session(tenant_id=tenant_id, *args, **kwargs)

        await self.isolation_manager.increment_usage(tenant_id, "sessions")

        return session_id
```

### 5.5 Hot Configuration Reload

```python
# Live config updates without restarting agents

class ConfigWatcher:
    """Watches for configuration changes and notifies active agents."""

    def __init__(self, config_store: ConfigurationStore, session_manager: SessionManager):
        self.config_store = config_store
        self.session_manager = session_manager
        self._watch_task: Optional[asyncio.Task] = None

    async def start_watching(self):
        """Start watching for config changes."""
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def _watch_loop(self):
        """Poll for configuration changes."""
        while True:
            try:
                # Use MongoDB change streams or polling
                async for change in self.config_store.watch_changes():
                    agent_id = change["agent_id"]

                    # Notify affected sessions
                    await self._notify_sessions(agent_id, change)

                await asyncio.sleep(5)  # Poll interval
            except Exception as e:
                logger.error(f"Config watch error: {e}")

    async def _notify_sessions(self, agent_id: str, change: dict):
        """Notify active sessions of config changes."""
        affected_sessions = [
            (sid, agent) for sid, agent in self.session_manager.active_sessions.items()
            if agent.config["agent_id"] == agent_id
        ]

        for session_id, agent in affected_sessions:
            # Option 1: Send notification to frontend
            await agent.data_streamer.send_structured_data(
                topic="config-update",
                data={
                    "type": "config_changed",
                    "agent_id": agent_id,
                    "changes": change,
                    "action": "reload_recommended"
                }
            )

            # Option 2: Hot reload (if supported by specific config changes)
            # await agent.reload_config(change)
```

---

## 6. Migration Strategy

### 6.1 Phase 1: Parallel Implementation

1. Keep existing `PTEInterviewAgent` working
2. Implement new dynamic system alongside
3. Create migration scripts for existing prompts/schemas
4. Add feature flag to switch between old/new

### 6.2 Phase 2: Configuration Migration

```python
# Migration script: Convert hardcoded config to database

async def migrate_existing_agent():
    """Migrate PTEInterviewAgent to dynamic configuration."""

    config_store = ConfigurationStore(...)

    # Create config from existing hardcoded values
    config = {
        "agent_id": "pte-interview-agent-v1",
        "agent_type": "interview",
        "version": "1.0.0",
        "enabled": True,

        "prompt_config": {
            "template_id": "pte_interview",
            "instructions": load_prompt("pte_interview.yaml"),
            "dynamic_vars": {},
        },

        "schema_config": {
            "schema_id": "interview_turn_json_v1",
            "schema_version": "1.0.0",
            "fields": [
                {
                    "field_name": "voice_instructions",
                    "field_type": "str",
                    "description": "Specific TTS directive for tone, pace, and emphasis",
                    "required": False,
                    "metadata": {"use_in_tts": True}
                },
                {
                    "field_name": "system_response",
                    "field_type": "str",
                    "description": "The officer's spoken response",
                    "required": True,
                },
                {
                    "field_name": "internal_assessment",
                    "field_type": "str",
                    "description": "Private evaluation of interaction",
                    "required": False,
                },
                {
                    "field_name": "interview_stage",
                    "field_type": "Literal['document_check', 'background_inquiry', 'academic_assessment', 'financial_review', 'intent_evaluation', 'final_decision']",
                    "description": "Current phase of the interview process",
                    "required": False,
                },
                {
                    "field_name": "credibility_score",
                    "field_type": "int",
                    "description": "Running assessment score (1–10)",
                    "required": False,
                    "constraints": {"ge": 1, "le": 10}
                },
                {
                    "field_name": "red_flags",
                    "field_type": "list[str]",
                    "description": "Specific concerns identified",
                    "required": False,
                    "default": []
                }
            ]
        },

        "llm_config": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0.7,
            "supports_structured_output": True,
        },

        "tts_config": {
            "provider": "deepgram",
            "instruction_field": "voice_instructions",
            "streaming": True,
        },

        "tracing_config": {
            "enabled": True,
            "provider": "langfuse",
            "sample_rate": 1.0,
        },

        "pipeline_config": {
            "vad_enabled": True,
            "noise_cancellation": True,
            "transcription_enabled": True,
        },

        "tenant_config": {
            "tenant_id": "default",
            "isolation_level": "session",
        }
    }

    await config_store.create_agent_config(config)

    print("Migration complete!")
```

### 6.3 Phase 3: Gradual Rollout

1. Deploy dynamic system to staging
2. Run A/B test: 10% traffic to new system
3. Monitor metrics: latency, errors, tracing overhead
4. Gradually increase to 100%
5. Deprecate old system

---

## 7. Performance Optimization

### 7.1 Schema Caching

```python
class SchemaRegistry:
    """Cache dynamically generated schemas to avoid regeneration."""

    def __init__(self):
        self._schemas: dict[str, Type[TypedDict]] = {}
        self._schema_versions: dict[str, str] = {}

    def get(self, schema_id: str, version: str) -> Optional[Type[TypedDict]]:
        """Get cached schema."""
        key = f"{schema_id}:{version}"
        return self._schemas.get(key)

    def register(self, schema_id: str, version: str, schema_class: Type[TypedDict]):
        """Register schema in cache."""
        key = f"{schema_id}:{version}"
        self._schemas[key] = schema_class
        self._schema_versions[schema_id] = version
```

### 7.2 Connection Pooling

```python
# Reuse database connections across sessions

from motor.motor_asyncio import AsyncIOMotorClient

class ConnectionPool:
    """Singleton connection pool for MongoDB."""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_client(cls, uri: str) -> AsyncIOMotorClient:
        if cls._client is None:
            cls._client = AsyncIOMotorClient(
                uri,
                maxPoolSize=50,
                minPoolSize=10,
            )
        return cls._client
```

---

## 8. Security Considerations

### 8.1 Configuration Access Control

```python
# Role-based access control for configuration

class ConfigACL:
    """Access control for agent configurations."""

    ROLE_PERMISSIONS = {
        "admin": ["create", "read", "update", "delete"],
        "developer": ["read", "update"],
        "viewer": ["read"],
    }

    def check_permission(self, user_role: str, action: str) -> bool:
        """Check if role has permission for action."""
        return action in self.ROLE_PERMISSIONS.get(user_role, [])

# Usage in API
@app.post("/api/v1/agents/config")
async def create_agent_config(
    config: AgentConfigRequest,
    user: User = Depends(get_current_user),
    acl: ConfigACL = Depends(get_acl)
):
    if not acl.check_permission(user.role, "create"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Proceed with creation
    ...
```

### 8.2 Input Validation

```python
# Prevent injection attacks in dynamic configurations

from pydantic import validator

class SchemaFieldRequest(BaseModel):
    field_name: str
    field_type: str

    @validator("field_name")
    def validate_field_name(cls, v):
        """Ensure field name is safe."""
        if not v.isidentifier():
            raise ValueError("Invalid field name: must be valid Python identifier")
        return v

    @validator("field_type")
    def validate_field_type(cls, v):
        """Ensure field type is whitelisted."""
        allowed_types = ["str", "int", "float", "bool", "list[str]", "list[int]", "dict"]
        if not any(v.startswith(t) for t in allowed_types):
            raise ValueError(f"Invalid field type: {v}")
        return v
```

---

## 9. Testing Strategy

### 9.1 Unit Tests for Dynamic Components

```python
# tests/test_agent_factory.py

import pytest
from src.orchestrator.agent_factory import AgentFactory

@pytest.mark.asyncio
async def test_create_agent_with_dynamic_schema():
    """Test agent creation with custom schema."""

    config_store = MockConfigurationStore()
    factory = AgentFactory(config_store)

    # Mock configuration
    config_store.set_mock_config("test-agent", {
        "agent_id": "test-agent",
        "agent_type": "interview",
        "schema_config": {
            "schema_id": "test_schema",
            "fields": [
                {
                    "field_name": "custom_field",
                    "field_type": "str",
                    "description": "Test field",
                    "required": True,
                }
            ]
        },
        # ... other config
    })

    agent = await factory.create_agent(
        agent_id="test-agent",
        session_id="test-session",
        user_id="test-user",
        room=MockRoom(),
    )

    # Verify schema has custom field
    assert hasattr(agent.schema_class, "__annotations__")
    assert "custom_field" in agent.schema_class.__annotations__
```

### 9.2 Integration Tests

```python
# tests/test_dynamic_agent_integration.py

@pytest.mark.asyncio
async def test_end_to_end_dynamic_agent():
    """Test full agent lifecycle with dynamic config."""

    # 1. Create configuration
    config_api = TestClient(app)
    response = config_api.post("/api/v1/agents/config", json={...})
    agent_id = response.json()["agent_id"]

    # 2. Create session
    session_manager = SessionManager(...)
    session_id = await session_manager.create_session(
        agent_id=agent_id,
        user_id="test-user",
        tenant_id="test-tenant",
        room=MockRoom(),
    )

    # 3. Verify agent is active
    assert session_id in session_manager.active_sessions

    # 4. Send message and verify response
    agent = session_manager.active_sessions[session_id]
    # ... test agent interaction
```

---

## 10. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
│                         (NGINX/ALB)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│   Admin API Service      │   │   LiveKit Agent Workers  │
│   (FastAPI)              │   │   (Dynamic Agents)       │
│   - Port 8000            │   │   - Scale 0-100          │
│   - Replica: 2-3         │   │   - Auto-scale on load   │
└──────────────────────────┘   └──────────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│   Configuration DB       │   │   LiveKit Server         │
│   (MongoDB/PostgreSQL)   │   │   (Media Server)         │
│   - Primary + Replicas   │   │   - SFU/MCU              │
└──────────────────────────┘   └──────────────────────────┘
```

---

## Conclusion

This architecture provides:

✅ **Full Runtime Configurability**: All agent attributes modifiable without code changes
✅ **Encapsulation**: Each agent is self-contained with its own config
✅ **Data Abstraction**: Standardized interfaces via RPC and APIs
✅ **Scalability**: Horizontal scaling of agent workers
✅ **Extensibility**: Easy to add new agent types and features
✅ **Multi-tenancy**: Tenant isolation and quotas
✅ **Optional Tracing**: Toggle Langfuse per agent/session
✅ **Type Safety**: Dynamic schema generation with validation
✅ **Production-Ready**: Error handling, logging, testing, security

This design transforms your current hardcoded system into a flexible, dashboard-configurable platform suitable for SaaS deployments where different customers need different agent behaviors, schemas, and features.
