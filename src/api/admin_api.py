"""
FastAPI admin API for dynamic agent configuration management.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orchestrator.configuration_store import ConfigurationStore

logger = logging.getLogger("admin-api")

# Module-level singleton for configuration store
_config_store: Optional[ConfigurationStore] = None


async def get_config_store() -> ConfigurationStore:
    """
    Get the singleton configuration store instance.

    This should only be called after the app lifespan has initialized the store.
    """
    if _config_store is None:
        raise RuntimeError("Configuration store not initialized. App lifespan not started.")
    return _config_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.

    Initializes configuration store on startup and closes it on shutdown.
    """
    global _config_store

    # Startup
    logger.info("Initializing configuration store...")
    _config_store = ConfigurationStore(
        connection_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        database="agent_configs",
    )
    await _config_store.initialize()
    logger.info("Configuration store ready")

    yield

    # Shutdown
    logger.info("Closing configuration store...")
    if _config_store:
        await _config_store.close()
    logger.info("Configuration store closed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Agent Configuration API",
    description="API for managing dynamic agent configurations",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for API requests/responses


class SchemaFieldRequest(BaseModel):
    """Schema field definition."""

    field_name: str = Field(..., description="Field name (valid Python identifier)")
    field_type: str = Field(..., description="Field type (e.g., 'str', 'int', 'list[str]')")
    description: str = Field(..., description="Field description")
    required: bool = Field(True, description="Whether field is required")
    default: Optional[Any] = Field(None, description="Default value if not required")
    constraints: Optional[dict] = Field(None, description="Field constraints (e.g., {'ge': 1, 'le': 10})")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class PromptConfig(BaseModel):
    """Prompt configuration."""

    template_id: str = Field(..., description="Prompt template identifier")
    instructions: str = Field(..., description="System instructions")
    dynamic_vars: Optional[dict] = Field({}, description="Dynamic variables")
    examples: Optional[list[dict]] = Field([], description="Few-shot examples")


class SchemaConfig(BaseModel):
    """Schema configuration."""

    schema_id: str = Field(..., description="Schema identifier")
    schema_version: str = Field("1.0.0", description="Schema version")
    fields: list[SchemaFieldRequest] = Field(..., description="Schema fields")


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = Field("openai", description="LLM provider")
    model: str = Field("gpt-5-nano", description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    supports_structured_output: bool = Field(True, description="Supports structured output")


class TTSConfig(BaseModel):
    """TTS configuration."""

    provider: str = Field("deepgram", description="TTS provider")
    model: Optional[str] = Field(None, description="Model name")
    voice: Optional[str] = Field(None, description="Voice identifier")
    instruction_field: Optional[str] = Field(
        "voice_instructions", description="Schema field for TTS instructions"
    )
    streaming: bool = Field(True, description="Enable streaming")


class TracingConfig(BaseModel):
    """Tracing configuration."""

    enabled: bool = Field(False, description="Enable tracing")
    provider: str = Field("langfuse", description="Tracing provider")
    sample_rate: float = Field(1.0, ge=0.0, le=1.0, description="Sample rate")


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    vad_enabled: bool = Field(True, description="Enable VAD")
    noise_cancellation: bool = Field(True, description="Enable noise cancellation")
    transcription_enabled: bool = Field(True, description="Enable transcription")


class TenantConfig(BaseModel):
    """Tenant configuration."""

    tenant_id: str = Field("default", description="Tenant identifier")
    isolation_level: str = Field("session", description="Isolation level")


class AgentConfigRequest(BaseModel):
    """Agent configuration request."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type")
    version: str = Field("1.0.0", description="Configuration version")
    enabled: bool = Field(True, description="Whether agent is enabled")
    prompt_config: PromptConfig
    schema_config: SchemaConfig
    llm_config: LLMConfig
    tts_config: TTSConfig
    tracing_config: TracingConfig = Field(default_factory=TracingConfig)
    pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)
    tenant_config: TenantConfig = Field(default_factory=TenantConfig)


# API Endpoints


@app.post("/api/v1/agents/config", status_code=201)
async def create_agent_config(config: AgentConfigRequest):
    """
    Create a new agent configuration.

    Args:
        config: Agent configuration

    Returns:
        Created configuration details
    """
    store = await get_config_store()
    try:
        config_id = await store.create_agent_config(config.dict())
        return {
            "status": "created",
            "config_id": config_id,
            "agent_id": config.agent_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/config/{agent_id}")
async def get_agent_config(agent_id: str, version: Optional[str] = None):
    """
    Retrieve agent configuration.

    Args:
        agent_id: Agent identifier
        version: Optional version (defaults to latest)

    Returns:
        Agent configuration
    """
    store = await get_config_store()
    try:
        config = await store.get_agent_config(agent_id, version)
        return config
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/agents/config/{agent_id}")
async def update_agent_config(agent_id: str, updates: dict):
    """
    Update agent configuration.

    Args:
        agent_id: Agent identifier
        updates: Fields to update

    Returns:
        Update status
    """
    store = await get_config_store()
    try:
        success = await store.update_agent_config(agent_id, updates)
        return {"status": "updated" if success else "no_changes", "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/agents/config/{agent_id}")
async def delete_agent_config(agent_id: str):
    """
    Delete (disable) agent configuration.

    Args:
        agent_id: Agent identifier

    Returns:
        Delete status
    """
    store = await get_config_store()
    try:
        success = await store.delete_agent_config(agent_id)
        return {"status": "deleted" if success else "not_found", "agent_id": agent_id}
    except Exception as e:
        logger.error(f"Failed to delete config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/config/{agent_id}/schema/fields")
async def add_schema_field(agent_id: str, field: SchemaFieldRequest):
    """
    Add a new field to agent's structured output schema.

    Args:
        agent_id: Agent identifier
        field: Field definition

    Returns:
        Update status
    """
    store = await get_config_store()
    try:
        config = await store.get_agent_config(agent_id)

        # Add field to schema
        config["schema_config"]["fields"].append(field.dict())

        # Update config
        await store.update_agent_config(
            agent_id, {"schema_config": config["schema_config"]}
        )

        return {"status": "field_added", "field_name": field.field_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add field: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/agents/config/{agent_id}/schema/fields/{field_name}")
async def remove_schema_field(agent_id: str, field_name: str):
    """
    Remove a field from agent's structured output schema.

    Args:
        agent_id: Agent identifier
        field_name: Name of field to remove

    Returns:
        Update status
    """
    store = await get_config_store()
    try:
        config = await store.get_agent_config(agent_id)

        # Remove field
        original_count = len(config["schema_config"]["fields"])
        config["schema_config"]["fields"] = [
            f
            for f in config["schema_config"]["fields"]
            if f["field_name"] != field_name
        ]

        if len(config["schema_config"]["fields"]) == original_count:
            raise HTTPException(status_code=404, detail=f"Field not found: {field_name}")

        # Update config
        await store.update_agent_config(
            agent_id, {"schema_config": config["schema_config"]}
        )

        return {"status": "field_removed", "field_name": field_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to remove field: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/config/{agent_id}/tracing/toggle")
async def toggle_tracing(agent_id: str, enabled: bool):
    """
    Toggle Langfuse tracing for an agent.

    Args:
        agent_id: Agent identifier
        enabled: Enable or disable tracing

    Returns:
        Update status
    """
    store = await get_config_store()
    try:
        config = await store.get_agent_config(agent_id)
        config["tracing_config"]["enabled"] = enabled

        await store.update_agent_config(
            agent_id, {"tracing_config": config["tracing_config"]}
        )

        return {"status": "tracing_updated", "enabled": enabled, "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to toggle tracing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/list")
async def list_agents(enabled_only: bool = True, limit: int = 100):
    """
    List all agent configurations.

    Args:
        enabled_only: Only return enabled configurations
        limit: Maximum number of results

    Returns:
        List of agent configurations
    """
    store = await get_config_store()
    try:
        configs = await store.list_agent_configs(enabled_only, limit)
        return {"agents": configs, "count": len(configs)}
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent-config-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
