"""
Pytest configuration and shared fixtures for agent testing.
"""

import os
import pytest
from livekit.agents import llm
from livekit.plugins import openai


@pytest.fixture
def test_llm() -> llm.LLM:
    """
    Fixture providing an LLM instance for testing and judgment.

    Uses gpt-4o-mini for cost-effective testing.
    """
    return openai.LLM(model="gpt-4o-mini")


@pytest.fixture
def mock_user_details() -> dict:
    """Fixture providing mock user details for testing."""
    return {
        "user_id": "test-user-123",
        "name": "Test User",
        "email": "test@example.com",
        "institution": "Test University",
        "field_of_study": "Computer Science",
    }


@pytest.fixture
def mock_room_metadata() -> dict:
    """Fixture providing mock room metadata for testing."""
    return {
        "user_id": "test-user-123",
        "tenant_id": "test-tenant",
        "purpose": "support",
        "user_details": {
            "name": "Test User",
            "email": "test@example.com",
        },
    }


@pytest.fixture(autouse=True)
def set_test_env_vars():
    """Set required environment variables for testing."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("DEEPGRAM_API_KEY", "test-key")
    os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
    os.environ.setdefault("LIVEKIT_API_KEY", "test-api-key")
    os.environ.setdefault("LIVEKIT_API_SECRET", "test-api-secret")
    yield


@pytest.fixture
def agent_config_support() -> dict:
    """Fixture providing support agent configuration for testing."""
    return {
        "agent_id": "support-agent-test",
        "agent_type": "support",
        "version": "1.0.0",
        "enabled": True,
        "prompt_config": {
            "template_id": "customer_support",
            "instructions": "You are a helpful customer support agent.",
            "dynamic_vars": {},
            "examples": [],
        },
        "schema_config": {
            "schema_id": "support_turn_v1",
            "schema_version": "1.0.0",
            "fields": [
                {
                    "field_name": "system_response",
                    "field_type": "str",
                    "description": "Support agent's response",
                    "required": True,
                },
                {
                    "field_name": "issue_category",
                    "field_type": "Literal['technical', 'billing', 'account', 'other']",
                    "description": "Category of customer issue",
                    "required": False,
                },
                {
                    "field_name": "sentiment",
                    "field_type": "Literal['positive', 'neutral', 'frustrated', 'angry']",
                    "description": "Customer sentiment",
                    "required": False,
                },
            ],
        },
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": None,
            "supports_structured_output": True,
        },
        "tts_config": {
            "provider": "deepgram",
            "streaming": True,
        },
        "stt_config": {
            "provider": "deepgram",
        },
        "tracing_config": {
            "enabled": False,  # Disabled for tests
            "provider": "langfuse",
            "sample_rate": 1.0,
        },
        "pipeline_config": {
            "vad_enabled": True,
            "noise_cancellation": True,
            "transcription_enabled": True,
        },
        "tenant_config": {
            "tenant_id": "test-tenant",
            "isolation_level": "session",
        },
    }
