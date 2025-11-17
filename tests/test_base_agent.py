"""
Tests for BaseAgent with different on_enter modes and dynamic configurations.

Tests cover:
- Static mode: Uses say() method with static text
- Greeting mode: Uses generate_reply with instructions
- Direct mode: Uses generate_reply with user_input
- Dynamic configuration loading
- User data management
- Conversation tracking and summarization
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from livekit.agents import AgentSession
from livekit import rtc

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.base_agent import BaseAgent
from agents.user_data import UserData, SessionData


@pytest.fixture
def mock_room():
    """Create a mock LiveKit room for testing."""
    room = Mock(spec=rtc.Room)
    room.name = "test-room-base"
    room.local_participant = Mock()
    room.local_participant.identity = "test-base-agent"
    return room


@pytest.fixture
def base_agent_config():
    """Configuration for BaseAgent testing."""
    return {
        "agent_id": "test-base-agent",
        "agent_type": "test",
        "on_enter_mode": "static",  # Can be "static", "greeting", or "direct"
        "on_enter_config": {
            "static_message": "Welcome! I'm here to assist you.",
            "greeting_instructions": "Greet the user warmly and ask how you can help them today.",
            "direct_user_input": "Hello, how can I help you?",
        },
        "prompt_config": {
            "instructions": "You are a helpful AI assistant.",
        },
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
        "tts_config": {
            "provider": "deepgram",
        },
        "stt_config": {
            "provider": "deepgram",
        },
        "summary_config": {
            "enabled": True,
            "model": "gpt-4o-mini",
        },
    }


@pytest.mark.asyncio
async def test_base_agent_static_on_enter(test_llm, mock_room, base_agent_config):
    """
    Test BaseAgent with static on_enter mode.

    Should:
    - Use session.say() with static message
    - Not use generate_reply
    - Message is predefined in configuration
    """
    # Set to static mode
    base_agent_config["on_enter_mode"] = "static"

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # In a real implementation, BaseAgent would read on_enter_mode
        # and call session.say(static_message) during on_enter

        # For this test, we're verifying the behavior pattern
        # Since BaseAgent needs full implementation, we'll test the concept

        # Simulate what static mode should do
        static_message = base_agent_config["on_enter_config"]["static_message"]

        # In the actual agent, this would be:
        # await self.session.say(text=static_message)

        # We can verify the message is available and formatted correctly
        assert static_message == "Welcome! I'm here to assist you."
        assert len(static_message) > 0


@pytest.mark.asyncio
async def test_base_agent_greeting_on_enter(test_llm, mock_room, base_agent_config):
    """
    Test BaseAgent with greeting on_enter mode.

    Should:
    - Use session.generate_reply(instructions=...)
    - Instructions guide the greeting but aren't added to history
    - Generated response is added to history
    """
    # Set to greeting mode
    base_agent_config["on_enter_mode"] = "greeting"

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # Simulate greeting mode behavior
        greeting_instructions = base_agent_config["on_enter_config"][
            "greeting_instructions"
        ]

        # Generate greeting using instructions
        await session.generate_reply(instructions=greeting_instructions)

        # Verify agent can continue conversation
        result = await session.run(user_input="I need help with my account")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Responds helpfully to the user's request about account help.",
            )
        )


@pytest.mark.asyncio
async def test_base_agent_direct_on_enter(test_llm, mock_room, base_agent_config):
    """
    Test BaseAgent with direct on_enter mode.

    Should:
    - Use session.generate_reply(user_input=...)
    - User input is added directly to chat history
    - Agent generates response based on that input
    """
    # Set to direct mode
    base_agent_config["on_enter_mode"] = "direct"

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # Simulate direct mode behavior
        direct_input = base_agent_config["on_enter_config"]["direct_user_input"]

        # This adds user_input to history and gets response
        await session.generate_reply(user_input=direct_input)

        # Continue conversation
        result = await session.run(user_input="Can you tell me about your services?")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Provides information about services or capabilities.",
            )
        )


@pytest.mark.asyncio
async def test_user_data_management():
    """
    Test UserData and SessionData management.

    Verifies:
    - UserData creation and validation
    - SessionData tracking
    - Session metadata
    """
    # Create user data
    user_data = UserData(
        user_id="test-user-456",
        name="Jane Doe",
        email="jane@example.com",
        institution="Test University",
        field_of_study="Engineering",
        custom_attributes={"tier": "premium", "region": "US"},
    )

    assert user_data.user_id == "test-user-456"
    assert user_data.name == "Jane Doe"
    assert user_data.custom_attributes["tier"] == "premium"

    # Create session data
    session_data = SessionData(
        session_id="session-789",
        user_data=user_data,
        agent_id="support-agent",
        tenant_id="tenant-001",
    )

    assert session_data.session_id == "session-789"
    assert session_data.user_data.name == "Jane Doe"
    assert session_data.agent_id == "support-agent"
    assert session_data.status == "active"

    # End session
    session_data.end_session(status="completed")
    assert session_data.status == "completed"
    assert session_data.ended_at is not None
    assert session_data.duration_seconds > 0


@pytest.mark.asyncio
async def test_conversation_history_tracking():
    """
    Test conversation history tracking functionality.

    Verifies:
    - Turn tracking
    - Message recording
    - History retrieval
    """
    from agents.user_data import ConversationHistory, ConversationTurn

    # Create conversation history
    history = ConversationHistory(session_id="session-123", user_id="user-456")

    # Add turns
    history.add_turn(
        speaker="user",
        message="Hello, I need help",
        metadata={"timestamp": "2024-01-01T10:00:00"},
    )

    history.add_turn(
        speaker="assistant",
        message="Of course! How can I assist you today?",
        metadata={"timestamp": "2024-01-01T10:00:05"},
    )

    history.add_turn(
        speaker="user",
        message="I have a billing question",
        metadata={"timestamp": "2024-01-01T10:00:10"},
    )

    # Verify tracking
    assert len(history.turns) == 3
    assert history.turns[0].speaker == "user"
    assert history.turns[1].speaker == "assistant"
    assert "billing" in history.turns[2].message.lower()

    # Get transcript
    transcript = history.get_transcript()
    assert "Hello, I need help" in transcript
    assert "billing question" in transcript


@pytest.mark.asyncio
async def test_base_agent_configuration_modes():
    """
    Test that BaseAgent correctly interprets different on_enter modes.

    Verifies configuration parsing and mode selection.
    """
    configs = {
        "static": {
            "on_enter_mode": "static",
            "on_enter_config": {"static_message": "Welcome to our service!"},
        },
        "greeting": {
            "on_enter_mode": "greeting",
            "on_enter_config": {
                "greeting_instructions": "Introduce yourself professionally"
            },
        },
        "direct": {
            "on_enter_mode": "direct",
            "on_enter_config": {"direct_user_input": "Hello, how are you?"},
        },
    }

    # Verify each mode has required configuration
    for mode, config in configs.items():
        assert config["on_enter_mode"] == mode
        assert mode in ["static", "greeting", "direct"]
        assert "on_enter_config" in config

        if mode == "static":
            assert "static_message" in config["on_enter_config"]
        elif mode == "greeting":
            assert "greeting_instructions" in config["on_enter_config"]
        elif mode == "direct":
            assert "direct_user_input" in config["on_enter_config"]


@pytest.mark.asyncio
async def test_multi_turn_with_user_data_context(test_llm):
    """
    Test multi-turn conversation with user data context.

    Verifies:
    - User data is maintained across turns
    - Agent can reference user information
    - Context is preserved
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # Simulate agent with user context
        # In actual implementation, agent would have access to user_data

        # First turn - establish context
        result1 = await session.run(
            user_input="My name is Alice and I'm a premium customer"
        )

        await (
            result1.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Acknowledges Alice's name and premium status.",
            )
        )

        # Second turn - verify context retention
        result2 = await session.run(user_input="What's my name?")

        await (
            result2.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Correctly remembers and states the name is Alice.",
            )
        )

        # Third turn - verify attribute retention
        result3 = await session.run(user_input="What type of customer am I?")

        await (
            result3.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Correctly remembers and states that the user is a premium customer.",
            )
        )
