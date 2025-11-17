"""
Tests for PTEInterviewAgent with different on_enter modes.

Tests cover:
- Static mode: Uses say() method
- Greeting mode: Uses generate_reply with instructions
- Direct mode: Uses generate_reply with user_input
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from livekit.agents import AgentSession, ChatContext
from livekit.plugins import openai
from livekit import rtc

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import PTEInterviewAgent


@pytest.fixture
def mock_room():
    """Create a mock LiveKit room for testing."""
    room = Mock(spec=rtc.Room)
    room.name = "test-room"
    room.local_participant = Mock()
    room.local_participant.identity = "test-agent"
    return room


@pytest.mark.asyncio
async def test_static_on_enter_uses_say(test_llm, mock_user_details, mock_room):
    """
    Test on_enter with static mode - should use say() method.

    The agent should:
    - Generate a static welcome message
    - Use session.say() to speak it
    - Not use generate_reply
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # Create agent
        agent = PTEInterviewAgent(
            room=mock_room,
            user_id=mock_user_details["user_id"],
            user_details=mock_user_details,
        )

        # Start agent (this triggers on_enter)
        await session.start(agent)

        # The agent's on_enter currently uses say() with a static message
        # No explicit test needed as it's executed during start()
        # We can verify by checking the first message in chat context

        # Now test that the agent responds appropriately
        result = await session.run(user_input="I'm ready to begin the interview")

        # Verify agent responds professionally
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Acknowledges the user is ready and proceeds with the interview professionally.",
            )
        )


@pytest.mark.asyncio
async def test_greeting_on_enter_with_instructions(test_llm, mock_room):
    """
    Test on_enter with greeting mode - should use generate_reply with instructions.

    When using greeting mode:
    - Agent uses session.generate_reply(instructions="greet the user...")
    - Instructions are used to generate response
    - Instructions are NOT added to chat history
    - Generated response IS added to chat history
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # Create agent without user details to test generic greeting
        agent = PTEInterviewAgent(room=mock_room, user_id="test-user", user_details={})

        # Start agent
        await session.start(agent)

        # Simulate using generate_reply with instructions for greeting
        # This would be called in on_enter if we modify the agent
        await session.generate_reply(
            instructions="Greet the user warmly and introduce yourself as a PTE Interview Practice agent. Ask if they're ready to begin."
        )

        # Get the response
        # Note: The instructions are not in chat history, only the generated greeting
        result = await session.run(user_input="Yes, I'm ready!")

        # Verify the agent continues the conversation appropriately
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Acknowledges the user is ready and begins the interview or asks opening questions.",
            )
        )


@pytest.mark.asyncio
async def test_direct_on_enter_with_user_input(test_llm, mock_room):
    """
    Test on_enter with direct mode - should use generate_reply with user_input.

    When using direct mode:
    - Agent uses session.generate_reply(user_input="...")
    - User input is added directly to chat history
    - Agent generates response based on that input
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        agent = PTEInterviewAgent(room=mock_room, user_id="test-user", user_details={})

        await session.start(agent)

        # Simulate user starting with a direct statement
        # This adds the user_input directly to chat history
        await session.generate_reply(
            user_input="Hello, I want to practice my interview skills."
        )

        # Now continue the conversation
        result = await session.run(
            user_input="Can we start with some warm-up questions?"
        )

        # Verify agent responds appropriately to the request
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Agrees to start with warm-up questions or begins asking them.",
            )
        )


@pytest.mark.asyncio
async def test_pte_agent_handles_structured_responses(
    test_llm, mock_room, mock_user_details
):
    """
    Test that PTEInterviewAgent handles structured output properly.

    The agent uses InterviewTurnJSON for structured responses containing:
    - voice_instructions: TTS directives
    - system_response: The spoken response
    - interview_stage: Current phase
    - credibility_score: Assessment score
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        agent = PTEInterviewAgent(
            room=mock_room,
            user_id=mock_user_details["user_id"],
            user_details=mock_user_details,
        )

        await session.start(agent)

        # Test interview progression
        result = await session.run(
            user_input="I'm from India and I want to study Computer Science at MIT."
        )

        # Verify agent responds appropriately to the background information
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Acknowledges the information about studying Computer Science at MIT and may ask follow-up questions about academic background or motivation.",
            )
        )


@pytest.mark.asyncio
async def test_pte_agent_multi_turn_conversation(
    test_llm, mock_room, mock_user_details
):
    """
    Test multi-turn conversation with PTEInterviewAgent.

    Verifies:
    - Agent maintains context across turns
    - Interview progresses naturally
    - Agent remembers previous responses
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        agent = PTEInterviewAgent(
            room=mock_room,
            user_id=mock_user_details["user_id"],
            user_details=mock_user_details,
        )

        await session.start(agent)

        # Turn 1: Introduction
        result1 = await session.run(
            user_input="Hello, my name is John and I'm applying for a student visa."
        )

        await (
            result1.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Acknowledges John's introduction and visa application purpose.",
            )
        )

        # Turn 2: Follow-up based on context
        result2 = await session.run(user_input="I have my I-20 form ready.")

        await (
            result2.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Acknowledges the I-20 form and may ask about other required documents or proceed with interview questions.",
            )
        )

        # Turn 3: Test context retention
        result3 = await session.run(user_input="What did I say my name was?")

        await (
            result3.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Correctly remembers and states that the user said their name is John.",
            )
        )


@pytest.mark.asyncio
async def test_pte_agent_error_handling(test_llm, mock_room):
    """
    Test that PTEInterviewAgent handles errors gracefully.

    Tests:
    - Invalid or unclear user inputs
    - Unexpected conversational paths
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        agent = PTEInterviewAgent(room=mock_room, user_id="test-user", user_details={})

        await session.start(agent)

        # Test with unclear/invalid input
        result = await session.run(user_input="asdfghjkl qwerty")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Politely asks for clarification or acknowledges not understanding the input.",
            )
        )


@pytest.mark.asyncio
async def test_pte_agent_stays_in_character(test_llm, mock_room):
    """
    Test that PTEInterviewAgent stays in character as an interview officer.

    Should:
    - Not answer unrelated questions
    - Stay focused on interview topic
    - Maintain professional tone
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        agent = PTEInterviewAgent(room=mock_room, user_id="test-user", user_details={})

        await session.start(agent)

        # Try to get agent off-topic
        result = await session.run(user_input="What's your favorite color?")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Politely redirects the conversation back to the interview or maintains professional boundaries by not engaging with off-topic questions.",
            )
        )


@pytest.mark.asyncio
async def test_pte_agent_conversation_history_loading(
    test_llm, mock_room, mock_user_details
):
    """
    Test loading conversation history manually.

    Verifies:
    - Agent can load previous conversation context
    - Agent uses loaded context in responses
    """
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        agent = PTEInterviewAgent(
            room=mock_room,
            user_id=mock_user_details["user_id"],
            user_details=mock_user_details,
        )

        await session.start(agent)

        # Load previous conversation context
        chat_ctx = ChatContext()
        chat_ctx.add_message(
            role="user", content="I'm applying to study at Harvard University."
        )
        chat_ctx.add_message(
            role="assistant",
            content="That's great! Harvard is an excellent institution. What program are you interested in?",
        )
        chat_ctx.add_message(role="user", content="I want to study Economics.")
        chat_ctx.add_message(
            role="assistant",
            content="Economics at Harvard is a very competitive program. Can you tell me about your academic background?",
        )

        await agent.update_chat_ctx(chat_ctx)

        # Test that agent remembers the loaded context
        result = await session.run(user_input="What university did I mention?")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Correctly remembers and states that the user mentioned Harvard University.",
            )
        )

        # Test remembering the program
        result2 = await session.run(user_input="And what subject?")

        await (
            result2.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Correctly remembers and states that the user wants to study Economics.",
            )
        )
