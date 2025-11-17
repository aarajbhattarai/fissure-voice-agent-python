"""
Tests for function tools with mocking.

Tests cover:
- Tool call with correct arguments
- Tool output validation
- Tool error handling
- Mocking tools for edge cases
- Complex tool interactions
"""

import pytest
from livekit.agents import AgentSession, mock_tools, function_tool
from livekit.plugins import openai


# Define a simple test agent with tools
class TestAgentWithTools:
    """Simple agent with tools for testing."""

    def __init__(self):
        from livekit.agents import Agent
        from livekit.plugins import deepgram

        self._agent = Agent(
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=deepgram.TTS(),
        )

    @function_tool()
    def get_user_balance(self, user_id: str) -> str:
        """Get the account balance for a user."""
        # This would query a database in reality
        return f"User {user_id} has a balance of $1,250.00"

    @function_tool()
    def process_payment(
        self, user_id: str, amount: float, currency: str = "USD"
    ) -> str:
        """Process a payment for a user."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return f"Processed payment of {amount} {currency} for user {user_id}"

    @function_tool()
    def lookup_weather(self, location: str) -> str:
        """Look up weather for a location."""
        # Simulated weather lookup
        weather_data = {
            "Tokyo": "sunny with a temperature of 70 degrees.",
            "London": "cloudy with occasional rain, temperature of 55 degrees.",
            "New York": "partly cloudy, temperature of 65 degrees.",
        }
        return weather_data.get(location, "UNSUPPORTED_LOCATION")


@pytest.fixture
def test_llm():
    """Fixture providing test LLM."""
    return openai.LLM(model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_tool_call_with_correct_arguments(test_llm):
    """
    Test that agent calls tool with correct arguments.

    Verifies:
    - Tool is called
    - Arguments are correct
    - Output is processed
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        result = await session.run(
            user_input="What's my account balance? My user ID is user-123"
        )

        # Verify tool call
        result.expect.next_event().is_function_call(
            name="get_user_balance", arguments={"user_id": "user-123"}
        )

        # Verify tool output
        result.expect.next_event().is_function_call_output()

        # Verify agent response incorporates the balance
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Informs the user about their account balance of $1,250.00.",
            )
        )


@pytest.mark.asyncio
async def test_tool_with_multiple_arguments(test_llm):
    """
    Test tool call with multiple arguments including optional ones.

    Verifies:
    - Multiple arguments are passed correctly
    - Optional arguments work
    - Default values are applied
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        result = await session.run(
            user_input="Process a payment of 50 dollars for user-456"
        )

        # Verify tool call with arguments
        result.expect.next_event().is_function_call(
            name="process_payment",
            arguments={"user_id": "user-456", "amount": 50.0},
        )

        # Verify output
        result.expect.next_event().is_function_call_output()

        # Verify response
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Confirms the payment was processed successfully.",
            )
        )


@pytest.mark.asyncio
async def test_mock_tool_for_error_handling(test_llm):
    """
    Test tool error handling with mocked tool.

    Verifies:
    - Tool errors are caught
    - Agent handles errors gracefully
    - Appropriate error message to user
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        # Mock the tool to raise an error
        with mock_tools(
            TestAgentWithTools,
            {
                "get_user_balance": lambda user_id: RuntimeError(
                    "Database connection failed"
                )
            },
        ):
            result = await session.run(
                user_input="What's my balance? User ID is user-789"
            )

            # Tool call happens
            result.expect.next_event().is_function_call(name="get_user_balance")

            # Tool returns error
            result.expect.next_event().is_function_call_output()

            # Agent communicates error to user
            await result.expect.next_event(type="message").judge(
                llm,
                intent="Acknowledges that there was an error retrieving the balance and communicates this to the user.",
            )


@pytest.mark.asyncio
async def test_mock_tool_for_edge_cases(test_llm):
    """
    Test edge cases using mocked tools.

    Tests:
    - Unsupported values
    - Invalid inputs
    - Boundary conditions
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        # Mock weather tool to return unsupported location
        def _mock_weather(location: str) -> str:
            if location.lower() == "mars":
                return "UNSUPPORTED_LOCATION"
            return "sunny"

        with mock_tools(TestAgentWithTools, {"lookup_weather": _mock_weather}):
            result = await session.run(user_input="What's the weather on Mars?")

            # Verify agent handles unsupported location
            await result.expect.next_event(type="message").judge(
                llm,
                intent="Communicates that weather information is not available for Mars or that the location is not supported.",
            )


@pytest.mark.asyncio
async def test_mock_tool_with_specific_returns(test_llm):
    """
    Test mocking tool with specific return values.

    Verifies:
    - Mocked tools can return specific test data
    - Agent processes mocked data correctly
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        # Mock with specific test data
        def _mock_balance(user_id: str) -> str:
            if user_id == "vip-user":
                return "User vip-user has a balance of $1,000,000.00"
            return "User has a balance of $0.00"

        with mock_tools(TestAgentWithTools, {"get_user_balance": _mock_balance}):
            result = await session.run(user_input="Check balance for vip-user")

            result.expect.next_event().is_function_call(name="get_user_balance")
            result.expect.next_event().is_function_call_output()

            # Verify large balance is communicated
            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(
                    llm,
                    intent="Informs about a balance of $1,000,000.00 for vip-user.",
                )
            )


@pytest.mark.asyncio
async def test_tool_validation_error(test_llm):
    """
    Test tool validation with invalid arguments.

    Verifies:
    - Validation errors are caught
    - Agent handles invalid inputs
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        # Mock to raise validation error
        def _mock_payment(user_id: str, amount: float, currency: str = "USD") -> str:
            if amount <= 0:
                raise ValueError("Amount must be positive")
            return f"Processed {amount} {currency}"

        with mock_tools(TestAgentWithTools, {"process_payment": _mock_payment}):
            result = await session.run(
                user_input="Process a payment of -50 dollars for user-999"
            )

            # Tool called with negative amount
            result.expect.next_event().is_function_call(name="process_payment")

            # Error returned
            result.expect.next_event().is_function_call_output()

            # Agent handles validation error
            await result.expect.next_event(type="message").judge(
                llm,
                intent="Explains that the payment amount must be positive or that there was an error with the requested amount.",
            )


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_turn(test_llm):
    """
    Test agent making multiple tool calls in a single turn.

    Verifies:
    - Multiple tools can be called
    - Tools are called in sequence
    - All outputs are processed
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        # Request that might trigger multiple tools
        result = await session.run(
            user_input="Check my balance for user-111 and also tell me the weather in Tokyo"
        )

        # May call get_user_balance
        # May call lookup_weather
        # Order depends on LLM

        # Use search to find both tools were called
        result.expect.contains_function_call(name="get_user_balance")
        result.expect.contains_function_call(name="lookup_weather")

        # Final message should reference both results
        await result.expect.next_event(type="message").judge(
            llm,
            intent="Provides information about both the user balance and the weather in Tokyo.",
        )


@pytest.mark.asyncio
async def test_tool_call_context_retention(test_llm):
    """
    Test that context is retained across multiple turns with tools.

    Verifies:
    - Tool results are remembered
    - Agent can reference previous tool outputs
    """
    agent = TestAgentWithTools()

    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent._agent)

        # First turn - use tool
        result1 = await session.run(user_input="What's the weather in Tokyo?")

        result1.expect.next_event().is_function_call(name="lookup_weather")
        result1.expect.next_event().is_function_call_output()

        await (
            result1.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="Provides weather information for Tokyo.")
        )

        # Second turn - reference previous tool result
        result2 = await session.run(user_input="What was that temperature again?")

        # Should remember from previous turn
        await (
            result2.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Remembers and states the temperature from the previous Tokyo weather query (70 degrees).",
            )
        )
