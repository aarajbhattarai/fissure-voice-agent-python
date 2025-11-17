# Testing Guide

Complete guide for testing LiveKit agents with pytest.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Structure](#test-structure)
3. [Writing Tests](#writing-tests)
4. [Running Tests](#running-tests)
5. [Schema Generation](#schema-generation)
6. [Test Categories](#test-categories)

---

## Quick Start

### Installation

Install test dependencies:

```bash
uv sync  # Installs all dependencies including test tools
```

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test File

```bash
uv run pytest tests/test_pte_interview_agent.py
```

### Run with Verbose Output

```bash
LIVEKIT_EVALS_VERBOSE=1 uv run pytest -s -o log_cli=true
```

---

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_agent.py                  # Original assistant agent tests
├── test_pte_interview_agent.py    # PTE Interview Agent tests
├── test_base_agent.py             # BaseAgent tests
├── test_routing.py                # Multi-agent routing tests
└── test_tools.py                  # Function tool tests
```

---

## Writing Tests

### Basic Test Template

```python
import pytest
from livekit.agents import AgentSession
from livekit.plugins import openai


@pytest.mark.asyncio
async def test_your_agent(test_llm):
    """Test description."""
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        # Start the agent
        await session.start(YourAgent())

        # Run a conversation turn
        result = await session.run(user_input="Hello")

        # Assert expected behavior
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="Makes a friendly greeting.",
            )
        )

        # Verify no unexpected events
        result.expect.no_more_events()
```

---

## Testing on_enter Modes

The BaseAgent supports three `on_enter` modes for session initialization:

### 1. Static Mode

Uses `session.say()` with a predefined message:

```python
@pytest.mark.asyncio
async def test_static_on_enter(test_llm):
    """Test static welcome message."""
    config = {
        "on_enter_mode": "static",
        "on_enter_config": {
            "static_message": "Welcome! How can I help?"
        }
    }

    # Agent would call: await session.say(text=static_message)
```

**Characteristics**:
- Static text from configuration
- No LLM call needed
- Message not added to chat history
- Fast and cost-effective

### 2. Greeting Mode

Uses `session.generate_reply(instructions=...)`:

```python
@pytest.mark.asyncio
async def test_greeting_on_enter(test_llm):
    """Test dynamic greeting generation."""
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent)

        # Generate greeting using instructions
        await session.generate_reply(
            instructions="Greet the user warmly and ask how you can help."
        )

        # Instructions are NOT in chat history
        # Only the generated greeting is added
```

**Characteristics**:
- LLM generates greeting based on instructions
- Instructions guide response but aren't stored in history
- Generated response IS added to history
- Dynamic and personalized

### 3. Direct Mode

Uses `session.generate_reply(user_input=...)`:

```python
@pytest.mark.asyncio
async def test_direct_on_enter(test_llm):
    """Test direct user input simulation."""
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent)

        # Simulate user starting conversation
        await session.generate_reply(
            user_input="Hello, I need help with my account"
        )

        # user_input IS added to chat history
        # Agent generates response based on it
```

**Characteristics**:
- User input added directly to chat history
- Agent responds as if user spoke first
- Both user input and response in history
- Good for testing specific conversation starts

---

## Test Assertions

### Message Assertions

```python
# Check for message
result.expect.next_event().is_message(role="assistant")

# Access message content
message = result.expect.next_event().is_message(role="assistant").event().item.content
```

### LLM-Based Judgment

```python
# Qualitative evaluation
await (
    result.expect.next_event()
    .is_message(role="assistant")
    .judge(
        llm,
        intent="Provides helpful information about the topic.",
    )
)
```

### Function Call Assertions

```python
# Verify function call
result.expect.next_event().is_function_call(
    name="get_weather",
    arguments={"location": "Tokyo"}
)

# Verify function output
result.expect.next_event().is_function_call_output(
    output="sunny with a temperature of 70 degrees."
)
```

### Navigation

```python
# Sequential navigation
result.expect.next_event().is_message(role="assistant")
result.expect.next_event().is_function_call(name="tool")

# Skip events
result.expect.skip_next()  # Skip one
result.expect.skip_next(2)  # Skip two

# Indexed access
result.expect[0].is_message(role="assistant")

# Search (order-agnostic)
result.expect.contains_message(role="assistant")
result.expect.contains_function_call(name="tool")
```

---

## Multi-Turn Conversations

Test conversation history and context retention:

```python
@pytest.mark.asyncio
async def test_multi_turn(test_llm):
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent)

        # Turn 1
        result1 = await session.run(user_input="My name is Alice")
        await result1.expect.next_event().is_message(role="assistant").judge(
            llm, intent="Acknowledges the name Alice"
        )

        # Turn 2 - test context retention
        result2 = await session.run(user_input="What's my name?")
        await result2.expect.next_event().is_message(role="assistant").judge(
            llm, intent="Correctly remembers the name is Alice"
        )
```

---

## Mocking Tools

Test edge cases by mocking function tools:

```python
from livekit.agents import mock_tools

@pytest.mark.asyncio
async def test_tool_error_handling(test_llm):
    async with (
        test_llm as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(agent)

        # Mock tool to raise error
        with mock_tools(
            YourAgent,
            {"get_weather": lambda: RuntimeError("Service unavailable")}
        ):
            result = await session.run(user_input="What's the weather?")

            await result.expect.next_event(type="message").judge(
                llm,
                intent="Informs user that weather service is unavailable.",
            )
```

### Complex Mock Functions

```python
def _mock_weather(location: str) -> str:
    if location == "Tokyo":
        return "sunny with a temperature of 70 degrees."
    elif location == "Paris":
        return "rainy with a temperature of 55 degrees."
    else:
        return "UNSUPPORTED_LOCATION"

with mock_tools(Agent, {"lookup_weather": _mock_weather}):
    # Test with mock
    pass
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest

# Run specific file
uv run pytest tests/test_routing.py

# Run specific test
uv run pytest tests/test_routing.py::test_explicit_agent_id_routing

# Run tests matching pattern
uv run pytest -k "routing"
```

### With Verbose Output

```bash
# LiveKit verbose eval output
LIVEKIT_EVALS_VERBOSE=1 uv run pytest -s -o log_cli=true

# Pytest verbose
uv run pytest -v

# Both
LIVEKIT_EVALS_VERBOSE=1 uv run pytest -v -s
```

### Using Test Markers

```bash
# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```

---

## Schema Generation

Generate Pydantic models from JSON Schema using datamodel-code-generator.

### Quick Start

```bash
# Generate all models
bash scripts/generate_models.sh
```

### Manual Generation

```bash
uv run datamodel-codegen \
  --input schemas/interview_turn.json \
  --output src/interview_agent/models/interview_turn.py \
  --output-model-type pydantic_v2.BaseModel \
  --field-constraints \
  --use-default \
  --target-python-version 3.11
```

### Using Generated Models

```python
from interview_agent.models import InterviewTurn, SupportTurn, SalesTurn

# Create instance
turn = InterviewTurn(
    system_response="Hello! How can I assist you today?",
    interview_stage="background_inquiry",
    credibility_score=8
)

# Validate data
turn.model_validate(data_dict)

# Export
turn.model_dump()
turn.model_dump_json()
```

### Available Schemas

- `schemas/interview_turn.json` → `InterviewTurn`
- `schemas/support_turn.json` → `SupportTurn`
- `schemas/sales_turn.json` → `SalesTurn`

---

## Test Categories

### Unit Tests

Test individual components in isolation:

```bash
uv run pytest -m unit
```

Examples:
- `test_routing.py` - Routing logic
- `test_user_data_management` - Data classes

### Integration Tests

Test agent behavior end-to-end:

```bash
uv run pytest -m integration
```

Examples:
- `test_pte_interview_agent.py` - Full agent tests
- `test_multi_turn_conversation` - Multi-turn flows

### Tool Tests

Test function tool integration:

```bash
uv run pytest -m tools
```

Examples:
- `test_tools.py` - Tool mocking and validation

---

## Best Practices

### 1. Use Fixtures

Define reusable test components in `conftest.py`:

```python
@pytest.fixture
def mock_user_details():
    return {
        "user_id": "test-123",
        "name": "Test User"
    }
```

### 2. Descriptive Test Names

```python
# Good
def test_agent_remembers_user_name_across_turns()

# Bad
def test_memory()
```

### 3. Clear Intent Strings

```python
# Good
await result.expect.next_event().is_message(role="assistant").judge(
    llm,
    intent="Acknowledges the user's name (Alice) and asks a follow-up question."
)

# Too vague
await result.expect.next_event().is_message(role="assistant").judge(
    llm,
    intent="Responds appropriately."
)
```

### 4. Test Edge Cases

```python
# Test normal case
test_valid_input()

# Test edge cases
test_empty_input()
test_very_long_input()
test_invalid_format()
test_special_characters()
```

### 5. Mock External Dependencies

Always mock:
- API calls
- Database queries
- File I/O
- Network requests

---

## CI/CD Integration

### GitHub Actions

Set API keys as secrets in repository settings:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: uv run pytest
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

```bash
# Make sure to install dependencies
uv sync
```

#### 2. API key errors

```bash
# Check environment variables
echo $OPENAI_API_KEY

# Or use .env.local
cp .env.example .env.local
# Edit .env.local with your keys
```

#### 3. Async test failures

Make sure to use `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio  # Required!
async def test_async_function():
    ...
```

#### 4. Judgment failures

If LLM judgments are inconsistent:
- Make intent strings more specific
- Provide more context in intent
- Use stricter evaluation criteria

---

## Additional Resources

- [LiveKit Testing Docs](https://docs.livekit.io/agents/build/testing)
- [pytest Documentation](https://docs.pytest.org/)
- [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator)
- [Pydantic V2 Docs](https://docs.pydantic.dev/latest/)

---

## Example Test Suite

See complete examples in:
- `tests/test_pte_interview_agent.py` - PTEInterviewAgent tests
- `tests/test_base_agent.py` - BaseAgent with on_enter modes
- `tests/test_routing.py` - Multi-agent routing
- `tests/test_tools.py` - Function tool mocking
- `tests/test_agent.py` - Original example tests
