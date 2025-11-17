# User Data Management & Conversation Summaries

## Overview

This document describes the user data management system and conversation summarization features added to the dynamic agent architecture.

---

## Table of Contents

1. [User Data Management](#user-data-management)
2. [Session Data](#session-data)
3. [Conversation History Tracking](#conversation-history-tracking)
4. [Conversation Summaries](#conversation-summaries)
5. [Storage & Retrieval](#storage--retrieval)
6. [Integration with Agents](#integration-with-agents)
7. [API Reference](#api-reference)
8. [Examples](#examples)

---

## User Data Management

### Overview

User data is managed using type-safe data classes that provide structured storage for user information throughout the session.

### Data Classes

#### 1. **UserData** (Dataclass - Fast)

```python
from agents.user_data import UserData

user_data = UserData(
    user_id="user-12345",
    name="John Doe",
    email="john.doe@example.com",
    age=25,
    nationality="USA",
    education_level="Bachelor's Degree",
    institution="MIT",
    field_of_study="Computer Science",
    language_preference="en",
)
```

**When to use**: When you need performance and don't require runtime validation.

#### 2. **UserDetailsModel** (Pydantic - Validated)

```python
from agents.user_data import UserDetailsModel

user_data = UserDetailsModel(
    user_id="user-12345",
    name="John Doe",
    email="john.doe@example.com",  # Validated email format
    age=25,  # Validated: 0 <= age <= 150
    language_preference="en",  # Validated: must be in allowed list
)
```

**When to use**: When you need input validation and API serialization.

### Available Fields

| Field | Type | Description | Optional |
|-------|------|-------------|----------|
| `user_id` | str | Unique user identifier | No |
| `name` | str | User's full name | Yes |
| `email` | str | User's email address | Yes |
| `phone` | str | User's phone number | Yes |
| `age` | int | User's age (0-150) | Yes |
| `nationality` | str | User's nationality | Yes |
| `education_level` | str | Highest education level | Yes |
| `institution` | str | Educational institution | Yes |
| `field_of_study` | str | Field of study | Yes |
| `language_preference` | str | Preferred language code | Yes (default: "en") |
| `timezone` | str | User's timezone | Yes |
| `custom_fields` | dict | Custom user data | Yes |

### Operations

```python
# Update user data
user_data.update(phone="+1-555-0123", age=26)

# Convert to dictionary
user_dict = user_data.to_dict()

# Create from dictionary
user_data = UserData.from_dict(user_dict)
```

---

## Session Data

### Overview

`SessionData` encapsulates all session-level information including user data, timestamps, and session metadata.

### Creating Session Data

```python
from agents.user_data import SessionData, UserData

user_data = UserData(user_id="user-123", name="John Doe")

session_data = SessionData(
    session_id="session-abc",
    user_data=user_data,
    tenant_id="customer-xyz",
    agent_id="interview-agent-v1",
)
```

### Session Lifecycle

```python
# Session starts automatically when created
print(f"Started at: {session_data.started_at}")
print(f"Status: {session_data.status}")  # "active"

# End session
session_data.end_session(status="completed")

print(f"Ended at: {session_data.ended_at}")
print(f"Duration: {session_data.duration_seconds}s")
```

### Session Status Values

- `active` - Session in progress
- `completed` - Session ended normally
- `aborted` - Session terminated by user
- `error` - Session ended due to error

---

## Conversation History Tracking

### Overview

`ConversationHistory` automatically tracks all conversation turns (user and agent messages) during a session.

### Creating Conversation History

```python
from agents.user_data import ConversationHistory

conversation = ConversationHistory(
    session_id="session-abc",
    user_id="user-123"
)
```

### Adding Conversation Turns

```python
# Agent message
conversation.add_turn(
    speaker="agent",
    message="Welcome! Let's begin your interview practice."
)

# User message
conversation.add_turn(
    speaker="user",
    message="Thank you! I'm ready."
)

# With structured data (from LLM response)
conversation.add_turn(
    speaker="agent",
    message="Tell me about your background.",
    structured_data={
        "interview_stage": "background_inquiry",
        "credibility_score": 8
    }
)
```

### Retrieving Conversation Data

```python
# Get all user messages
user_messages = conversation.get_user_messages()

# Get all agent messages
agent_messages = conversation.get_agent_messages()

# Get full transcript
transcript = conversation.get_full_transcript()
# Output:
# AGENT: Welcome! Let's begin your interview practice.
# USER: Thank you! I'm ready.
# AGENT: Tell me about your background.

# Get conversation stats
print(f"Total turns: {conversation.total_turns}")
```

---

## Conversation Summaries

### Overview

The `ConversationSummarizer` uses LLMs to generate intelligent summaries of conversations after sessions end.

### Creating a Summarizer

```python
from agents.conversation_summary import ConversationSummarizer

summarizer = ConversationSummarizer(
    llm_provider="openai",
    model="gpt-5-nano",
    # api_key=...,  # Optional, uses env var by default
)
```

### Generating Summaries

#### Full Summary

```python
summary_data = await summarizer.generate_summary(
    transcript=conversation.get_full_transcript(),
    user_data={
        "name": "John Doe",
        "institution": "MIT",
        "field_of_study": "Computer Science"
    },
    session_metadata={
        "session_id": "session-abc",
        "total_turns": 12,
        "duration_seconds": 180
    }
)

print(summary_data["summary"])  # Full detailed summary
print(summary_data["structured_summary"])  # Parsed sections
print(summary_data["tokens_used"])  # LLM tokens used
```

#### Quick Summary

```python
quick_summary = await summarizer.generate_quick_summary(
    transcript=conversation.get_full_transcript(),
    max_length=200
)

print(quick_summary)  # Brief 2-3 sentence summary
```

### Summary Structure

Generated summaries include:

1. **Overview** - Brief description of conversation
2. **Key Topics Discussed** - Main topics covered
3. **User Insights** - User's goals, concerns, background
4. **Agent Performance** - How well agent addressed needs
5. **Outcomes** - Decisions made, actions agreed upon
6. **Key Quotes** - Notable statements
7. **Recommendations** - Suggestions for follow-up

### Custom Summary Prompts

```python
custom_prompt = """
{context}

CONVERSATION:
{transcript}

Provide a summary focusing on:
1. Technical skills demonstrated
2. Communication effectiveness
3. Areas for improvement
"""

summary_data = await summarizer.generate_summary(
    transcript=transcript,
    user_data=user_data,
    custom_prompt=custom_prompt
)
```

---

## Storage & Retrieval

### Overview

`SessionStorage` provides MongoDB-backed persistence for sessions, conversations, and summaries.

### Creating Storage

```python
from agents.session_storage import create_session_storage

storage = await create_session_storage(
    connection_string="mongodb://localhost:27017",
    database="agent_data"
)
```

### Storing Data

#### Save Session

```python
# Save complete session data
doc_id = await storage.save_session(session_data)
```

#### Save Conversation

```python
# Save conversation history
doc_id = await storage.save_conversation(conversation_history)
```

#### Save Summary

```python
# Save conversation summary
doc_id = await storage.save_summary(
    session_id="session-abc",
    user_id="user-123",
    summary_data=summary_data,
    conversation_history=conversation_history.to_dict()
)
```

### Retrieving Data

#### Get Session

```python
session = await storage.get_session("session-abc")
```

#### Get Conversation

```python
conversation = await storage.get_conversation("session-abc")
```

#### Get Summary

```python
summary = await storage.get_summary("session-abc")
```

#### Get User's History

```python
# Get last 10 sessions for user
sessions = await storage.get_user_sessions("user-123", limit=10)

# Get last 10 conversations
conversations = await storage.get_user_conversations("user-123", limit=10)

# Get last 10 summaries
summaries = await storage.get_user_summaries("user-123", limit=10)
```

### Analytics

```python
# Get session statistics
stats = await storage.get_session_stats(user_id="user-123")

print(f"Total sessions: {stats['total_sessions']}")
print(f"Avg duration: {stats['avg_duration_seconds']}s")
print(f"Total turns: {stats['total_turns']}")
```

### Search Conversations

```python
# Search conversations by text
results = await storage.search_conversations(
    search_text="visa interview",
    user_id="user-123",
    limit=10
)
```

---

## Integration with Agents

### Automatic Integration

The `BaseAgent` class automatically integrates all these features:

#### 1. User Data

```python
# Agent automatically creates UserData from user_details
agent = BaseAgent(
    ...,
    user_details={
        "user_id": "user-123",
        "name": "John Doe",
        "email": "john@example.com",
        "institution": "MIT"
    }
)

# Access user data
print(agent.user_data.name)
print(agent.user_data.institution)

# Update user data
agent.user_data.update(phone="+1-555-0123")
```

#### 2. Session Data

```python
# Agent automatically creates SessionData
print(f"Session ID: {agent.session_data.session_id}")
print(f"Status: {agent.session_data.status}")
print(f"Duration: {agent.session_data.duration_seconds}")
```

#### 3. Conversation Tracking

```python
# Agent automatically tracks all conversations
print(f"Total turns: {agent.conversation_history.total_turns}")
print(agent.conversation_history.get_full_transcript())
```

#### 4. Summary Generation

```python
# Agent automatically generates summary on close()
await agent.close()

# Summary is available in conversation_history
print(agent.conversation_history.summary)
```

### Configuring Summary Generation

In agent configuration:

```python
{
    "agent_id": "my-agent",
    # ... other config

    "summary_config": {
        "enabled": True,  # Enable/disable summaries
        "model": "gpt-5-nano",  # LLM model for summaries
    }
}
```

### Session Manager Integration

```python
from orchestrator import SessionManager
from agents.session_storage import create_session_storage

# Create storage
storage = await create_session_storage(
    connection_string="mongodb://localhost:27017",
    database="agent_data"
)

# Create session manager with storage
session_manager = SessionManager(
    agent_factory=factory,
    config_store=config_store,
    session_storage=storage,  # Add storage
)

# When session ends, automatically:
# 1. Generates summary
# 2. Saves session data
# 3. Saves conversation history
# 4. Saves summary
await session_manager.end_session(session_id)
```

---

## API Reference

### UserData

**Methods:**
- `to_dict() -> dict` - Convert to dictionary
- `update(**kwargs)` - Update fields
- `from_dict(data: dict) -> UserData` - Create from dictionary

### SessionData

**Methods:**
- `to_dict() -> dict` - Convert to dictionary
- `end_session(status: str, error: Optional[str])` - Mark session ended
- `from_dict(data: dict) -> SessionData` - Create from dictionary

**Properties:**
- `session_id: str` - Session identifier
- `user_data: UserData` - User information
- `tenant_id: str` - Tenant identifier
- `agent_id: str` - Agent identifier
- `started_at: datetime` - Session start time
- `ended_at: Optional[datetime]` - Session end time
- `duration_seconds: Optional[float]` - Session duration
- `status: str` - Session status

### ConversationHistory

**Methods:**
- `add_turn(speaker, message, structured_data, audio_duration_ms)` - Add conversation turn
- `get_user_messages() -> list[str]` - Get all user messages
- `get_agent_messages() -> list[str]` - Get all agent messages
- `get_full_transcript() -> str` - Get full transcript
- `set_summary(summary: str)` - Set summary
- `to_dict() -> dict` - Convert to dictionary

**Properties:**
- `session_id: str` - Session identifier
- `user_id: str` - User identifier
- `turns: list[ConversationTurn]` - All conversation turns
- `summary: Optional[str]` - Conversation summary
- `total_turns: int` - Total number of turns

### ConversationSummarizer

**Methods:**
- `generate_summary(transcript, user_data, session_metadata, custom_prompt) -> dict` - Generate full summary
- `generate_quick_summary(transcript, max_length) -> str` - Generate brief summary
- `generate_structured_summary(transcript, schema) -> dict` - Generate summary with custom schema

### SessionStorage

**Methods:**
- `save_session(session_data) -> str` - Save session
- `update_session(session_id, updates) -> bool` - Update session
- `get_session(session_id) -> Optional[dict]` - Get session
- `get_user_sessions(user_id, limit) -> list[dict]` - Get user's sessions
- `save_conversation(conversation_history) -> str` - Save conversation
- `get_conversation(session_id) -> Optional[dict]` - Get conversation
- `get_user_conversations(user_id, limit) -> list[dict]` - Get user's conversations
- `save_summary(session_id, user_id, summary_data, conversation_history) -> str` - Save summary
- `get_summary(session_id) -> Optional[dict]` - Get summary
- `get_user_summaries(user_id, limit) -> list[dict]` - Get user's summaries
- `get_session_stats(user_id) -> dict` - Get statistics
- `search_conversations(search_text, user_id, limit) -> list[dict]` - Search conversations

---

## Examples

### Example 1: Basic User Data

```python
from agents.user_data import UserData

# Create user data
user = UserData(
    user_id="user-123",
    name="John Doe",
    email="john@example.com",
    institution="MIT"
)

# Update
user.update(phone="+1-555-0123")

# Convert
user_dict = user.to_dict()
```

### Example 2: Track Conversation

```python
from agents.user_data import ConversationHistory

conversation = ConversationHistory(
    session_id="session-abc",
    user_id="user-123"
)

# Add turns
conversation.add_turn(speaker="agent", message="Hello!")
conversation.add_turn(speaker="user", message="Hi!")

# Get transcript
print(conversation.get_full_transcript())
```

### Example 3: Generate Summary

```python
from agents.conversation_summary import ConversationSummarizer

summarizer = ConversationSummarizer()

summary_data = await summarizer.generate_summary(
    transcript=conversation.get_full_transcript(),
    user_data={"name": "John Doe"}
)

print(summary_data["summary"])
```

### Example 4: Store and Retrieve

```python
from agents.session_storage import create_session_storage

# Create storage
storage = await create_session_storage(
    "mongodb://localhost:27017",
    "agent_data"
)

# Save
await storage.save_session(session_data)
await storage.save_conversation(conversation)

# Retrieve
session = await storage.get_session("session-abc")
conversation = await storage.get_conversation("session-abc")
```

### Example 5: Complete Integration

```python
# Agent automatically handles everything
agent = BaseAgent(
    room=room,
    user_id="user-123",
    session_id="session-abc",
    user_details={
        "name": "John Doe",
        "email": "john@example.com",
        "institution": "MIT"
    },
    config=config,
    ...
)

# Conversation is tracked automatically
# Summary is generated on close
await agent.close()

# Access results
print(agent.conversation_history.summary)
print(f"Total turns: {agent.conversation_history.total_turns}")
```

---

## Frontend Integration

### Receiving Real-time Data

The agent sends data to frontend via LiveKit RPC topics:

#### 1. Conversation Summary

```javascript
// Listen for summary
room.on('dataReceived', (payload, participant, kind, topic) => {
  if (topic === 'conversation-summary') {
    const data = JSON.parse(payload);
    console.log('Summary:', data.summary);
    console.log('Structured:', data.structured_summary);
  }
});
```

#### 2. Session Complete

```javascript
// Listen for session complete event
room.on('dataReceived', (payload, participant, kind, topic) => {
  if (topic === 'session-complete') {
    const data = JSON.parse(payload);
    console.log('Session ended');
    console.log('Total turns:', data.total_turns);
    console.log('Duration:', data.duration_seconds);
    console.log('Summary available:', data.summary_available);
  }
});
```

### API Endpoints

Create API endpoints to retrieve stored data:

```python
# api/sessions.py
from fastapi import APIRouter
from agents.session_storage import SessionStorage

router = APIRouter()

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    storage = get_storage()  # Get storage instance
    session = await storage.get_session(session_id)
    return session

@router.get("/sessions/{session_id}/summary")
async def get_summary(session_id: str):
    storage = get_storage()
    summary = await storage.get_summary(session_id)
    return summary

@router.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 10):
    storage = get_storage()
    sessions = await storage.get_user_sessions(user_id, limit)
    return {"sessions": sessions}
```

---

## Best Practices

### 1. User Data

- ✅ Use `UserData` for performance-critical paths
- ✅ Use `UserDetailsModel` for API input validation
- ✅ Store custom fields in `custom_fields` dict
- ✅ Update `user_data` as you learn more about the user

### 2. Conversation Tracking

- ✅ Let `BaseAgent` handle tracking automatically
- ✅ Access conversation history for context-aware responses
- ✅ Use structured_data in turns for rich metadata

### 3. Summaries

- ✅ Enable summaries for important sessions
- ✅ Disable for high-volume, low-value sessions
- ✅ Use quick summaries for real-time display
- ✅ Use full summaries for analysis and reporting

### 4. Storage

- ✅ Always use `SessionStorage` in production
- ✅ Create indexes for fast queries
- ✅ Implement data retention policies
- ✅ Back up MongoDB regularly

### 5. Performance

- ✅ Summary generation is async and non-blocking
- ✅ Storage operations happen after agent closes
- ✅ Use background tasks for heavy operations
- ✅ Cache frequently accessed data

---

## Troubleshooting

### Issue: Summary not generated

**Possible causes:**
1. `summary_config.enabled` is `false` in agent config
2. OpenAI API key not set
3. Conversation too short (< 2 turns)

**Solution:**
```python
# Check config
config["summary_config"]["enabled"] = True

# Verify API key
import os
assert os.getenv("OPENAI_API_KEY"), "API key not set"
```

### Issue: Storage failures

**Possible causes:**
1. MongoDB not running
2. Connection string incorrect
3. Database permissions

**Solution:**
```bash
# Check MongoDB
docker ps | grep mongo

# Test connection
mongo mongodb://localhost:27017

# Check logs
docker logs mongodb
```

### Issue: User data not persisting

**Possible causes:**
1. `SessionStorage` not configured in `SessionManager`
2. Session not ending properly

**Solution:**
```python
# Ensure storage is configured
session_manager = SessionManager(
    ...,
    session_storage=storage  # Must be set
)

# Ensure session ends
await session_manager.end_session(session_id)
```

---

## Next Steps

1. **Run Examples** - Try `examples/user_data_session_example.py`
2. **Configure Storage** - Set up MongoDB and SessionStorage
3. **Customize Summaries** - Create custom summary prompts
4. **Build Analytics** - Use stored data for insights
5. **Extend User Data** - Add custom fields for your use case

---

## Resources

- [Implementation Examples](./examples/user_data_session_example.py)
- [Dynamic Agent Architecture](./DYNAMIC_AGENT_ARCHITECTURE.md)
- [API Documentation](./src/api/admin_api.py)
- [Best Practices](./BEST_PRACTICES.md)
