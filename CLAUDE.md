# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

You are an expert Python backend engineer working with LiveKit. You specialize in real-time communication systems, and you must write clean, production-ready code. This is a **PTE (Pearson Test of English) Interview Practice Agent** built on LiveKit Agents framework. The agent provides voice-based interview practice sessions with structured assessment, behavioral analysis, and real-time feedback. It's a customized implementation based on the LiveKit Agents Python starter template.

## Key Technologies

- **LiveKit Agents**: Core framework for real-time voice AI
- **OpenAI GPT-5-nano**: LLM for interview conversation
- **Deepgram**: STT (Speech-to-Text) and TTS (Text-to-Speech)
- **Langfuse**: Observability and tracing via OpenTelemetry
- **Whispey**: Voice analytics integration
- **MongoDB**: Data persistence layer
- **UV**: Fast Python package manager (replaces pip)

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Download required models (Silero VAD, turn detector)
uv run python src/agent.py download-files
```

### Running the Agent
```bash
# Dev mode - for use with frontend or telephony
uv run python src/agent.py dev

```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_agent.py
```

### Code Quality
```bash
# Format and lint with ruff
uv run ruff format .
uv run ruff check .
```

## Architecture Overview

### Main Entry Point (`src/agent.py`)

The entrypoint function orchestrates:
1. **Langfuse tracing setup** - OpenTelemetry spans for observability
2. **Room connection** - LiveKit room with audio/video streams
3. **Agent session** - Configured with VAD, noise cancellation, transcription
4. **Whispey analytics** - Voice analytics tracking and export

### Core Agent: `PTEInterviewAgent`

Custom agent class inheriting from `Agent` that implements:

- **Structured Output Processing**: Uses `InterviewTurnJSON` TypedDict for structured LLM responses containing:
  - `voice_instructions`: TTS directives for tone/pace
  - `system_response`: The spoken response
  - `internal_assessment`: Private evaluation
  - `interview_stage`: Current interview phase (document_check, background_inquiry, academic_assessment, etc.)
  - `credibility_score`: Running assessment (1-10)
  - `red_flags`: Identified concerns

- **Custom Pipeline Nodes**:
  - `llm_node()`: Overridden to support structured output via OpenAI's response_format
  - `tts_node()`: Smart text processing that extracts system_response from structured JSON
  - `transcription_node()`: Handles structured output in transcriptions

- **Non-blocking Strucutred Data Storage**: Background writer pattern using asyncio.Queue to store interview data without blocking the main conversation flow

- **Interview State Management**: Tracks stages, questions, responses, observations, and behavioral analysis

### Utilities (`src/interview_agent/utilities/`)

- **`structured_output.py`**: Core logic for detecting and processing structured vs plain text streams. Uses `pydantic_core.from_json` with `allow_partial="trailing-strings"` for incremental JSON parsing

- **`prompt_loader.py`**: Loads interview prompts from `src/interview_agent/prompts/*.yaml` files

- **`setup_langfuse.py`**: Configures OpenTelemetry trace provider with Langfuse OTLP exporter

- **`prewarm.py`**: Preloads Silero VAD model during worker initialization (reduces cold start latency)

- **`user_details.py`**: Provides default user configuration

- **`aws_egress.py`**: S3 recording integration (currently commented out in main agent)

### MongoDB Handler (`src/interview_agent/mongodb_handler/`)

Custom database handler for MongoDB with SQL-like query translation:
- Parses SQL AST using `mindsdb-sql-parser`
- Translates to MongoDB operations (find, aggregate, insert, etc.)
- Supports flattening nested documents
- Used for storing/querying interview data persistence

## Important Implementation Details

### Structured Output Flow

1. LLM generates JSON with `InterviewTurnJSON` schema
2. `process_structured_output()` detects JSON vs plain text via `detect_output_mode()`
3. `process_structured_json()` incrementally parses and yields `system_response` deltas
4. TTS applies `voice_instructions` before generating audio
5. Full structured data stored asynchronously via `JSONStorage`

### Background JSON Storage Pattern

Prevents blocking conversation flow:
- Write operations queued to `asyncio.Queue`
- Background writer task processes queue continuously
- Graceful shutdown flushes remaining items
- Storage organized as `interview_data/{user_id}/interview_{session_id}_{timestamp}.json`

### VAD and Turn Detection

- **Silero VAD**: Preloaded in `prewarm_fnc` for fast startup
- **MultilingualModel**: LiveKit turn detector for contextually-aware speaker detection

### Tracing and Observability

All major operations wrapped in OpenTelemetry spans:
- Set user attributes: `langfuse.user.id`, `langfuse.session.id`
- Track input/output for LLM, TTS, state changes
- Force flush on shutdown via callback

## Environment Variables

Required in `.env.local`:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY` (if using Cartesia TTS)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `WHISPEY_API_KEY`

## Deployment

Production-ready Docker configuration:
```bash
# Build image
docker build -t interview-agent .

# Run container
docker run --env-file .env.local interview-agent
```

The Dockerfile:
- Uses UV for fast dependency installation
- Pre-downloads models via `download-files` command
- Runs as non-root user for security
- Executes `uv run src/agent.py start` by default

## Key Files to Understand

1. **`src/agent.py`**: Main entrypoint and PTEInterviewAgent class
2. **`src/interview_agent/utilities/structured_output.py`**: Structured output processing logic
3. **`src/interview_agent/prompts/pte_interview.yaml`**: Interview instructions/prompts
4. **`pyproject.toml`**: Dependencies and project configuration
5. **`taskfile.yaml`**: Task runner configuration for common commands

## Common Patterns

### Adding New Interview Stages

1. Update `InterviewTurnJSON` Literal type in `structured_output.py`
2. Modify prompt in `src/interview_agent/prompts/pte_interview.yaml`
3. Agent will automatically handle new stage in structured responses

### Customizing Voice Behavior

LLM can specify `voice_instructions` in structured output to control TTS tone/pace dynamically per response.

### Extending with Function Tools

Use `@function_tool()` decorator from `livekit.agents` to add callable functions to the agent's context (see README example with `search_knowledge_base`).

## Testing Notes

- Tests use pytest with asyncio support
- Eval framework based on LiveKit Agents testing/evaluation
- Must provide API keys as GitHub secrets for CI (currently uses OpenAI)
