# Architecture Overview: Configuration vs Worker Execution

## The Confusion

You're seeing two different processes and wondering how they connect:

1. **Admin API**: `POST /api/v1/agents/config` → calls `create_agent_config()`
2. **Worker Process**: `cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, ...))`

**Key Point**: These are **two separate processes** that serve different purposes!

---

## Two-Process Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYER                          │
│  (Optional - Only needed if using dynamic agent system)         │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   Admin API          │  ← FastAPI server for config management
    │  (admin_api.py)      │
    │   Port 8000          │
    └──────────┬───────────┘
               │
               │ CRUD operations
               ↓
    ┌──────────────────────┐
    │     MongoDB          │  ← Stores agent configurations
    │  (agent_configs)     │     (prompts, schemas, LLM settings)
    └──────────┬───────────┘
               ↑
               │ Reads configs
               │
┌──────────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                              │
│           (This is what actually runs the agents)                │
└──────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  LiveKit Worker      │  ← Agent worker process
    │  (multi_agent_       │     Started via cli.run_app()
    │   worker.py)         │
    └──────────┬───────────┘
               │
               │ Handles voice sessions
               ↓
    ┌──────────────────────┐
    │  LiveKit Server      │  ← WebRTC infrastructure
    │  (Cloud/Self-hosted) │
    └──────────┬───────────┘
               │
               │ WebRTC
               ↓
    ┌──────────────────────┐
    │    Frontend          │  ← User interface
    │  (React/Next.js)     │
    └──────────────────────┘
```

---

## Process 1: Admin API (Configuration Management)

**Purpose**: Manage agent configurations (CRUD operations)

**File**: `src/api/admin_api.py`

**How to Run**:
```bash
uvicorn src.api.admin_api:app --reload
```

**What It Does**:
- Creates/updates/deletes agent configurations in MongoDB
- Does **NOT** run any agents
- Does **NOT** handle voice sessions
- Just stores configuration data

**Example**:
```python
# This ONLY saves configuration to MongoDB
# It does NOT start any agent worker
POST /api/v1/agents/config
{
  "agent_id": "support-agent-v1",
  "prompt_config": { ... },
  "schema_config": { ... }
}
```

**Analogy**: Think of this as **writing a recipe book**. You're just saving recipes (agent configs), not cooking anything.

---

## Process 2: LiveKit Worker (Agent Execution)

**Purpose**: Actually run agents and handle voice sessions

**File**: `examples/multi_agent_worker.py` or `src/agent.py`

**How to Run**:
```bash
# Original PTE Interview Agent
uv run python src/agent.py dev

# Or multi-agent worker
uv run python examples/multi_agent_worker.py dev
```

**What It Does**:
- Starts a LiveKit worker process
- Waits for users to join rooms
- When user joins, reads configuration from MongoDB
- Creates agent instance and handles voice conversation
- Runs continuously, handling multiple sessions concurrently

**Analogy**: This is **the chef in the kitchen**. It reads recipes from the recipe book (MongoDB) and cooks (runs agents) when orders come in (users join).

---

## How They Work Together

### Flow Diagram

```
Step 1: Create Configuration (One-time setup)
┌─────────────┐
│  Developer  │
└──────┬──────┘
       │
       │ 1. Create config via API or script
       ↓
┌──────────────────────┐
│  Admin API           │
│  POST /api/v1/       │──→ 2. Save to MongoDB
│  agents/config       │
└──────────────────────┘
       │
       ↓
┌──────────────────────┐
│    MongoDB           │
│  {agent configs}     │
└──────────────────────┘


Step 2: Start Worker (Always running)
┌──────────────────────┐
│  Developer           │
│  $ uv run python     │
│    multi_agent_      │
│    worker.py dev     │
└──────┬───────────────┘
       │
       │ 3. Start worker process
       ↓
┌──────────────────────┐
│  LiveKit Worker      │
│  (cli.run_app)       │◄──── Worker is now waiting for users
│  Status: Running     │
└──────────────────────┘


Step 3: User Session (Happens when user joins)
┌──────────────────────┐
│   Frontend           │
│   User clicks        │
│   "Start Interview"  │
└──────┬───────────────┘
       │
       │ 4. Join room with metadata
       │    { purpose: 'support' }
       ↓
┌──────────────────────┐
│  LiveKit Server      │
│                      │
└──────┬───────────────┘
       │
       │ 5. Trigger entrypoint()
       ↓
┌──────────────────────┐
│  Worker: entrypoint()│
│  - Read metadata     │
│  - Determine agent   │
└──────┬───────────────┘
       │
       │ 6. Load config from MongoDB
       ↓
┌──────────────────────┐
│    MongoDB           │
│  Returns config for  │
│  'support-agent-v1'  │
└──────┬───────────────┘
       │
       │ 7. Config data
       ↓
┌──────────────────────┐
│  AgentFactory        │
│  create_agent()      │
│  - Build schema      │
│  - Create agent      │
└──────┬───────────────┘
       │
       │ 8. Agent instance
       ↓
┌──────────────────────┐
│  BaseAgent           │
│  Status: Active      │
│  Handling voice      │
└──────┬───────────────┘
       │
       │ 9. Voice data
       ↓
┌──────────────────────┐
│   Frontend           │
│   User conversation  │
└──────────────────────┘
```

---

## Detailed Sequence

### 1. Configuration Setup (Run Once or When Updating)

```bash
# Option A: Use setup script (recommended)
uv run python examples/setup_agents.py

# This script:
# 1. Connects to MongoDB
# 2. Calls config_store.create_agent_config() for each agent
# 3. Saves configurations to database
# 4. Exits (does NOT start worker)
```

```bash
# Option B: Use Admin API (for dynamic updates)
uvicorn src.api.admin_api:app --reload

# Then from frontend/Postman:
POST http://localhost:8000/api/v1/agents/config
{
  "agent_id": "support-agent-v1",
  ...
}

# This saves to MongoDB and returns
# Does NOT start any agent
```

### 2. Start Worker (Keep Running)

```bash
# Start the worker - this is what actually runs agents
uv run python examples/multi_agent_worker.py dev

# This command:
# 1. Calls cli.run_app(WorkerOptions(...))
# 2. Connects to LiveKit server
# 3. Registers entrypoint function
# 4. Waits for users to join rooms
# 5. Runs continuously (does NOT exit)
```

**Output**:
```
🚀 Starting multi-agent worker...
Supported agents: support, sales, onboarding, interview
Routing based on room metadata 'purpose' or 'agent_id'
INFO:livekit.agents:Worker connected to LiveKit server
INFO:livekit.agents:Waiting for jobs...
```

### 3. User Joins (Happens Automatically)

When a user joins from frontend:

```typescript
// Frontend code
await room.connect(wsURL, token, {
  metadata: JSON.stringify({
    purpose: 'support',
    user_id: 'user123'
  })
});
```

**What Happens**:
1. LiveKit server receives join request
2. Triggers worker's `entrypoint(ctx)` function
3. Worker reads room metadata
4. Worker determines agent type: `determine_agent_type()` → `"support-agent-v1"`
5. Worker loads config from MongoDB: `config_store.get_agent_config("support-agent-v1")`
6. Worker creates agent: `factory.create_agent(config)`
7. Agent session starts: `session.start(room, agent, ...)`
8. Voice conversation begins

---

## Code Walkthrough

### Admin API: create_agent_config

```python
# src/api/admin_api.py

@app.post("/api/v1/agents/config")
async def create_agent_config(
    config: AgentConfigRequest,
    store: ConfigurationStore = Depends(get_config_store)
):
    """
    Create new agent configuration.

    THIS ONLY SAVES TO DATABASE!
    Does NOT start any worker or agent.
    """
    try:
        config_id = await store.create_agent_config(config.dict())
        return {"config_id": config_id, "message": "Configuration created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**What this does**: Saves configuration to MongoDB and returns immediately.

### Worker: cli.run_app

```python
# examples/multi_agent_worker.py

if __name__ == "__main__":
    # This starts the LiveKit worker
    # It does NOT return - runs continuously
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,  # Called for each user join
            prewarm_fnc=prewarm          # Called on worker startup
        )
    )
```

**What this does**:
- Starts a long-running process
- Connects to LiveKit server
- Waits for room events
- Calls `entrypoint()` each time a user joins a room

### Worker: entrypoint (Called per session)

```python
# examples/multi_agent_worker.py

async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Called automatically when a user joins a room.
    This is where configuration meets execution.
    """

    # 1. Extract metadata
    room_metadata = json.loads(ctx.room.metadata or "{}")

    # 2. Determine which agent to use
    agent_id = determine_agent_type(room_metadata)
    # Returns: "support-agent-v1", "sales-agent-v1", etc.

    # 3. Load configuration from MongoDB
    config_store = ConfigurationStore(...)
    await config_store.initialize()

    # 4. Create agent using configuration
    factory = AgentFactory(config_store)
    session_manager = SessionManager(factory, config_store)

    # 5. Create session (this reads config from MongoDB)
    session_id = await session_manager.create_session(
        agent_id=agent_id,  # "support-agent-v1"
        user_id=user_id,
        room=ctx.room
    )

    # 6. Get the created agent instance
    agent = session_manager.active_sessions[session_id]

    # 7. Start voice session
    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent, ...)
```

---

## Common Patterns

### Pattern 1: Original PTEInterviewAgent (Hardcoded)

**No MongoDB needed** - Everything is in code

```bash
# Single command - no configuration setup needed
uv run python src/agent.py dev
```

```python
# src/agent.py

async def entrypoint(ctx: agents.JobContext):
    # Hardcoded agent - no config loading
    agent = PTEInterviewAgent(
        room=ctx.room,
        user_id=user_id,
        instructions=load_prompt("pte_interview.yaml")
    )

    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Pattern 2: Dynamic Multi-Agent (Configuration-based)

**Requires MongoDB** - Configurations stored externally

```bash
# Step 1: Setup configurations (one time)
uv run python examples/setup_agents.py

# Step 2: Start worker (keeps running)
uv run python examples/multi_agent_worker.py dev
```

```python
# examples/multi_agent_worker.py

async def entrypoint(ctx: agents.JobContext):
    # Dynamic agent - loads config from MongoDB
    agent_id = determine_agent_type(metadata)
    config = await config_store.get_agent_config(agent_id)
    agent = await factory.create_agent(config, ctx.room)

    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Pattern 3: Dynamic with Admin API (Full flexibility)

**Best for production** - Update configs without restarting worker

```bash
# Terminal 1: Start Admin API (optional, for runtime updates)
uvicorn src.api.admin_api:app --reload

# Terminal 2: Start worker
uv run python examples/multi_agent_worker.py dev
```

**Runtime updates**:
```bash
# Update prompt without restarting worker
curl -X PATCH http://localhost:8000/api/v1/agents/config/support-agent-v1 \
  -H "Content-Type: application/json" \
  -d '{"prompt_config": {"instructions": "New instructions..."}}'

# Next user who joins will get updated configuration
# Existing sessions continue with old config
```

---

## Key Takeaways

| Aspect | Admin API | LiveKit Worker |
|--------|-----------|----------------|
| **Purpose** | Manage configurations | Run agents |
| **Process** | FastAPI server | LiveKit worker |
| **Command** | `uvicorn src.api.admin_api:app` | `uv run python multi_agent_worker.py dev` |
| **Lifetime** | Request/response | Long-running |
| **Port** | 8000 (HTTP) | Connects to LiveKit (WebSocket) |
| **Triggers** | Manual API calls | User joins room |
| **Database** | Writes to MongoDB | Reads from MongoDB |
| **Required?** | Optional (can use scripts) | Required for agents to run |

---

## FAQ

### Q: Do I need to run both Admin API and Worker?

**A**: No. You only need the **worker**. The Admin API is optional for runtime configuration updates.

**Minimum setup**:
```bash
# Setup configs once (using script, not API)
uv run python examples/setup_agents.py

# Start worker (required)
uv run python examples/multi_agent_worker.py dev
```

### Q: When do configurations get loaded?

**A**: Configurations are loaded **when a user joins a room**, not when the worker starts.

```
Worker Start → Wait for users → User Joins → Load Config → Create Agent → Start Session
```

### Q: Can I update configurations without restarting the worker?

**A**: Yes! That's the whole point of the dynamic system.

1. Update config in MongoDB (via API or script)
2. Next user who joins gets the new config
3. Existing sessions continue with old config

### Q: How does the worker know which agent to use?

**A**: Via room metadata from the frontend:

```typescript
// Frontend specifies agent type
await room.connect(wsURL, token, {
  metadata: JSON.stringify({
    purpose: 'support'  // → routes to support-agent-v1
  })
});
```

### Q: What if I don't want dynamic agents?

**A**: Use the original `src/agent.py` pattern - no MongoDB needed:

```bash
# Simple, hardcoded agent
uv run python src/agent.py dev
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     THE KEY DISTINCTION                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Admin API / Setup Scripts                                  │
│  ========================                                    │
│  - Purpose: Store/update configurations                     │
│  - Runs: On-demand (when you want to change configs)       │
│  - Command: uvicorn src.api.admin_api:app                   │
│            OR uv run python examples/setup_agents.py        │
│  - Lifetime: Request/response or one-time script           │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  LiveKit Worker                                             │
│  ===============                                             │
│  - Purpose: Actually run agents and handle voice sessions   │
│  - Runs: Continuously (24/7 in production)                  │
│  - Command: uv run python multi_agent_worker.py dev         │
│  - Lifetime: Long-running service                           │
│  - Triggers: Automatically when users join rooms            │
│                                                              │
│  These are SEPARATE processes that communicate via MongoDB  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The Admin API **prepares** the configurations.
The Worker **executes** the agents.

They're like a **recipe book** (Admin API) and a **chef** (Worker) - completely separate roles that work together.
