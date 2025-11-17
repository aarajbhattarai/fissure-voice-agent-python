# Dynamic Agent System - Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing and using the dynamic agent architecture.

---

## Prerequisites

1. **MongoDB** - For configuration storage
   ```bash
   # Using Docker
   docker run -d -p 27017:27017 --name mongodb mongo:latest
   ```

2. **Python Dependencies**
   ```bash
   # Install new dependencies
   uv sync
   ```

3. **Environment Variables**
   ```bash
   # Add to .env.local
   MONGO_URI=mongodb://localhost:27017
   ```

---

## Step 1: Database Setup

### Initialize Configuration Store

```python
from orchestrator.configuration_store import ConfigurationStore

# Create store instance
store = ConfigurationStore(
    connection_string="mongodb://localhost:27017",
    database="agent_configs"
)

# Initialize indexes
await store.initialize()
```

---

## Step 2: Migrate Existing Agent

### Run Migration Script

```bash
# Migrate PTEInterviewAgent to dynamic configuration
uv run python examples/migration_script.py
```

This creates a configuration entry in MongoDB with:
- Agent ID: `pte-interview-agent-v1`
- All existing fields from `InterviewTurnJSON`
- LLM, TTS, STT configurations
- Tracing configuration (enabled by default)

### Verify Migration

```bash
# Start the admin API
uv run python src/api/admin_api.py

# In another terminal, verify the configuration
curl http://localhost:8000/api/v1/agents/config/pte-interview-agent-v1 | jq
```

---

## Step 3: Start Admin API

The Admin API provides REST endpoints for managing agent configurations.

```bash
# Start API server
uv run python src/api/admin_api.py
```

API will be available at: `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI

---

## Step 4: Create Custom Agent Configurations

### Option 1: Via Python Script

```bash
# Create a sales agent
uv run python examples/create_custom_agent.py
```

### Option 2: Via API

```bash
# Create interview agent variant
curl -X POST http://localhost:8000/api/v1/agents/config \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "interview-agent-strict-v1",
    "agent_type": "interview",
    "version": "1.0.0",
    "enabled": true,
    "prompt_config": {
      "template_id": "strict_interview",
      "instructions": "You are a very strict interviewer..."
    },
    "schema_config": {
      "schema_id": "strict_schema_v1",
      "fields": [
        {
          "field_name": "system_response",
          "field_type": "str",
          "description": "Response to candidate",
          "required": true
        }
      ]
    },
    "llm_config": {
      "provider": "openai",
      "model": "gpt-5-nano",
      "temperature": 0.5,
      "supports_structured_output": true
    },
    "tts_config": {
      "provider": "deepgram",
      "instruction_field": "voice_instructions"
    },
    "tracing_config": {
      "enabled": true,
      "provider": "langfuse"
    }
  }'
```

---

## Step 5: Dynamic Schema Modification

### Add New Field to Existing Agent

```bash
# Add pronunciation_score field
curl -X POST http://localhost:8000/api/v1/agents/config/pte-interview-agent-v1/schema/fields \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "pronunciation_score",
    "field_type": "int",
    "description": "Pronunciation quality score (0-100)",
    "required": false,
    "default": 0,
    "constraints": {"ge": 0, "le": 100},
    "metadata": {
      "use_in_tts": false,
      "hidden": false
    }
  }'
```

### Remove Field from Schema

```bash
# Remove a field
curl -X DELETE http://localhost:8000/api/v1/agents/config/pte-interview-agent-v1/schema/fields/pronunciation_score
```

---

## Step 6: Toggle Features at Runtime

### Enable/Disable Tracing

```bash
# Disable tracing
curl -X POST "http://localhost:8000/api/v1/agents/config/pte-interview-agent-v1/tracing/toggle?enabled=false"

# Enable tracing
curl -X POST "http://localhost:8000/api/v1/agents/config/pte-interview-agent-v1/tracing/toggle?enabled=true"
```

### Update Agent Configuration

```bash
# Update LLM temperature
curl -X PUT http://localhost:8000/api/v1/agents/config/pte-interview-agent-v1 \
  -H "Content-Type: application/json" \
  -d '{
    "llm_config": {
      "temperature": 0.9
    }
  }'
```

---

## Step 7: Run Dynamic Agent

### Update Entrypoint

Replace the hardcoded entrypoint in `src/agent.py` with the dynamic version:

```bash
# Copy the dynamic entrypoint
cp examples/dynamic_agent_entrypoint.py src/agent_dynamic.py
```

### Start Agent Worker

```bash
# Run with dynamic configuration
uv run python src/agent_dynamic.py dev
```

### Connect from Frontend

When connecting to LiveKit room, pass agent configuration in room metadata:

```javascript
// Frontend code
const roomMetadata = {
  agent_id: "pte-interview-agent-v1",  // Which agent config to use
  user_id: "user-123",
  tenant_id: "customer-abc",
  user_details: {
    name: "John Doe",
    email: "john@example.com"
  }
};

const room = new Room({
  metadata: JSON.stringify(roomMetadata)
});
```

---

## Step 8: Multi-Tenant Setup

### Create Tenant-Specific Agents

```python
# Create agent for Tenant A
await store.create_agent_config({
    "agent_id": "interview-agent-tenant-a",
    "tenant_config": {
        "tenant_id": "tenant-a",
        "isolation_level": "organization"
    },
    # ... rest of config
})

# Create agent for Tenant B with different settings
await store.create_agent_config({
    "agent_id": "interview-agent-tenant-b",
    "tenant_config": {
        "tenant_id": "tenant-b",
        "isolation_level": "organization"
    },
    "llm_config": {
        "temperature": 0.3,  # Different temperature
    },
    # ... rest of config
})
```

---

## Step 9: Monitoring and Observability

### View Active Sessions

```python
from orchestrator import SessionManager

# Get session count
count = session_manager.get_active_session_count()
print(f"Active sessions: {count}")

# Get sessions by tenant
tenant_sessions = session_manager.get_sessions_by_tenant("tenant-a")
print(f"Tenant A sessions: {tenant_sessions}")
```

### Cleanup Inactive Sessions

```python
# Cleanup sessions inactive for more than 1 hour
await session_manager.cleanup_inactive_sessions(timeout_seconds=3600)
```

---

## Step 10: Advanced Patterns

### Session-Specific Overrides

```python
# Create session with custom overrides
session_id = await session_manager.create_session(
    agent_id="pte-interview-agent-v1",
    user_id="user-123",
    tenant_id="default",
    room=room,
    overrides={
        "llm_config": {
            "temperature": 0.95  # Override just for this session
        },
        "tracing_config": {
            "enabled": False  # Disable tracing for this session
        }
    }
)
```

### Custom Agent Types

Create a specialized agent by extending `BaseAgent`:

```python
# src/agents/specialized_agent.py

from agents.base_agent import BaseAgent

class SpecializedInterviewAgent(BaseAgent):
    """Specialized interview agent with custom behavior."""

    async def on_enter(self) -> None:
        """Custom welcome message."""
        await self.session.say(
            text=f"Welcome {self.user_details.get('name')}. "
                 f"I'm your specialized interview officer."
        )

    async def on_custom_event(self, event_data: dict):
        """Handle custom events."""
        # Custom business logic
        pass
```

Register in factory:

```python
# In agent_factory.py
factory.agent_registry["specialized_interview"] = SpecializedInterviewAgent
```

---

## Best Practices

### 1. Configuration Versioning

Always version your configurations:

```python
config = {
    "agent_id": "my-agent",
    "version": "1.2.0",  # Semantic versioning
    # ...
}
```

### 2. Schema Field Naming

Use snake_case for field names:
```python
{
    "field_name": "credibility_score",  # ✅ Good
    "field_name": "CredibilityScore",   # ❌ Bad
}
```

### 3. Gradual Rollout

When updating configurations:
1. Create new version with `agent_id: "my-agent-v2"`
2. Test thoroughly
3. Gradually migrate traffic
4. Deprecate old version

### 4. Tracing Strategy

Enable tracing for:
- Production debugging
- Performance analysis
- Quality monitoring

Disable tracing for:
- High-volume, low-value sessions
- Cost optimization
- Privacy-sensitive scenarios

### 5. Schema Design

Keep schemas focused:
- Include only necessary fields
- Use `hidden: true` for internal fields
- Set appropriate defaults
- Add validation constraints

### 6. Security

**Never** store sensitive data in configurations:
- API keys → Environment variables
- User passwords → Never in configs
- PII → User details, not config

### 7. Performance

- Cache configurations (default TTL: 5 minutes)
- Use connection pooling for MongoDB
- Limit concurrent sessions per tenant
- Clean up inactive sessions regularly

---

## Troubleshooting

### Issue: Configuration not found

```bash
# Verify agent exists
curl http://localhost:8000/api/v1/agents/list | jq
```

### Issue: Schema validation error

Check field types are valid:
- `str`, `int`, `float`, `bool`
- `list[str]`, `list[int]`
- `Literal['option1', 'option2']`

### Issue: Tracing not working

Verify:
1. `tracing_config.enabled` is `true`
2. Langfuse credentials in `.env.local`
3. Check Langfuse dashboard

### Issue: Agent not responding

Check:
1. MongoDB is running
2. Configuration is enabled
3. Room metadata contains correct `agent_id`
4. Logs for errors

---

## Migration Checklist

- [ ] MongoDB running and accessible
- [ ] Run migration script
- [ ] Verify configuration via API
- [ ] Test with dev environment
- [ ] Update frontend to pass room metadata
- [ ] Deploy admin API
- [ ] Update worker entrypoint
- [ ] Test dynamic schema changes
- [ ] Test tracing toggle
- [ ] Monitor production rollout

---

## Next Steps

1. **Build Dashboard UI** - Create React/Next.js dashboard for visual config management
2. **Add Authentication** - Implement JWT auth for admin API
3. **Metrics Collection** - Track agent performance metrics
4. **A/B Testing** - Framework for testing different configurations
5. **Auto-scaling** - Dynamic worker scaling based on load

---

## Support

For issues or questions:
1. Check logs: `uv run python src/agent_dynamic.py dev`
2. Verify configuration: `curl http://localhost:8000/api/v1/agents/config/<agent_id>`
3. Check MongoDB: `mongo agent_configs` → `db.agent_configurations.find()`

---

## Resources

- [DYNAMIC_AGENT_ARCHITECTURE.md](./DYNAMIC_AGENT_ARCHITECTURE.md) - Full architecture design
- [Examples](./examples/) - Example scripts and usage patterns
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [LiveKit Agents Docs](https://docs.livekit.io/agents/) - LiveKit framework docs
