# Dynamic Agent System - Best Practices & Advanced Patterns

## Table of Contents

1. [Configuration Management](#configuration-management)
2. [Schema Design](#schema-design)
3. [Performance Optimization](#performance-optimization)
4. [Security Guidelines](#security-guidelines)
5. [Multi-Tenancy](#multi-tenancy)
6. [Testing Strategies](#testing-strategies)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Deployment Patterns](#deployment-patterns)
9. [Error Handling](#error-handling)
10. [Advanced Patterns](#advanced-patterns)

---

## Configuration Management

### Versioning Strategy

**Use Semantic Versioning**

```python
# Good: Clear version progression
"version": "1.0.0"  # Initial release
"version": "1.1.0"  # New feature (backward compatible)
"version": "2.0.0"  # Breaking change

# Bad: No versioning
"version": "latest"
```

**Version Naming Convention**

```python
# Pattern: <agent-type>-<variant>-v<version>
"agent_id": "interview-strict-v1"
"agent_id": "interview-friendly-v1"
"agent_id": "sales-enterprise-v2"
```

### Configuration Immutability

**Create New Versions Instead of Modifying**

```python
# ✅ Good: Create new version for breaking changes
async def update_agent_breaking_change():
    # Copy existing config
    old_config = await store.get_agent_config("interview-v1")

    # Create new version with changes
    new_config = {
        **old_config,
        "agent_id": "interview-v2",
        "version": "2.0.0",
        "schema_config": {
            # New schema structure
        }
    }

    await store.create_agent_config(new_config)

# ❌ Bad: Modify existing production config
await store.update_agent_config("interview-v1", {
    "schema_config": new_schema  # Breaking change!
})
```

### Configuration Validation

**Validate Before Deployment**

```python
async def validate_configuration(config: dict) -> list[str]:
    """Validate configuration and return errors."""
    errors = []

    # Check required fields
    required = ["agent_id", "prompt_config", "schema_config", "llm_config"]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate schema fields
    for field_def in config.get("schema_config", {}).get("fields", []):
        if not field_def["field_name"].isidentifier():
            errors.append(f"Invalid field name: {field_def['field_name']}")

        # Validate field types
        valid_types = ["str", "int", "float", "bool", "list[str]", "list[int]", "dict"]
        if not any(field_def["field_type"].startswith(t) for t in valid_types):
            errors.append(f"Invalid field type: {field_def['field_type']}")

    # Validate LLM config
    if config.get("llm_config", {}).get("temperature", 0) > 2.0:
        errors.append("Temperature must be <= 2.0")

    return errors

# Usage
errors = await validate_configuration(new_config)
if errors:
    raise ValueError(f"Configuration validation failed: {errors}")
```

---

## Schema Design

### Field Naming Conventions

```python
# ✅ Good: Descriptive, snake_case
{
    "field_name": "credibility_score",
    "field_name": "interview_stage",
    "field_name": "system_response",
}

# ❌ Bad: Unclear, inconsistent
{
    "field_name": "score",          # Too vague
    "field_name": "InterviewStage", # PascalCase
    "field_name": "sys_resp",       # Abbreviations
}
```

### Field Organization

**Separate Concerns with Metadata**

```python
fields = [
    # Public fields (visible to frontend)
    {
        "field_name": "system_response",
        "field_type": "str",
        "description": "Agent's response to user",
        "required": True,
        "metadata": {
            "use_in_tts": False,
            "hidden": False,
            "category": "output"
        }
    },

    # TTS control fields
    {
        "field_name": "voice_instructions",
        "field_type": "str",
        "description": "TTS tone and pace directives",
        "required": False,
        "metadata": {
            "use_in_tts": True,  # Used by TTS node
            "hidden": False,
            "category": "control"
        }
    },

    # Internal fields (hidden from frontend)
    {
        "field_name": "internal_assessment",
        "field_type": "str",
        "description": "Private evaluation",
        "required": False,
        "metadata": {
            "use_in_tts": False,
            "hidden": True,       # Not sent to frontend
            "category": "internal"
        }
    },
]
```

### Schema Evolution

**Backward-Compatible Changes**

```python
# ✅ Safe: Adding optional fields
{
    "field_name": "new_metric",
    "field_type": "int",
    "required": False,  # Optional
    "default": 0,       # Has default
}

# ❌ Unsafe: Adding required fields
{
    "field_name": "new_required_field",
    "field_type": "str",
    "required": True,  # Breaking change!
}
```

### Type Safety

**Use Constraints for Validation**

```python
# Score fields
{
    "field_name": "quality_score",
    "field_type": "int",
    "constraints": {
        "ge": 0,   # Greater than or equal
        "le": 100  # Less than or equal
    }
}

# Enum-like fields
{
    "field_name": "sentiment",
    "field_type": "Literal['positive', 'neutral', 'negative']",
    "description": "Detected sentiment"
}

# Length constraints
{
    "field_name": "summary",
    "field_type": "str",
    "constraints": {
        "min_length": 10,
        "max_length": 500
    }
}
```

---

## Performance Optimization

### Configuration Caching

**Implement Multi-Level Caching**

```python
class ConfigurationStore:
    def __init__(self, redis_client=None):
        self._memory_cache = {}  # L1: In-memory
        self._redis = redis_client  # L2: Redis (shared across workers)
        self._db = None  # L3: MongoDB

    async def get_agent_config(self, agent_id: str) -> dict:
        # L1: Check in-memory cache
        if agent_id in self._memory_cache:
            config, timestamp = self._memory_cache[agent_id]
            if time.time() - timestamp < 300:  # 5 min TTL
                return config

        # L2: Check Redis
        if self._redis:
            cached = await self._redis.get(f"config:{agent_id}")
            if cached:
                config = json.loads(cached)
                self._memory_cache[agent_id] = (config, time.time())
                return config

        # L3: Query database
        config = await self._db.configs.find_one({"agent_id": agent_id})

        # Populate caches
        if self._redis:
            await self._redis.setex(
                f"config:{agent_id}",
                300,  # 5 min TTL
                json.dumps(config)
            )

        self._memory_cache[agent_id] = (config, time.time())

        return config
```

### Schema Registry

**Cache Generated Schema Classes**

```python
class SchemaRegistry:
    """Singleton registry for generated schemas."""

    _instance = None
    _schemas: dict[str, Type[TypedDict]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_or_create(
        self,
        schema_id: str,
        schema_config: dict,
        factory_fn: Callable
    ) -> Type[TypedDict]:
        """Get cached schema or create new one."""
        if schema_id not in self._schemas:
            self._schemas[schema_id] = factory_fn(schema_config)
        return self._schemas[schema_id]
```

### Connection Pooling

**Reuse Database Connections**

```python
from motor.motor_asyncio import AsyncIOMotorClient

class MongoConnectionPool:
    _client: Optional[AsyncIOMotorClient] = None

    @classmethod
    def get_client(cls, uri: str) -> AsyncIOMotorClient:
        if cls._client is None:
            cls._client = AsyncIOMotorClient(
                uri,
                maxPoolSize=50,      # Max connections
                minPoolSize=10,      # Min connections
                maxIdleTimeMS=45000, # Close idle after 45s
                serverSelectionTimeoutMS=5000,
            )
        return cls._client
```

### Lazy Loading

**Load Resources On-Demand**

```python
class BaseAgent:
    def __init__(self, ...):
        # Don't initialize heavy resources in __init__
        self._llm = None
        self._tts = None

    @property
    def llm(self):
        """Lazy-load LLM."""
        if self._llm is None:
            self._llm = self._create_llm(self.config["llm_config"])
        return self._llm

    @property
    def tts(self):
        """Lazy-load TTS."""
        if self._tts is None:
            self._tts = self._create_tts(self.config["tts_config"])
        return self._tts
```

---

## Security Guidelines

### Access Control

**Implement Role-Based Access Control (RBAC)**

```python
from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"

class Permission(str, Enum):
    CREATE_AGENT = "create_agent"
    UPDATE_AGENT = "update_agent"
    DELETE_AGENT = "delete_agent"
    VIEW_AGENT = "view_agent"

ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.CREATE_AGENT,
        Permission.UPDATE_AGENT,
        Permission.DELETE_AGENT,
        Permission.VIEW_AGENT,
    ],
    Role.DEVELOPER: [
        Permission.UPDATE_AGENT,
        Permission.VIEW_AGENT,
    ],
    Role.VIEWER: [
        Permission.VIEW_AGENT,
    ],
}

def check_permission(user_role: Role, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS.get(user_role, [])
```

### Input Validation

**Sanitize All Inputs**

```python
from pydantic import validator, Field

class SchemaFieldRequest(BaseModel):
    field_name: str = Field(..., regex=r'^[a-z_][a-z0-9_]*$')

    @validator("field_name")
    def validate_field_name(cls, v):
        # Prevent Python keywords
        import keyword
        if keyword.iskeyword(v):
            raise ValueError(f"Field name cannot be Python keyword: {v}")

        # Prevent dunder methods
        if v.startswith("__") and v.endswith("__"):
            raise ValueError(f"Field name cannot be dunder method: {v}")

        return v

    @validator("field_type")
    def validate_field_type(cls, v):
        # Whitelist allowed types
        allowed = [
            "str", "int", "float", "bool",
            "list[str]", "list[int]", "list[float]",
            "dict"
        ]

        if not any(v.startswith(t) for t in allowed) and not v.startswith("Literal["):
            raise ValueError(f"Invalid field type: {v}")

        return v
```

### Secrets Management

**Never Store Secrets in Configurations**

```python
# ✅ Good: Reference secrets, don't store them
{
    "llm_config": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",  # Reference to env var
    }
}

# ❌ Bad: Hardcoded secrets
{
    "llm_config": {
        "provider": "openai",
        "api_key": "sk-abc123...",  # Never do this!
    }
}
```

### Audit Logging

**Log All Configuration Changes**

```python
async def create_agent_config(self, config: dict, user_id: str) -> str:
    """Create config with audit trail."""

    # Create audit log entry
    audit_entry = {
        "timestamp": datetime.utcnow(),
        "user_id": user_id,
        "action": "create_agent_config",
        "agent_id": config["agent_id"],
        "changes": config,
    }

    await self.audit_logs.insert_one(audit_entry)

    # Create config
    result = await self.configs.insert_one(config)

    return str(result.inserted_id)
```

---

## Multi-Tenancy

### Tenant Isolation

**Implement Strict Data Isolation**

```python
class TenantIsolationMiddleware:
    """Ensures tenant data isolation."""

    def __init__(self, config_store: ConfigurationStore):
        self.config_store = config_store

    async def get_agent_config(
        self,
        agent_id: str,
        tenant_id: str
    ) -> dict:
        """Get config with tenant check."""
        config = await self.config_store.get_agent_config(agent_id)

        # Verify tenant has access
        if config["tenant_config"]["tenant_id"] != tenant_id:
            raise PermissionError(
                f"Tenant {tenant_id} cannot access agent {agent_id}"
            )

        return config
```

### Resource Quotas

**Enforce Per-Tenant Limits**

```python
class TenantQuotaManager:
    """Manages per-tenant resource quotas."""

    def __init__(self):
        self.quotas = {
            "free": {
                "max_sessions": 10,
                "max_agents": 3,
                "max_llm_calls_per_day": 1000,
            },
            "pro": {
                "max_sessions": 100,
                "max_agents": 20,
                "max_llm_calls_per_day": 50000,
            },
            "enterprise": {
                "max_sessions": 1000,
                "max_agents": 100,
                "max_llm_calls_per_day": 1000000,
            }
        }

    async def check_quota(
        self,
        tenant_id: str,
        tier: str,
        resource: str
    ) -> bool:
        """Check if tenant has available quota."""
        usage = await self._get_usage(tenant_id, resource)
        limit = self.quotas[tier][resource]

        return usage < limit

    async def increment_usage(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1
    ):
        """Increment resource usage counter."""
        await self.usage_db.update_one(
            {"tenant_id": tenant_id},
            {"$inc": {f"usage.{resource}": amount}},
            upsert=True
        )
```

---

## Testing Strategies

### Unit Tests

**Test Schema Generation**

```python
import pytest
from orchestrator.agent_factory import AgentFactory

@pytest.mark.asyncio
async def test_dynamic_schema_creation():
    """Test dynamic schema generation."""

    schema_config = {
        "schema_id": "test_schema",
        "fields": [
            {
                "field_name": "test_field",
                "field_type": "str",
                "description": "Test field",
                "required": True,
            }
        ]
    }

    factory = AgentFactory(mock_config_store)
    schema_class = factory._create_schema_from_config(schema_config)

    # Verify schema structure
    assert "test_field" in schema_class.__annotations__
    assert schema_class.__annotations__["test_field"][0] == str
```

### Integration Tests

**Test End-to-End Agent Creation**

```python
@pytest.mark.asyncio
async def test_dynamic_agent_creation_flow():
    """Test complete agent creation flow."""

    # 1. Setup
    config_store = ConfigurationStore(...)
    await config_store.initialize()

    # 2. Create config
    config_id = await config_store.create_agent_config({
        "agent_id": "test-agent",
        # ... config
    })

    # 3. Create factory and session manager
    factory = AgentFactory(config_store)
    session_manager = SessionManager(factory, config_store)

    # 4. Create session
    session_id = await session_manager.create_session(
        agent_id="test-agent",
        user_id="test-user",
        tenant_id="test-tenant",
        room=MockRoom(),
    )

    # 5. Verify agent is active
    assert session_id in session_manager.active_sessions

    # 6. Cleanup
    await session_manager.end_session(session_id)
    await config_store.close()
```

### Load Testing

**Test Concurrent Session Creation**

```python
import asyncio
import pytest

@pytest.mark.asyncio
async def test_concurrent_session_creation():
    """Test system under concurrent load."""

    session_manager = SessionManager(...)

    # Create 100 sessions concurrently
    tasks = [
        session_manager.create_session(
            agent_id="test-agent",
            user_id=f"user-{i}",
            tenant_id="test",
            room=MockRoom(),
        )
        for i in range(100)
    ]

    session_ids = await asyncio.gather(*tasks)

    # Verify all created successfully
    assert len(session_ids) == 100
    assert session_manager.get_active_session_count() == 100
```

---

## Monitoring & Debugging

### Structured Logging

**Use Structured Logs for Better Debugging**

```python
import structlog

logger = structlog.get_logger()

class BaseAgent:
    def __init__(self, ...):
        self.logger = logger.bind(
            agent_id=self.config["agent_id"],
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.config["tenant_config"]["tenant_id"],
        )

    async def llm_node(self, ...):
        self.logger.info(
            "llm_request",
            model=self.config["llm_config"]["model"],
            temperature=self.config["llm_config"]["temperature"],
        )

        # ... process

        self.logger.info(
            "llm_response",
            response_length=len(response),
            duration_ms=duration * 1000,
        )
```

### Metrics Collection

**Track Key Performance Indicators**

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
session_created = Counter(
    "agent_session_created_total",
    "Total sessions created",
    ["agent_id", "tenant_id"]
)

session_duration = Histogram(
    "agent_session_duration_seconds",
    "Session duration",
    ["agent_id"]
)

active_sessions = Gauge(
    "agent_active_sessions",
    "Current active sessions",
    ["agent_id", "tenant_id"]
)

# Use in code
class SessionManager:
    async def create_session(self, ...):
        session_created.labels(
            agent_id=agent_id,
            tenant_id=tenant_id
        ).inc()

        active_sessions.labels(
            agent_id=agent_id,
            tenant_id=tenant_id
        ).inc()

        # ... create session
```

### Health Checks

**Implement Comprehensive Health Checks**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness_check(
    config_store: ConfigurationStore = Depends(get_config_store)
):
    """Readiness check with dependency validation."""
    checks = {}

    # Check MongoDB
    try:
        await config_store.configs.find_one({})
        checks["mongodb"] = "healthy"
    except Exception as e:
        checks["mongodb"] = f"unhealthy: {e}"

    # Check configuration cache
    checks["cache_size"] = len(config_store._cache)

    # Overall status
    status = "healthy" if all(
        v == "healthy" for v in checks.values() if isinstance(v, str)
    ) else "degraded"

    return {"status": status, "checks": checks}
```

---

## Deployment Patterns

### Blue-Green Deployment

**Deploy New Versions Safely**

```python
# 1. Deploy new version alongside old
await config_store.create_agent_config({
    "agent_id": "interview-agent-v2",  # New version
    "enabled": True,
    # ... new configuration
})

# 2. Route 10% traffic to new version
async def select_agent(user_id: str) -> str:
    # Hash-based routing for consistent user experience
    if hash(user_id) % 10 == 0:
        return "interview-agent-v2"  # New version
    return "interview-agent-v1"      # Old version

# 3. Monitor metrics for v2

# 4. Gradually increase traffic
# 20%, 50%, 100%

# 5. Deprecate v1
await config_store.update_agent_config(
    "interview-agent-v1",
    {"enabled": False}
)
```

### Feature Flags

**Toggle Features Dynamically**

```python
# Add feature flags to config
{
    "agent_id": "my-agent",
    "feature_flags": {
        "enable_voice_analytics": True,
        "enable_sentiment_analysis": False,
        "enable_auto_scoring": True,
    }
}

# Use in agent
class BaseAgent:
    def is_feature_enabled(self, feature: str) -> bool:
        return self.config.get("feature_flags", {}).get(feature, False)

    async def process_response(self, response: str):
        # Conditional feature execution
        if self.is_feature_enabled("enable_sentiment_analysis"):
            sentiment = await self.analyze_sentiment(response)
```

---

## Error Handling

### Graceful Degradation

**Handle Failures Gracefully**

```python
class BaseAgent:
    async def llm_node(self, ...):
        try:
            # Try structured output
            async for chunk in self._llm_structured(chat_ctx):
                yield chunk

        except Exception as e:
            logger.error(f"Structured output failed: {e}")

            # Fallback to plain text
            logger.info("Falling back to plain text mode")
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk
```

### Retry Logic

**Implement Exponential Backoff**

```python
import asyncio

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay}s..."
            )
            await asyncio.sleep(delay)

# Usage
config = await retry_with_backoff(
    lambda: config_store.get_agent_config("my-agent")
)
```

---

## Advanced Patterns

### Hot Configuration Reload

**Update Agent Configs Without Restart**

```python
class ConfigWatcher:
    """Watch for configuration changes and reload."""

    def __init__(self, config_store, session_manager):
        self.config_store = config_store
        self.session_manager = session_manager

    async def start_watching(self):
        """Start watching for config changes."""
        async for change in self.config_store.watch_changes():
            await self._handle_config_change(change)

    async def _handle_config_change(self, change: dict):
        """Handle configuration change event."""
        agent_id = change["agent_id"]

        # Notify all active sessions using this agent
        for session_id, agent in self.session_manager.active_sessions.items():
            if agent.config["agent_id"] == agent_id:
                # Send notification to frontend
                await agent.data_streamer.send_structured_data(
                    topic="config-update",
                    data={
                        "type": "config_changed",
                        "message": "Agent configuration updated. Reconnect to use new version.",
                        "agent_id": agent_id,
                    }
                )
```

### A/B Testing Framework

**Test Different Configurations**

```python
class ABTestManager:
    """Manage A/B tests for agent configurations."""

    def __init__(self):
        self.experiments = {}

    def create_experiment(
        self,
        name: str,
        variant_a: str,  # agent_id
        variant_b: str,  # agent_id
        traffic_split: float = 0.5,
    ):
        """Create new A/B test."""
        self.experiments[name] = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "traffic_split": traffic_split,
            "metrics": {"a": {}, "b": {}},
        }

    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Get variant for user."""
        experiment = self.experiments[experiment_name]

        # Consistent hash-based assignment
        if hash(f"{experiment_name}:{user_id}") % 100 < experiment["traffic_split"] * 100:
            return experiment["variant_a"]
        return experiment["variant_b"]

    async def record_metric(
        self,
        experiment_name: str,
        variant: str,
        metric_name: str,
        value: float,
    ):
        """Record experiment metric."""
        # Store in database for analysis
        pass

# Usage
ab_test = ABTestManager()
ab_test.create_experiment(
    name="interview_temperature_test",
    variant_a="interview-temp-0.7",
    variant_b="interview-temp-0.9",
    traffic_split=0.5,
)

# In entrypoint
agent_id = ab_test.get_variant("interview_temperature_test", user_id)
```

---

## Conclusion

These best practices ensure your dynamic agent system is:

- ✅ **Maintainable** - Clean code, good patterns
- ✅ **Scalable** - Caching, pooling, optimization
- ✅ **Secure** - RBAC, validation, audit logs
- ✅ **Reliable** - Error handling, retry logic, health checks
- ✅ **Observable** - Logging, metrics, tracing
- ✅ **Flexible** - A/B testing, feature flags, hot reload

Continue refining based on production learnings and evolving requirements.
