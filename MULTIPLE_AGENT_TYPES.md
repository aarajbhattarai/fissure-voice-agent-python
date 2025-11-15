```markdown
# Running Multiple Agent Types - Complete Guide

This guide shows you how to run **different types of agents** (Support, Sales, Onboarding, Interview) using the dynamic agent architecture.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Agent Types](#available-agent-types)
3. [Setup Process](#setup-process)
4. [Running Agents](#running-agents)
5. [Routing Strategies](#routing-strategies)
6. [Frontend Integration](#frontend-integration)
7. [Managing Multiple Workers](#managing-multiple-workers)
8. [Production Deployment](#production-deployment)

---

## Quick Start

### 1. Setup MongoDB

```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 2. Configure Environment

```bash
# .env.local
MONGO_URI=mongodb://localhost:27017
LIVEKIT_URL=wss://your-livekit-server
LIVEKIT_API_KEY=your-key
LIVEKIT_API_SECRET=your-secret
OPENAI_API_KEY=your-key
DEEPGRAM_API_KEY=your-key
```

### 3. Create Agent Configurations

```bash
# This creates Support, Sales, and Onboarding agents in MongoDB
uv run python examples/setup_agents.py
```

### 4. Start Multi-Agent Worker

```bash
# This worker can handle ALL agent types
uv run python examples/multi_agent_worker.py dev
```

### 5. Connect from Frontend

```javascript
// Support session
const metadata = { purpose: 'support', user_id: 'user-123' };

// Sales session
const metadata = { purpose: 'sales', user_id: 'user-456' };

// Onboarding session
const metadata = { purpose: 'onboarding', user_id: 'user-789' };
```

**That's it!** The same worker handles all agent types automatically.

---

## Available Agent Types

### 1. **Support Agent** (`support-agent-v1`)

**Purpose:** Customer support for SaaS products

**Schema Fields:**
- `system_response` - Agent's response
- `issue_category` - technical, billing, account, feature_request, other
- `sentiment` - Customer's emotional state
- `issue_resolved` - Whether problem is solved
- `escalation_needed` - Should escalate to human?

**Use Cases:**
- Technical support
- Billing questions
- Account help
- Feature requests

**Example Conversation:**
```
User: "My account login isn't working"
Agent: "I'm sorry to hear that. Let me help you troubleshoot.
       Have you tried resetting your password?"
```

### 2. **Sales Agent** (`sales-agent-v1`)

**Purpose:** Sales conversations and lead qualification

**Schema Fields:**
- `system_response` - Agent's response
- `sales_stage` - discovery, needs_analysis, presentation, objection_handling, closing
- `customer_intent` - Buying readiness
- `pain_points` - Identified problems
- `qualification_score` - Lead quality (0-100)
- `objections_raised` - Customer concerns

**Use Cases:**
- Product demos
- Lead qualification
- Pricing discussions
- Objection handling

**Example Conversation:**
```
User: "Tell me about your pricing"
Agent: "Before we dive into pricing, I'd love to understand
       your current challenges. What problems are you trying
       to solve?"
```

### 3. **Onboarding Agent** (`onboarding-agent-v1`)

**Purpose:** Guide new users through product setup

**Schema Fields:**
- `system_response` - Agent's response
- `onboarding_step` - welcome, account_setup, profile_creation, feature_tour, first_action, completion
- `completion_percentage` - Progress (0-100)
- `user_understanding` - clear, confused, needs_help
- `next_step` - What to do next

**Use Cases:**
- New user welcome
- Account setup
- Feature introduction
- First-time guidance

**Example Conversation:**
```
User: "I just signed up, what should I do first?"
Agent: "Welcome! I'm excited to help you get started.
       Let's begin by setting up your profile.
       What would you like us to call you?"
```

### 4. **Interview Agent** (`pte-interview-agent-v1`)

**Purpose:** PTE interview practice (your existing agent)

**Schema Fields:**
- `system_response` - Agent's response
- `interview_stage` - document_check, background_inquiry, etc.
- `credibility_score` - Assessment (1-10)
- `red_flags` - Concerns identified
- `voice_instructions` - TTS directives

**Use Cases:**
- Interview practice
- Communication assessment
- Scenario simulation

---

## Setup Process

### Step 1: Create Agent Configurations

Run the setup script to create all agent configs in MongoDB:

```bash
uv run python examples/setup_agents.py
```

**Output:**
```
==============================================================
AGENT CONFIGURATION SETUP
==============================================================

Connecting to MongoDB: mongodb://localhost:27017
✅ Connected to MongoDB

Creating agent configurations...

✅ Support Agent
   Agent ID: support-agent-v1
   Schema: support_turn_v1
   Fields: 7
   Config ID: 507f1f77bcf86cd799439011

✅ Sales Agent
   Agent ID: sales-agent-v1
   Schema: sales_turn_v1
   Fields: 8
   Config ID: 507f1f77bcf86cd799439012

✅ Onboarding Agent
   Agent ID: onboarding-agent-v1
   Schema: onboarding_turn_v1
   Fields: 6
   Config ID: 507f1f77bcf86cd799439013

==============================================================
SETUP SUMMARY
==============================================================
Created: 3
Already existed: 0
Failed: 0
Total: 3

✅ Agent configurations ready!
```

### Step 2: Verify Configurations

Use the Admin API to view configurations:

```bash
# Start API
uv run python src/api/admin_api.py

# Visit http://localhost:8000/docs
# Or use curl
curl http://localhost:8000/api/v1/agents/list | jq
```

---

## Running Agents

### Option 1: Single Multi-Agent Worker (Recommended)

**One worker handles all agent types:**

```bash
uv run python examples/multi_agent_worker.py dev
```

**How it works:**
- Worker listens for room join events
- Reads `purpose` or `agent_id` from room metadata
- Routes to appropriate agent configuration
- Creates correct agent instance dynamically

**Logs:**
```
🚀 Starting multi-agent worker...
Supported agents: support, sales, onboarding, interview
Routing based on room metadata 'purpose' or 'agent_id'

# User joins for support
INFO: New session request - User: user-123, Tenant: default
INFO: Routing support session to support-agent-v1
INFO: Selected agent: support-agent-v1
INFO: Created session: session-abc with agent: support-agent-v1

# Another user joins for sales (concurrent!)
INFO: New session request - User: user-456, Tenant: default
INFO: Routing sales session to sales-agent-v1
INFO: Selected agent: sales-agent-v1
INFO: Created session: session-def with agent: sales-agent-v1
```

### Option 2: Dedicated Workers per Agent Type

**Run separate workers for each agent type:**

```bash
# Terminal 1: Support agent only
AGENT_TYPE=support uv run python examples/dedicated_support_worker.py dev

# Terminal 2: Sales agent only
AGENT_TYPE=sales uv run python examples/dedicated_sales_worker.py dev

# Terminal 3: Onboarding agent only
AGENT_TYPE=onboarding uv run python examples/dedicated_onboarding_worker.py dev
```

**When to use:**
- Different resource requirements per agent
- Separate scaling needs
- Isolation for testing

### Option 3: Keep Using PTEInterviewAgent

**Continue using your existing agent:**

```bash
# Your original command still works!
uv run src/agent.py dev
```

**No changes needed** - the new dynamic architecture is fully backward compatible.

---

## Routing Strategies

The multi-agent worker supports **4 routing strategies**:

### 1. Explicit Agent ID (Highest Priority)

Frontend specifies exact agent to use:

```javascript
const metadata = {
  agent_id: 'support-agent-v1',  // Explicit
  user_id: 'user-123'
};
```

**Backend routing:**
```python
def determine_agent_type(room_metadata: dict) -> str:
    if "agent_id" in room_metadata:
        return room_metadata["agent_id"]
    # ...
```

### 2. Purpose-Based Routing

Frontend specifies purpose/intent:

```javascript
const metadata = {
  purpose: 'support',  // Auto-routes to support-agent-v1
  user_id: 'user-123'
};
```

**Backend routing:**
```python
purpose = room_metadata.get("purpose", "").lower()
agent_map = {
    "support": "support-agent-v1",
    "sales": "sales-agent-v1",
    "onboarding": "onboarding-agent-v1",
}
return agent_map.get(purpose, "support-agent-v1")
```

### 3. User Attribute-Based Routing

Route based on user characteristics:

```javascript
const metadata = {
  user_id: 'user-123',
  user_details: {
    is_new_user: true,       // → Onboarding agent
    tier: 'premium',         // → Priority support
    last_purchase: '2024-01-15',
  }
};
```

**Backend routing:**
```python
user_details = room_metadata.get("user_details", {})

# New users → Onboarding
if user_details.get("is_new_user", False):
    return "onboarding-agent-v1"

# Premium → Priority support
if user_details.get("tier") == "premium":
    return "support-agent-v1"
```

### 4. Default Fallback

If no routing info provided, use default:

```python
# Fallback to support agent
return "support-agent-v1"
```

### Custom Routing Logic

**Example: Time-based routing**

```python
from datetime import datetime

def determine_agent_type(room_metadata: dict) -> str:
    # Business hours → Human-like support
    # After hours → More automated responses

    hour = datetime.now().hour

    if 9 <= hour < 17:  # Business hours
        return "support-agent-friendly-v1"
    else:
        return "support-agent-concise-v1"
```

**Example: A/B testing**

```python
import hashlib

def determine_agent_type(room_metadata: dict) -> str:
    user_id = room_metadata.get("user_id", "")

    # Consistent hash-based routing
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)

    if hash_val % 2 == 0:
        return "sales-agent-v1"  # Variant A
    else:
        return "sales-agent-v2"  # Variant B
```

---

## Frontend Integration

### React Component Example

```jsx
import { useState } from 'react';
import { Room } from 'livekit-client';

function MultiAgentInterface() {
  const [agentType, setAgentType] = useState('support');
  const [room, setRoom] = useState(null);

  const startSession = async (type) => {
    // 1. Get token with purpose
    const response = await fetch('/api/livekit/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        roomName: `${type}-${Date.now()}`,
        userName: 'John Doe',
        metadata: {
          purpose: type,          // 'support', 'sales', 'onboarding'
          user_id: 'user-123',
          user_details: {
            name: 'John Doe',
            email: 'john@example.com',
            tier: 'premium',
          }
        }
      })
    });

    const { token } = await response.json();

    // 2. Connect to room
    const newRoom = new Room();

    newRoom.on('dataReceived', (payload, participant, kind, topic) => {
      const data = JSON.parse(new TextDecoder().decode(payload));

      if (topic === 'conversation-summary') {
        console.log('Summary:', data.summary);
      }
    });

    await newRoom.connect(LIVEKIT_URL, token);
    await newRoom.localParticipant.setMicrophoneEnabled(true);

    setRoom(newRoom);
  };

  return (
    <div>
      <h1>Choose Agent Type</h1>

      <button onClick={() => startSession('support')}>
        💬 Customer Support
      </button>

      <button onClick={() => startSession('sales')}>
        💰 Talk to Sales
      </button>

      <button onClick={() => startSession('onboarding')}>
        🚀 Get Started (Onboarding)
      </button>

      <button onClick={() => startSession('interview')}>
        🎤 Practice Interview
      </button>
    </div>
  );
}
```

### Backend Token API

```javascript
// /api/livekit/token
import { AccessToken } from 'livekit-server-sdk';

export default async function handler(req, res) {
  const { roomName, userName, metadata } = req.body;

  const at = new AccessToken(
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET,
    {
      identity: userName,
      metadata: JSON.stringify(metadata),  // Pass purpose and user data
    }
  );

  at.addGrant({
    roomJoin: true,
    room: roomName,
    canPublish: true,
    canSubscribe: true,
  });

  res.json({ token: at.toJwt() });
}
```

---

## Managing Multiple Workers

### Scenario 1: One Worker for Everything

```bash
# Single command handles all agent types
uv run python examples/multi_agent_worker.py dev
```

**Pros:**
- ✅ Simple deployment
- ✅ Single point of management
- ✅ Automatic load balancing

**Cons:**
- ❌ All agents share resources
- ❌ Can't scale agent types independently

### Scenario 2: Multiple Workers (Same Code)

```bash
# Terminal 1
uv run python examples/multi_agent_worker.py dev

# Terminal 2
uv run python examples/multi_agent_worker.py dev

# Terminal 3
uv run python examples/multi_agent_worker.py dev
```

**LiveKit automatically load-balances** across workers!

```
User 1 (support) → Worker 1
User 2 (sales)   → Worker 2
User 3 (support) → Worker 3
User 4 (sales)   → Worker 1 (round-robin)
```

**Pros:**
- ✅ Horizontal scaling
- ✅ Higher concurrency
- ✅ Fault tolerance

### Scenario 3: Dedicated Workers per Type

```bash
# Support worker (Terminals 1-2)
AGENT_FILTER=support uv run python examples/multi_agent_worker.py dev
AGENT_FILTER=support uv run python examples/multi_agent_worker.py dev

# Sales worker (Terminals 3-5)
AGENT_FILTER=sales uv run python examples/multi_agent_worker.py dev
AGENT_FILTER=sales uv run python examples/multi_agent_worker.py dev
AGENT_FILTER=sales uv run python examples/multi_agent_worker.py dev
```

**Modify routing to filter:**

```python
async def entrypoint(ctx: agents.JobContext) -> None:
    agent_id = determine_agent_type(room_metadata)

    # Filter: Only handle specific agent type
    allowed_filter = os.getenv("AGENT_FILTER")
    if allowed_filter and allowed_filter not in agent_id:
        logger.info(f"Skipping {agent_id} (filter: {allowed_filter})")
        return  # Don't handle this session

    # Proceed with agent creation
    # ...
```

**Pros:**
- ✅ Independent scaling per agent type
- ✅ Resource isolation
- ✅ Easier monitoring

---

## Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # MongoDB for configurations
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  # Support agent workers
  support-worker:
    image: your-agent:latest
    command: uv run python examples/multi_agent_worker.py start
    environment:
      - LIVEKIT_URL=${LIVEKIT_URL}
      - LIVEKIT_API_KEY=${LIVEKIT_API_KEY}
      - LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}
      - MONGO_URI=mongodb://mongodb:27017
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
      - AGENT_FILTER=support
    deploy:
      replicas: 3

  # Sales agent workers
  sales-worker:
    image: your-agent:latest
    command: uv run python examples/multi_agent_worker.py start
    environment:
      - LIVEKIT_URL=${LIVEKIT_URL}
      - LIVEKIT_API_KEY=${LIVEKIT_API_KEY}
      - LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}
      - MONGO_URI=mongodb://mongodb:27017
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
      - AGENT_FILTER=sales
    deploy:
      replicas: 5  # More sales workers

volumes:
  mongodb_data:
```

### Kubernetes

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-worker
spec:
  replicas: 10  # 10 workers total
  template:
    spec:
      containers:
      - name: agent-worker
        image: your-agent:latest
        command: ["uv", "run", "python", "examples/multi_agent_worker.py", "start"]
        env:
        - name: LIVEKIT_URL
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: url
        - name: MONGO_URI
          value: "mongodb://mongodb-service:27017"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Summary

### What You Can Do

✅ **Run multiple agent types** from one worker
✅ **Route users** to different agents automatically
✅ **Scale independently** per agent type
✅ **Configure dynamically** without code changes
✅ **Monitor separately** per agent type
✅ **A/B test** different agent configurations

### Quick Command Reference

```bash
# Setup
uv run python examples/setup_agents.py

# Run multi-agent worker
uv run python examples/multi_agent_worker.py dev

# Run original PTE agent
uv run src/agent.py dev

# Start admin API
uv run python src/api/admin_api.py
```

### Frontend Metadata

```javascript
// Support
{ purpose: 'support', user_id: '...' }

// Sales
{ purpose: 'sales', user_id: '...' }

// Onboarding
{ purpose: 'onboarding', user_id: '...' }

// Interview
{ purpose: 'interview', user_id: '...' }

// Explicit
{ agent_id: 'support-agent-v1', user_id: '...' }
```

---

**You now have a complete multi-agent system!** 🎉

Each agent type has its own:
- Specialized prompts
- Custom schema fields
- Unique behavior
- Independent configuration

All running from the **same codebase** with **zero code changes** needed to add new agent types!
```
