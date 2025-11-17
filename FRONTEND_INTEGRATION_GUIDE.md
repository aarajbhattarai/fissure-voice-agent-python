# How Your PTE Interview Agent Works - Complete Guide

## Overview

This document explains how your current PTE Interview Agent system works, how the frontend connects to the backend, and how to integrate the new dynamic agent architecture with user data and conversation summaries.

---

## Table of Contents

1. [Current System Architecture](#current-system-architecture)
2. [How the Connection Works](#how-the-connection-works)
3. [Data Flow](#data-flow)
4. [Frontend Integration](#frontend-integration)
5. [Integrating Dynamic Agent Features](#integrating-dynamic-agent-features)
6. [Passing User Data from Frontend](#passing-user-data-from-frontend)
7. [Receiving Summaries on Frontend](#receiving-summaries-on-frontend)
8. [Complete Integration Example](#complete-integration-example)

---

## Current System Architecture

### Backend (What You're Running)

```
┌─────────────────────────────────────────────────────────┐
│  Terminal: uv run src/agent.py dev                     │
│                                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │  LiveKit Agent Worker                           │   │
│  │  - Listens for room join events                │   │
│  │  - Creates PTEInterviewAgent per session        │   │
│  │  - Processes audio/video streams               │   │
│  │  - Sends structured data via RPC               │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       ↕
                LiveKit Server
                (Cloud or Self-Hosted)
                       ↕
┌─────────────────────────────────────────────────────────┐
│  Frontend (Your React/Next.js App)                     │
│  - LiveKit Client SDK                                   │
│  - Connects to LiveKit room                            │
│  - Sends/receives audio/video                          │
│  - Receives structured data (RPC)                      │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Backend Agent Worker** (`uv run src/agent.py dev`)
   - Runs continuously
   - Waits for room join events from LiveKit Server
   - Creates one PTEInterviewAgent instance per user session

2. **LiveKit Server** (Cloud or Self-Hosted)
   - Central signaling server
   - Routes audio/video streams
   - Routes RPC messages (structured data)
   - Manages rooms and participants

3. **Frontend Application** (Your Web App)
   - User interface
   - LiveKit client SDK
   - Creates/joins rooms
   - Displays interview interface

---

## How the Connection Works

### Step-by-Step Flow

#### 1. **Backend Starts**

```bash
# You run this command
uv run src/agent.py dev
```

What happens:
```python
# src/agent.py:560
cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
```

- Agent worker connects to LiveKit Server using `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- Worker registers as available
- **Waits for room join events**

#### 2. **User Opens Frontend**

User visits your frontend application (e.g., `https://yourapp.com/interview`)

#### 3. **Frontend Creates/Joins Room**

Your frontend code (pseudocode):
```javascript
// Frontend creates a LiveKit access token
const response = await fetch('/api/get-token', {
  method: 'POST',
  body: JSON.stringify({
    roomName: 'interview-room-123',
    participantName: 'John Doe',
    metadata: {
      user_id: 'user-123',
      name: 'John Doe',
      email: 'john@example.com'
    }
  })
});

const { token } = await response.json();

// Connect to LiveKit room
const room = new Room();
await room.connect(LIVEKIT_URL, token);

// Enable microphone
await room.localParticipant.setMicrophoneEnabled(true);
```

#### 4. **LiveKit Server Notifies Backend**

- LiveKit Server sees new participant joined room
- Triggers agent worker via webhook/event
- Backend's `entrypoint()` function is called

```python
# src/agent.py:494
async def entrypoint(ctx: agents.JobContext) -> None:
    # Called when user joins room
    await ctx.connect()  # Connect to the room

    # Create PTEInterviewAgent for this user
    agent = PTEInterviewAgent(
        room=ctx.room,
        user_details=user_details,
        user_id="user-Aaraj",  # Currently hardcoded
    )

    # Start agent session
    await session.start(room=ctx.room, agent=agent, ...)
```

#### 5. **Agent Says Welcome**

```python
# src/agent.py:102
async def on_enter(self) -> None:
    welcome_message = f"Welcome to your PTE Interview Practice Session..."
    await self.session.say(text=welcome_message)
```

- Agent speaks welcome message
- Audio is streamed to frontend via LiveKit
- User hears the agent speaking

#### 6. **Bidirectional Communication**

Now the conversation loop begins:

```
User speaks → Microphone
    ↓
Frontend sends audio → LiveKit Server
    ↓
Backend receives audio → STT (Deepgram)
    ↓
Text → LLM (OpenAI GPT-5-nano)
    ↓
Response → TTS (Deepgram)
    ↓
Audio → LiveKit Server → Frontend → Speaker
```

---

## Data Flow

### Audio/Video Flow

```
┌──────────┐         ┌──────────────┐         ┌──────────┐
│ Frontend │ ◄─────► │ LiveKit      │ ◄─────► │ Backend  │
│          │  WebRTC │ Server       │  WebRTC │ Agent    │
│ Speaker/ │         │              │         │          │
│ Mic      │         │              │         │ STT/TTS  │
└──────────┘         └──────────────┘         └──────────┘
```

### Structured Data Flow (RPC)

Your agent already sends structured data to frontend via **Data Channel** (RPC):

```python
# Backend sends data
await self.data_streamer.send_structured_data(
    topic="llm-transcription",
    data={
        "type": "llm_chunk",
        "content": "Hello, tell me about yourself",
        "session_id": self.session_id,
        ...
    }
)
```

```javascript
// Frontend receives data
room.on('dataReceived', (payload, participant, kind, topic) => {
  if (topic === 'llm-transcription') {
    const data = JSON.parse(payload);
    console.log('Agent said:', data.content);
    // Update UI with agent's response
  }
});
```

### Current RPC Topics Your Agent Uses

| Topic | Purpose | When Sent |
|-------|---------|-----------|
| `llm-transcription` | Streaming agent responses (chunks) | During agent response |
| `llm-response-complete` | Complete agent response | After agent finishes speaking |
| `llm-structured-response` | Full structured data from LLM | After LLM generates response |
| `interview-metrics` | Real-time metrics (optional) | When you call `send_interview_metrics()` |
| `interview-feedback` | Structured feedback (optional) | When you call `send_structured_feedback()` |

---

## Frontend Integration

### Basic Frontend Setup

Your frontend likely looks something like this:

```javascript
// 1. Get access token from your backend API
const getToken = async (roomName, userName, metadata) => {
  const response = await fetch('/api/livekit/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ roomName, userName, metadata })
  });
  return response.json();
};

// 2. Connect to LiveKit room
const connectToInterview = async () => {
  const { token } = await getToken(
    'interview-room-' + Date.now(),
    'John Doe',
    { user_id: 'user-123' }
  );

  const room = new Room();

  // Listen for data from agent
  room.on('dataReceived', handleAgentData);

  // Listen for agent speaking
  room.on('trackSubscribed', handleTrackSubscribed);

  // Connect
  await room.connect(LIVEKIT_URL, token);

  // Enable microphone
  await room.localParticipant.setMicrophoneEnabled(true);
};

// 3. Handle agent data
const handleAgentData = (payload, participant, kind, topic) => {
  const data = JSON.parse(new TextDecoder().decode(payload));

  switch(topic) {
    case 'llm-transcription':
      // Show agent's response in real-time
      updateTranscript(data.content);
      break;

    case 'llm-response-complete':
      // Agent finished speaking
      console.log('Agent response complete');
      break;

    case 'interview-metrics':
      // Update metrics display
      updateMetrics(data);
      break;
  }
};

// 4. Handle agent audio
const handleTrackSubscribed = (track, publication, participant) => {
  if (track.kind === 'audio' && participant.identity.includes('agent')) {
    // Play agent audio
    const audioElement = track.attach();
    document.body.appendChild(audioElement);
  }
};
```

### Your Backend API (Token Generation)

You need a backend API endpoint to generate LiveKit tokens:

```javascript
// Example: /api/livekit/token endpoint (Next.js API route)
import { AccessToken } from 'livekit-server-sdk';

export default async function handler(req, res) {
  const { roomName, userName, metadata } = req.body;

  const at = new AccessToken(
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET,
    {
      identity: userName,
      metadata: JSON.stringify(metadata), // Pass user data here
    }
  );

  at.addGrant({
    roomJoin: true,
    room: roomName,
    canPublish: true,
    canSubscribe: true,
  });

  const token = at.toJwt();

  res.status(200).json({ token });
}
```

---

## Integrating Dynamic Agent Features

### Option 1: Keep Using PTEInterviewAgent (Easiest)

You can update your existing `PTEInterviewAgent` to use the new user data and summary features without changing to the full dynamic system.

#### Update src/agent.py

```python
# Add imports
from agents import (
    UserData,
    SessionData,
    ConversationHistory,
    ConversationSummarizer,
    create_session_storage
)

class PTEInterviewAgent(Agent):
    def __init__(self, room, user_id, user_details=None, instructions=...):
        super().__init__(...)

        # Add user data management
        self.user_data = UserData(
            user_id=user_id,
            name=user_details.get("name"),
            email=user_details.get("email"),
            institution=user_details.get("institution"),
            field_of_study=user_details.get("field_of_study"),
            age=user_details.get("age"),
        )

        # Add session data
        self.session_data = SessionData(
            session_id=self.session_id,
            user_data=self.user_data,
            tenant_id="default",
            agent_id="pte-interview-agent",
        )

        # Add conversation tracking
        self.conversation_history = ConversationHistory(
            session_id=self.session_id,
            user_id=user_id
        )

        # Add summarizer
        self.summarizer = ConversationSummarizer(
            llm_provider="openai",
            model="gpt-5-nano"
        )

    async def stt_node(self, audio, model_settings):
        # Track user messages
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcribed_text = event.alternatives[0].text

                # Track in conversation history
                self.conversation_history.add_turn(
                    speaker="user",
                    message=transcribed_text
                )

            yield event

    async def transcription_node(self, text, model_settings):
        # Track agent messages
        acc_text = ""

        async def text_processor():
            nonlocal acc_text
            async for delta in text:
                acc_text += delta
                yield delta

        # Process
        processed = process_structured_output(
            text_processor(),
            force_structured=self._is_llm_response
        )

        result = Agent.default.transcription_node(self, processed, model_settings)

        async for chunk in result:
            yield chunk

        # Track agent message
        if acc_text:
            self.conversation_history.add_turn(
                speaker="agent",
                message=acc_text
            )

    async def close(self):
        # End session
        self.session_data.end_session(status="completed")

        # Generate summary
        transcript = self.conversation_history.get_full_transcript()
        summary_data = await self.summarizer.generate_summary(
            transcript=transcript,
            user_data=self.user_data.to_dict()
        )

        # Send summary to frontend
        await self.data_streamer.send_structured_data(
            topic="conversation-summary",
            data={
                "summary": summary_data["summary"],
                "total_turns": self.conversation_history.total_turns,
                "duration": self.session_data.duration_seconds,
            }
        )

        # Save to MongoDB (optional)
        # storage = await create_session_storage("mongodb://localhost:27017")
        # await storage.save_session(self.session_data)
        # await storage.save_conversation(self.conversation_history)
        # await storage.save_summary(...)

        await super().close()
```

### Option 2: Use Full Dynamic Agent Architecture

Replace `PTEInterviewAgent` with `BaseAgent` and use dynamic configuration.

See `examples/dynamic_agent_entrypoint.py` for complete implementation.

---

## Passing User Data from Frontend

### Method 1: Via Room Metadata (Recommended)

When user joins room, pass user data in room metadata:

```javascript
// Frontend: When creating/joining room
const roomMetadata = {
  agent_id: "pte-interview-agent-v1",
  user_id: "user-123",
  user_details: {
    name: "John Doe",
    email: "john@example.com",
    age: 25,
    nationality: "USA",
    institution: "MIT",
    field_of_study: "Computer Science"
  }
};

// Include in token request
const { token } = await getToken(
  roomName,
  userName,
  roomMetadata  // Pass metadata
);
```

```javascript
// Backend API: When generating token
const at = new AccessToken(..., {
  identity: userName,
  metadata: JSON.stringify(metadata), // Include user data
});
```

```python
# Backend agent: Read from room metadata
async def entrypoint(ctx: agents.JobContext) -> None:
    import json

    # Extract user data from room metadata
    room_metadata = json.loads(ctx.room.metadata or "{}")
    user_id = room_metadata.get("user_id", "anonymous")
    user_details = room_metadata.get("user_details", {})

    # Create agent with user data
    agent = PTEInterviewAgent(
        room=ctx.room,
        user_id=user_id,
        user_details=user_details  # Use real data now!
    )

    await session.start(room=ctx.room, agent=agent, ...)
```

### Method 2: Via Participant Attributes

```javascript
// Frontend: Set participant attributes
await room.localParticipant.setAttributes({
  user_id: 'user-123',
  name: 'John Doe',
  email: 'john@example.com',
  institution: 'MIT'
});
```

```python
# Backend: Read from participant attributes
async def entrypoint(ctx: agents.JobContext) -> None:
    # Wait for remote participant
    while not ctx.room.remote_participants:
        await asyncio.sleep(0.1)

    participant = list(ctx.room.remote_participants.values())[0]
    user_data = participant.attributes  # Access attributes

    agent = PTEInterviewAgent(
        room=ctx.room,
        user_id=user_data.get("user_id"),
        user_details=user_data
    )
```

---

## Receiving Summaries on Frontend

### Listen for Summary Topic

```javascript
// Frontend: Listen for conversation summary
room.on('dataReceived', (payload, participant, kind, topic) => {
  if (topic === 'conversation-summary') {
    const data = JSON.parse(new TextDecoder().decode(payload));

    console.log('📝 Conversation Summary:');
    console.log(data.summary);
    console.log('Total turns:', data.total_turns);
    console.log('Duration:', data.duration, 'seconds');

    // Show summary in UI
    displaySummary(data);
  }

  if (topic === 'session-complete') {
    const data = JSON.parse(new TextDecoder().decode(payload));

    console.log('✅ Session Complete');
    console.log('Summary available:', data.summary_available);

    // Update UI
    showSessionComplete(data);
  }
});

// Display summary in UI
const displaySummary = (summaryData) => {
  const summaryDiv = document.getElementById('interview-summary');

  summaryDiv.innerHTML = `
    <h2>Interview Summary</h2>
    <div class="summary-content">
      ${summaryData.summary}
    </div>
    <div class="summary-stats">
      <span>Total Exchanges: ${summaryData.total_turns}</span>
      <span>Duration: ${Math.round(summaryData.duration)}s</span>
    </div>
  `;

  summaryDiv.style.display = 'block';
};
```

### React Component Example

```jsx
import { useEffect, useState } from 'react';
import { Room } from 'livekit-client';

function InterviewSession() {
  const [summary, setSummary] = useState(null);
  const [sessionComplete, setSessionComplete] = useState(false);

  useEffect(() => {
    const room = new Room();

    // Listen for agent data
    room.on('dataReceived', (payload, participant, kind, topic) => {
      const data = JSON.parse(new TextDecoder().decode(payload));

      if (topic === 'conversation-summary') {
        setSummary(data);
      }

      if (topic === 'session-complete') {
        setSessionComplete(true);
      }
    });

    // Connect to room
    connectToRoom(room);

    return () => room.disconnect();
  }, []);

  return (
    <div>
      {sessionComplete && summary && (
        <div className="summary-card">
          <h2>Interview Summary</h2>
          <p>{summary.summary}</p>
          <div className="stats">
            <span>Turns: {summary.total_turns}</span>
            <span>Duration: {summary.duration}s</span>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Complete Integration Example

### 1. Frontend: Start Interview with User Data

```javascript
// pages/interview.js
import { useState } from 'react';
import { Room } from 'livekit-client';

export default function InterviewPage() {
  const [room, setRoom] = useState(null);
  const [summary, setSummary] = useState(null);

  const startInterview = async (userData) => {
    // 1. Get token with user data
    const response = await fetch('/api/livekit/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        roomName: `interview-${Date.now()}`,
        userName: userData.name,
        metadata: {
          user_id: userData.id,
          user_details: {
            name: userData.name,
            email: userData.email,
            institution: userData.institution,
            field_of_study: userData.fieldOfStudy,
            age: userData.age,
          }
        }
      })
    });

    const { token } = await response.json();

    // 2. Create room and listen for data
    const newRoom = new Room();

    newRoom.on('dataReceived', (payload, participant, kind, topic) => {
      const data = JSON.parse(new TextDecoder().decode(payload));

      switch(topic) {
        case 'conversation-summary':
          setSummary(data);
          break;

        case 'llm-transcription':
          // Handle real-time transcription
          console.log('Agent:', data.content);
          break;
      }
    });

    // 3. Connect
    await newRoom.connect(process.env.NEXT_PUBLIC_LIVEKIT_URL, token);

    // 4. Enable microphone
    await newRoom.localParticipant.setMicrophoneEnabled(true);

    setRoom(newRoom);
  };

  return (
    <div>
      <button onClick={() => startInterview({
        id: 'user-123',
        name: 'John Doe',
        email: 'john@example.com',
        institution: 'MIT',
        fieldOfStudy: 'Computer Science',
        age: 25
      })}>
        Start Interview
      </button>

      {summary && (
        <div className="summary">
          <h2>Summary</h2>
          <p>{summary.summary}</p>
        </div>
      )}
    </div>
  );
}
```

### 2. Backend API: Token Generation

```javascript
// pages/api/livekit/token.js
import { AccessToken } from 'livekit-server-sdk';

export default async function handler(req, res) {
  const { roomName, userName, metadata } = req.body;

  const at = new AccessToken(
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET,
    {
      identity: userName,
      metadata: JSON.stringify(metadata),
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

### 3. Backend Agent: Updated Entrypoint

```python
# src/agent.py
async def entrypoint(ctx: agents.JobContext) -> None:
    import json

    # Get user data from room metadata
    room_metadata = json.loads(ctx.room.metadata or "{}")
    user_details = room_metadata.get("user_details", {})
    user_id = room_metadata.get("user_id", "anonymous")

    logger.info(f"Starting interview for {user_details.get('name', 'Unknown')}")

    # Setup tracing
    trace_provider = setup_langfuse(metadata={"langfuse.session.id": ctx.room.name})
    ctx.add_shutdown_callback(lambda: trace_provider.force_flush())

    # Connect to room
    await ctx.connect()

    # Create agent with real user data
    agent = PTEInterviewAgent(
        room=ctx.room,
        user_id=user_id,
        user_details=user_details  # Real user data!
    )

    # Start session
    session = AgentSession(vad=ctx.proc.userdata["vad"], use_tts_aligned_transcript=True)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            audio_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=True,
            transcription_enabled=True
        ),
    )
```

---

## Summary

### How It Works

1. **Frontend** creates room and passes user data in metadata
2. **LiveKit Server** notifies backend agent worker
3. **Backend** creates PTEInterviewAgent with user data
4. **Conversation** happens bidirectionally via LiveKit
5. **Agent** tracks conversation and generates summary
6. **Summary** sent to frontend via RPC topic `conversation-summary`
7. **Frontend** displays summary to user

### Key Points

✅ **No hardcoded user data** - Pass from frontend
✅ **Real-time structured data** - Via LiveKit RPC topics
✅ **Automatic conversation tracking** - Built into agent
✅ **LLM-powered summaries** - Generated on session end
✅ **Persistent storage** - Optional MongoDB integration

### Next Steps

1. Update `src/agent.py` to read user data from room metadata
2. Update frontend to pass user data when joining room
3. Add listener for `conversation-summary` topic on frontend
4. Test end-to-end with real user data
5. (Optional) Add MongoDB for persistent storage

---

All the pieces are already in place - you just need to connect them! 🚀
