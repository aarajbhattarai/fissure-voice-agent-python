"""
Example: Using user data management and conversation summaries.

Demonstrates:
1. Creating user data with data classes
2. Managing session data
3. Tracking conversation history
4. Generating conversation summaries
5. Storing and retrieving session data

Usage:
    uv run python examples/user_data_session_example.py
"""

import asyncio
import os

from dotenv import load_dotenv

# Add src to path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.user_data import (
    ConversationHistory,
    SessionData,
    UserData,
    UserDetailsModel,
)
from agents.conversation_summary import ConversationSummarizer
from agents.session_storage import create_session_storage

load_dotenv()


async def example_user_data_management():
    """Demonstrate user data management."""
    print("\n" + "=" * 60)
    print("Example 1: User Data Management")
    print("=" * 60 + "\n")

    # Create user data using dataclass (fast, no validation)
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

    print("User Data created:")
    print(f"  Name: {user_data.name}")
    print(f"  Email: {user_data.email}")
    print(f"  Institution: {user_data.institution}")
    print(f"  Field of Study: {user_data.field_of_study}")

    # Update user data
    user_data.update(phone="+1-555-0123")
    print(f"\nUpdated phone: {user_data.phone}")

    # Convert to dictionary
    user_dict = user_data.to_dict()
    print(f"\nUser data as dict: {len(user_dict)} fields")

    # Create with validation using Pydantic model
    print("\n\nUsing Pydantic model with validation:")
    try:
        validated_user = UserDetailsModel(
            user_id="user-67890",
            name="Jane Smith",
            email="jane.smith@example.com",
            age=30,
            language_preference="en",
        )
        print(f"✅ Validated user: {validated_user.name}")
    except Exception as e:
        print(f"❌ Validation error: {e}")


async def example_session_management():
    """Demonstrate session data management."""
    print("\n" + "=" * 60)
    print("Example 2: Session Data Management")
    print("=" * 60 + "\n")

    # Create user data
    user_data = UserData(
        user_id="user-12345",
        name="John Doe",
        email="john.doe@example.com",
    )

    # Create session data
    session_data = SessionData(
        session_id="session-abc123",
        user_data=user_data,
        tenant_id="customer-xyz",
        agent_id="interview-agent-v1",
    )

    print("Session created:")
    print(f"  Session ID: {session_data.session_id}")
    print(f"  User: {session_data.user_data.name}")
    print(f"  Agent: {session_data.agent_id}")
    print(f"  Status: {session_data.status}")

    # Simulate session ending
    await asyncio.sleep(2)  # Simulate 2 seconds of conversation
    session_data.end_session(status="completed")

    print(f"\nSession ended:")
    print(f"  Status: {session_data.status}")
    print(f"  Duration: {session_data.duration_seconds:.2f}s")


async def example_conversation_tracking():
    """Demonstrate conversation history tracking."""
    print("\n" + "=" * 60)
    print("Example 3: Conversation History Tracking")
    print("=" * 60 + "\n")

    # Create conversation history
    conversation = ConversationHistory(
        session_id="session-abc123", user_id="user-12345"
    )

    # Simulate conversation
    conversation.add_turn(
        speaker="agent",
        message="Welcome to your PTE interview practice session. Let's begin!",
    )

    conversation.add_turn(
        speaker="user", message="Thank you! I'm ready to start."
    )

    conversation.add_turn(
        speaker="agent",
        message="Great! Could you please tell me about your educational background?",
    )

    conversation.add_turn(
        speaker="user",
        message="I completed my Bachelor's degree in Computer Science from MIT in 2023.",
    )

    conversation.add_turn(
        speaker="agent",
        message="Excellent. And what are your plans for graduate studies?",
    )

    conversation.add_turn(
        speaker="user",
        message="I plan to pursue a Master's in AI at Stanford University.",
    )

    print(f"Conversation tracked:")
    print(f"  Total turns: {conversation.total_turns}")
    print(f"  User messages: {len(conversation.get_user_messages())}")
    print(f"  Agent messages: {len(conversation.get_agent_messages())}")

    print(f"\nFull transcript:")
    print("-" * 60)
    print(conversation.get_full_transcript())
    print("-" * 60)

    return conversation


async def example_summary_generation(conversation: ConversationHistory):
    """Demonstrate conversation summary generation."""
    print("\n" + "=" * 60)
    print("Example 4: Conversation Summary Generation")
    print("=" * 60 + "\n")

    # Create summarizer
    summarizer = ConversationSummarizer(
        llm_provider="openai", model="gpt-5-nano"
    )

    # Generate summary
    print("Generating summary... (this may take a few seconds)")

    transcript = conversation.get_full_transcript()

    try:
        summary_data = await summarizer.generate_summary(
            transcript=transcript,
            user_data={
                "name": "John Doe",
                "institution": "MIT",
                "field_of_study": "Computer Science",
            },
            session_metadata={
                "session_id": conversation.session_id,
                "total_turns": conversation.total_turns,
            },
        )

        print("\n✅ Summary generated successfully!")
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(summary_data["summary"])
        print("=" * 60)

        print(f"\nTokens used: {summary_data.get('tokens_used', 'N/A')}")
        print(f"Model: {summary_data.get('model', 'N/A')}")

        # Generate quick summary
        print("\n\nGenerating quick summary...")
        quick_summary = await summarizer.generate_quick_summary(transcript, max_length=200)
        print(f"\n✅ Quick summary: {quick_summary}")

        return summary_data

    except Exception as e:
        print(f"\n❌ Failed to generate summary: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env.local file")
        return None


async def example_storage():
    """Demonstrate storing and retrieving session data."""
    print("\n" + "=" * 60)
    print("Example 5: Session Storage")
    print("=" * 60 + "\n")

    # Create storage
    print("Connecting to MongoDB...")
    try:
        storage = await create_session_storage(
            connection_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            database="agent_data_examples",
        )
        print("✅ Connected to MongoDB")

        # Create sample session
        user_data = UserData(
            user_id="user-storage-example",
            name="Alice Johnson",
            email="alice@example.com",
        )

        session_data = SessionData(
            session_id="session-storage-example",
            user_data=user_data,
            tenant_id="demo-tenant",
            agent_id="interview-agent-v1",
        )

        # Simulate session
        await asyncio.sleep(1)
        session_data.end_session(status="completed")

        # Save session
        doc_id = await storage.save_session(session_data)
        print(f"\n✅ Session saved with ID: {doc_id}")

        # Retrieve session
        retrieved = await storage.get_session(session_data.session_id)
        print(f"✅ Session retrieved:")
        print(f"   User: {retrieved['user_data']['name']}")
        print(f"   Duration: {retrieved['duration_seconds']}s")

        # Save conversation
        conversation = ConversationHistory(
            session_id=session_data.session_id, user_id=user_data.user_id
        )
        conversation.add_turn(speaker="agent", message="Hello!")
        conversation.add_turn(speaker="user", message="Hi there!")

        await storage.save_conversation(conversation)
        print(f"\n✅ Conversation saved ({conversation.total_turns} turns)")

        # Retrieve conversation
        retrieved_conv = await storage.get_conversation(session_data.session_id)
        print(f"✅ Conversation retrieved:")
        print(f"   Turns: {retrieved_conv['total_turns']}")

        # Get user sessions
        user_sessions = await storage.get_user_sessions(user_data.user_id, limit=5)
        print(f"\n✅ Found {len(user_sessions)} sessions for user")

        # Get stats
        stats = await storage.get_session_stats(user_data.user_id)
        print(f"\n📊 Session statistics:")
        print(f"   Total sessions: {stats['total_sessions']}")
        print(f"   Avg duration: {stats['avg_duration_seconds']:.2f}s")

    except Exception as e:
        print(f"\n❌ Storage error: {e}")
        print("Make sure MongoDB is running and MONGO_URI is set")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("USER DATA & CONVERSATION SUMMARY EXAMPLES")
    print("=" * 60)

    # Run examples
    await example_user_data_management()
    await example_session_management()
    conversation = await example_conversation_tracking()

    # Generate summary (requires OpenAI API key)
    if os.getenv("OPENAI_API_KEY"):
        summary_data = await example_summary_generation(conversation)
    else:
        print("\n⚠️  Skipping summary generation (OPENAI_API_KEY not set)")

    # Storage example (requires MongoDB)
    if os.getenv("MONGO_URI"):
        await example_storage()
    else:
        print("\n⚠️  Skipping storage example (MONGO_URI not set)")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
