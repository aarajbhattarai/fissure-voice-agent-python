"""
Setup script to create all agent configurations in MongoDB.

This script creates configurations for:
- Support Agent
- Sales Agent
- Onboarding Agent
- Interview Agent (if using dynamic system)

Usage:
    uv run python examples/setup_agents.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator import ConfigurationStore
from examples.agent_configurations import (
    SUPPORT_AGENT_CONFIG,
    SALES_AGENT_CONFIG,
    ONBOARDING_AGENT_CONFIG,
)

load_dotenv()


async def main():
    """Setup all agent configurations."""

    print("\n" + "=" * 60)
    print("AGENT CONFIGURATION SETUP")
    print("=" * 60 + "\n")

    # Check MongoDB connection
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("❌ Error: MONGO_URI not set in environment")
        print("   Please add to .env.local: MONGO_URI=mongodb://localhost:27017")
        return

    print(f"Connecting to MongoDB: {mongo_uri}")

    # Initialize configuration store
    try:
        config_store = ConfigurationStore(
            connection_string=mongo_uri, database="agent_configs"
        )

        await config_store.initialize()
        print("✅ Connected to MongoDB\n")

    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("   Make sure MongoDB is running:")
        print("   docker run -d -p 27017:27017 --name mongodb mongo:latest")
        return

    # Agent configurations to create
    agents = [
        ("Support Agent", SUPPORT_AGENT_CONFIG),
        ("Sales Agent", SALES_AGENT_CONFIG),
        ("Onboarding Agent", ONBOARDING_AGENT_CONFIG),
    ]

    print("Creating agent configurations...\n")

    created = 0
    existing = 0
    failed = 0

    for name, config in agents:
        try:
            config_id = await config_store.create_agent_config(config)
            print(f"✅ {name}")
            print(f"   Agent ID: {config['agent_id']}")
            print(f"   Schema: {config['schema_config']['schema_id']}")
            print(f"   Fields: {len(config['schema_config']['fields'])}")
            print(f"   Config ID: {config_id}\n")
            created += 1

        except ValueError as e:
            print(f"⚠️  {name}")
            print(f"   Already exists: {config['agent_id']}\n")
            existing += 1

        except Exception as e:
            print(f"❌ {name}")
            print(f"   Error: {e}\n")
            failed += 1

    # Summary
    print("=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"Created: {created}")
    print(f"Already existed: {existing}")
    print(f"Failed: {failed}")
    print(f"Total: {len(agents)}")
    print()

    if created > 0 or existing > 0:
        print("✅ Agent configurations ready!")
        print("\nNext steps:")
        print("1. Start the multi-agent worker:")
        print("   uv run python examples/multi_agent_worker.py dev")
        print()
        print("2. Or use the Admin API:")
        print("   uv run python src/api/admin_api.py")
        print("   Visit http://localhost:8000/docs")
        print()
        print("3. Connect from frontend with metadata:")
        print("   { purpose: 'support' }  → Support Agent")
        print("   { purpose: 'sales' }    → Sales Agent")
        print("   { purpose: 'onboarding' } → Onboarding Agent")

    # List all configured agents
    print("\n" + "=" * 60)
    print("ALL CONFIGURED AGENTS")
    print("=" * 60 + "\n")

    all_configs = await config_store.list_agent_configs(enabled_only=True, limit=100)

    if not all_configs:
        print("No agents configured yet.")
    else:
        for config in all_configs:
            print(f"📋 {config['agent_id']}")
            print(f"   Type: {config['agent_type']}")
            print(f"   Version: {config['version']}")
            print(f"   Enabled: {config['enabled']}")
            print(f"   Schema: {config['schema_config']['schema_id']}")
            print()

    await config_store.close()


if __name__ == "__main__":
    asyncio.run(main())
