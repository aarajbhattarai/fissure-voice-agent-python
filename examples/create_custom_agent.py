"""
Example: Create a custom sales agent configuration via API.

Usage:
    # Start the admin API first
    uv run python src/api/admin_api.py

    # Then run this script
    uv run python examples/create_custom_agent.py
"""

import asyncio

import httpx


async def create_sales_agent():
    """Create a sales agent configuration."""

    config = {
        "agent_id": "sales-agent-v1",
        "agent_type": "sales",
        "version": "1.0.0",
        "enabled": True,
        "prompt_config": {
            "template_id": "sales_conversation",
            "instructions": """You are an expert sales representative for our SaaS product.
Your goal is to understand customer needs, demonstrate value, and guide them towards a purchase decision.
Be consultative, not pushy. Focus on solving their problems.""",
            "dynamic_vars": {},
            "examples": [],
        },
        "schema_config": {
            "schema_id": "sales_turn_json_v1",
            "schema_version": "1.0.0",
            "fields": [
                {
                    "field_name": "voice_instructions",
                    "field_type": "str",
                    "description": "TTS tone and pace instructions",
                    "required": False,
                    "metadata": {"use_in_tts": True, "hidden": False},
                },
                {
                    "field_name": "system_response",
                    "field_type": "str",
                    "description": "The sales rep's spoken response",
                    "required": True,
                    "metadata": {"use_in_tts": False, "hidden": False},
                },
                {
                    "field_name": "customer_intent",
                    "field_type": "Literal['discovery', 'objection', 'interest', 'ready_to_buy', 'not_interested']",
                    "description": "Detected customer intent",
                    "required": False,
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
                {
                    "field_name": "pain_points",
                    "field_type": "list[str]",
                    "description": "Identified customer pain points",
                    "required": False,
                    "default": [],
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
                {
                    "field_name": "qualification_score",
                    "field_type": "int",
                    "description": "Lead qualification score (0-100)",
                    "required": False,
                    "constraints": {"ge": 0, "le": 100},
                    "metadata": {"use_in_tts": False, "hidden": False},
                },
                {
                    "field_name": "next_action",
                    "field_type": "str",
                    "description": "Recommended next action in sales process",
                    "required": False,
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
            ],
        },
        "llm_config": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0.8,
            "max_tokens": None,
            "supports_structured_output": True,
        },
        "tts_config": {
            "provider": "deepgram",
            "model": None,
            "voice": None,
            "instruction_field": "voice_instructions",
            "streaming": True,
        },
        "tracing_config": {
            "enabled": True,
            "provider": "langfuse",
            "sample_rate": 1.0,
        },
        "pipeline_config": {
            "vad_enabled": True,
            "noise_cancellation": True,
            "transcription_enabled": True,
        },
        "tenant_config": {
            "tenant_id": "default",
            "isolation_level": "session",
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/agents/config", json=config
            )
            response.raise_for_status()

            result = response.json()
            print("✅ Sales agent created successfully!")
            print(f"   Agent ID: {result['agent_id']}")
            print(f"   Config ID: {result['config_id']}")

        except httpx.HTTPStatusError as e:
            print(f"❌ Failed to create agent: {e}")
            print(f"   Response: {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def add_budget_field():
    """Add a budget tracking field to the sales agent schema."""

    field = {
        "field_name": "estimated_budget",
        "field_type": "str",
        "description": "Customer's estimated budget range",
        "required": False,
        "metadata": {"use_in_tts": False, "hidden": True},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/agents/config/sales-agent-v1/schema/fields",
                json=field,
            )
            response.raise_for_status()

            result = response.json()
            print(f"✅ Added field: {result['field_name']}")

        except httpx.HTTPStatusError as e:
            print(f"❌ Failed to add field: {e}")
            print(f"   Response: {e.response.text}")


async def toggle_tracing():
    """Toggle tracing for the sales agent."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/agents/config/sales-agent-v1/tracing/toggle",
                params={"enabled": False},
            )
            response.raise_for_status()

            result = response.json()
            print(f"✅ Tracing toggled: {result}")

        except httpx.HTTPStatusError as e:
            print(f"❌ Failed to toggle tracing: {e}")


async def list_all_agents():
    """List all agent configurations."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get("http://localhost:8000/api/v1/agents/list")
            response.raise_for_status()

            result = response.json()
            print(f"\n📋 Total agents: {result['count']}")
            for agent in result["agents"]:
                print(f"   - {agent['agent_id']} ({agent['agent_type']}) - v{agent['version']}")

        except httpx.HTTPStatusError as e:
            print(f"❌ Failed to list agents: {e}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Creating Sales Agent Configuration")
    print("=" * 60)

    await create_sales_agent()
    await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("Adding Budget Field to Schema")
    print("=" * 60)

    await add_budget_field()
    await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("Toggling Tracing")
    print("=" * 60)

    await toggle_tracing()
    await asyncio.sleep(1)

    print("\n" + "=" * 60)
    print("Listing All Agents")
    print("=" * 60)

    await list_all_agents()


if __name__ == "__main__":
    asyncio.run(main())
