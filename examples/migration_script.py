"""
Migration script to convert existing PTEInterviewAgent to dynamic configuration.

Usage:
    uv run python examples/migration_script.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator.configuration_store import ConfigurationStore

load_dotenv()


async def migrate_pte_interview_agent():
    """Migrate PTEInterviewAgent to dynamic configuration."""

    # Initialize configuration store
    config_store = ConfigurationStore(
        connection_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        database="agent_configs",
    )

    await config_store.initialize()

    # Create configuration from existing hardcoded values
    config = {
        "agent_id": "pte-interview-agent-v1",
        "agent_type": "interview",
        "version": "1.0.0",
        "enabled": True,
        "prompt_config": {
            "template_id": "pte_interview",
            "instructions": """You are a strict US VISA Interview Officer specialized in assessing true, deserving and qualified students.
Simulate real life interview, and start directly with your assessment. Your intro has been done.""",
            "dynamic_vars": {},
            "examples": [],
        },
        "schema_config": {
            "schema_id": "interview_turn_json_v1",
            "schema_version": "1.0.0",
            "fields": [
                {
                    "field_name": "voice_instructions",
                    "field_type": "str",
                    "description": "Specific TTS directive for tone, pace, and emphasis",
                    "required": False,
                    "metadata": {"use_in_tts": True, "hidden": False},
                },
                {
                    "field_name": "system_response",
                    "field_type": "str",
                    "description": "The officer's spoken response",
                    "required": True,
                    "metadata": {"use_in_tts": False, "hidden": False},
                },
                {
                    "field_name": "internal_assessment",
                    "field_type": "str",
                    "description": "Private evaluation of interaction",
                    "required": False,
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
                {
                    "field_name": "analysis",
                    "field_type": "list[str]",
                    "description": "Behavioral and content analysis points",
                    "required": False,
                    "default": [],
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
                {
                    "field_name": "interview_stage",
                    "field_type": "Literal['document_check', 'background_inquiry', 'academic_assessment', 'financial_review', 'intent_evaluation', 'final_decision']",
                    "description": "Current phase of the interview process",
                    "required": False,
                    "metadata": {"use_in_tts": False, "hidden": False},
                },
                {
                    "field_name": "credibility_score",
                    "field_type": "int",
                    "description": "Running assessment score (1–10)",
                    "required": False,
                    "constraints": {"ge": 1, "le": 10},
                    "metadata": {"use_in_tts": False, "hidden": False},
                },
                {
                    "field_name": "red_flags",
                    "field_type": "list[str]",
                    "description": "Specific concerns identified",
                    "required": False,
                    "default": [],
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
                {
                    "field_name": "next_focus_area",
                    "field_type": "str",
                    "description": "Recommended next area to probe",
                    "required": False,
                    "metadata": {"use_in_tts": False, "hidden": True},
                },
            ],
        },
        "llm_config": {
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0.7,
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
        "stt_config": {
            "provider": "deepgram",
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

    try:
        config_id = await config_store.create_agent_config(config)
        print(f"✅ Migration complete!")
        print(f"   Agent ID: {config['agent_id']}")
        print(f"   Config ID: {config_id}")
        print(f"   Schema: {config['schema_config']['schema_id']}")
        print(f"   Fields: {len(config['schema_config']['fields'])}")
    except ValueError as e:
        print(f"❌ Migration failed: {e}")
        print("   Agent configuration may already exist.")
    finally:
        await config_store.close()


if __name__ == "__main__":
    asyncio.run(migrate_pte_interview_agent())
