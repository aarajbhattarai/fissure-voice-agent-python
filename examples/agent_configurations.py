"""
Example configurations for different agent types.

This module contains pre-configured agent definitions for:
- Support Agent
- Sales Agent
- Interview Agent
- Custom agents

Usage:
    from examples.agent_configurations import SUPPORT_AGENT_CONFIG
    await config_store.create_agent_config(SUPPORT_AGENT_CONFIG)
"""

# =======================
# SUPPORT AGENT
# =======================

SUPPORT_AGENT_CONFIG = {
    "agent_id": "support-agent-v1",
    "agent_type": "support",
    "version": "1.0.0",
    "enabled": True,

    "prompt_config": {
        "template_id": "customer_support",
        "instructions": """You are a helpful customer support agent for a SaaS product.

Your role:
- Listen carefully to customer issues
- Ask clarifying questions
- Provide clear, step-by-step solutions
- Escalate to human support when necessary
- Maintain a friendly, professional tone

Guidelines:
- Be empathetic and patient
- Provide specific solutions, not generic advice
- If you don't know, say so and offer to connect them with a specialist
- Keep track of the conversation context
- Summarize the issue and solution at the end

Always be helpful and professional.""",
        "dynamic_vars": {},
        "examples": [],
    },

    "schema_config": {
        "schema_id": "support_turn_v1",
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
                "description": "Support agent's response to customer",
                "required": True,
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "issue_category",
                "field_type": "Literal['technical', 'billing', 'account', 'feature_request', 'other']",
                "description": "Category of customer issue",
                "required": False,
                "metadata": {"use_in_tts": False, "hidden": True},
            },
            {
                "field_name": "sentiment",
                "field_type": "Literal['positive', 'neutral', 'frustrated', 'angry']",
                "description": "Customer sentiment",
                "required": False,
                "metadata": {"use_in_tts": False, "hidden": True},
            },
            {
                "field_name": "issue_resolved",
                "field_type": "bool",
                "description": "Whether issue is resolved",
                "required": False,
                "default": False,
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "escalation_needed",
                "field_type": "bool",
                "description": "Whether to escalate to human",
                "required": False,
                "default": False,
                "metadata": {"use_in_tts": False, "hidden": True},
            },
            {
                "field_name": "next_action",
                "field_type": "str",
                "description": "Recommended next step",
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

    "summary_config": {
        "enabled": True,
        "model": "gpt-5-nano",
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


# =======================
# SALES AGENT
# =======================

SALES_AGENT_CONFIG = {
    "agent_id": "sales-agent-v1",
    "agent_type": "sales",
    "version": "1.0.0",
    "enabled": True,

    "prompt_config": {
        "template_id": "sales_conversation",
        "instructions": """You are an expert sales representative for our SaaS product.

Your role:
- Understand customer needs and pain points
- Demonstrate how our product solves their problems
- Build rapport and trust
- Guide towards a purchase decision
- Handle objections professionally

Sales methodology:
1. Discovery - Understand their situation
2. Needs Analysis - Identify pain points
3. Solution Presentation - Show how we help
4. Objection Handling - Address concerns
5. Closing - Guide to next steps

Be consultative, not pushy. Focus on value, not features.""",
        "dynamic_vars": {},
        "examples": [],
    },

    "schema_config": {
        "schema_id": "sales_turn_v1",
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
                "description": "Sales agent's response",
                "required": True,
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "sales_stage",
                "field_type": "Literal['discovery', 'needs_analysis', 'presentation', 'objection_handling', 'closing']",
                "description": "Current stage in sales process",
                "required": False,
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "customer_intent",
                "field_type": "Literal['exploring', 'interested', 'objection', 'ready_to_buy', 'not_interested']",
                "description": "Customer's buying intent",
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
                "field_name": "objections_raised",
                "field_type": "list[str]",
                "description": "Customer objections",
                "required": False,
                "default": [],
                "metadata": {"use_in_tts": False, "hidden": True},
            },
            {
                "field_name": "next_action",
                "field_type": "str",
                "description": "Recommended next step",
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

    "summary_config": {
        "enabled": True,
        "model": "gpt-5-nano",
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


# =======================
# ONBOARDING AGENT
# =======================

ONBOARDING_AGENT_CONFIG = {
    "agent_id": "onboarding-agent-v1",
    "agent_type": "onboarding",
    "version": "1.0.0",
    "enabled": True,

    "prompt_config": {
        "template_id": "onboarding_guide",
        "instructions": """You are a friendly onboarding specialist helping new users get started.

Your role:
- Welcome new users warmly
- Guide them through setup step-by-step
- Answer questions patiently
- Celebrate their progress
- Ensure they understand key features

Guidelines:
- Use simple, non-technical language
- Give one instruction at a time
- Confirm understanding before moving forward
- Be encouraging and positive
- Offer help proactively

Make the onboarding experience delightful!""",
        "dynamic_vars": {},
        "examples": [],
    },

    "schema_config": {
        "schema_id": "onboarding_turn_v1",
        "schema_version": "1.0.0",
        "fields": [
            {
                "field_name": "voice_instructions",
                "field_type": "str",
                "description": "TTS tone - friendly and encouraging",
                "required": False,
                "metadata": {"use_in_tts": True, "hidden": False},
            },
            {
                "field_name": "system_response",
                "field_type": "str",
                "description": "Onboarding guide's response",
                "required": True,
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "onboarding_step",
                "field_type": "Literal['welcome', 'account_setup', 'profile_creation', 'feature_tour', 'first_action', 'completion']",
                "description": "Current onboarding step",
                "required": False,
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "completion_percentage",
                "field_type": "int",
                "description": "Onboarding progress (0-100)",
                "required": False,
                "constraints": {"ge": 0, "le": 100},
                "metadata": {"use_in_tts": False, "hidden": False},
            },
            {
                "field_name": "user_understanding",
                "field_type": "Literal['clear', 'confused', 'needs_help']",
                "description": "User's understanding level",
                "required": False,
                "metadata": {"use_in_tts": False, "hidden": True},
            },
            {
                "field_name": "next_step",
                "field_type": "str",
                "description": "Next onboarding action",
                "required": False,
                "metadata": {"use_in_tts": False, "hidden": True},
            },
        ],
    },

    "llm_config": {
        "provider": "openai",
        "model": "gpt-5-nano",
        "temperature": 0.9,  # More creative for friendly tone
        "max_tokens": None,
        "supports_structured_output": True,
    },

    "tts_config": {
        "provider": "deepgram",
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

    "summary_config": {
        "enabled": True,
        "model": "gpt-5-nano",
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


# =======================
# HELPER FUNCTIONS
# =======================

def get_agent_config(agent_type: str) -> dict:
    """
    Get agent configuration by type.

    Args:
        agent_type: Type of agent (support, sales, onboarding, interview)

    Returns:
        Agent configuration dictionary

    Raises:
        ValueError: If agent type not found
    """
    configs = {
        "support": SUPPORT_AGENT_CONFIG,
        "sales": SALES_AGENT_CONFIG,
        "onboarding": ONBOARDING_AGENT_CONFIG,
    }

    if agent_type not in configs:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {', '.join(configs.keys())}"
        )

    return configs[agent_type]


async def setup_all_agents(config_store):
    """
    Setup all predefined agent configurations.

    Args:
        config_store: ConfigurationStore instance
    """
    agents = [
        SUPPORT_AGENT_CONFIG,
        SALES_AGENT_CONFIG,
        ONBOARDING_AGENT_CONFIG,
    ]

    for config in agents:
        try:
            await config_store.create_agent_config(config)
            print(f"✅ Created {config['agent_id']}")
        except ValueError as e:
            print(f"⚠️  {config['agent_id']} already exists: {e}")
        except Exception as e:
            print(f"❌ Failed to create {config['agent_id']}: {e}")
