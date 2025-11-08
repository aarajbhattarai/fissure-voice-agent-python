"""
Agent factory for dynamic agent instantiation.
"""

import copy
import logging
from typing import Any, Optional, Type

from livekit import rtc
from livekit.plugins import deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from opentelemetry import trace
from pydantic import Field
from typing_extensions import Literal, TypedDict

from interview_agent.utilities.setup_langfuse import setup_langfuse

from .configuration_store import ConfigurationStore

logger = logging.getLogger("agent-factory")


class AgentFactory:
    """
    Dynamically creates agent instances based on configuration.
    Implements the Factory Pattern for agent creation.
    """

    def __init__(self, config_store: ConfigurationStore):
        """
        Initialize agent factory.

        Args:
            config_store: Configuration store instance
        """
        self.config_store = config_store
        self.schema_registry: dict[str, Type[TypedDict]] = {}
        self.agent_registry: dict[str, Type] = {}

    async def create_agent(
        self,
        agent_id: str,
        session_id: str,
        user_id: str,
        room: rtc.Room,
        user_details: Optional[dict[str, Any]] = None,
        overrides: Optional[dict[str, Any]] = None,
    ):
        """
        Creates and configures an agent instance dynamically.

        Args:
            agent_id: Configuration identifier
            session_id: Unique session identifier
            user_id: User identifier
            room: LiveKit room instance
            user_details: User-specific metadata
            overrides: Session-specific config overrides

        Returns:
            Configured agent instance
        """
        # 1. Load configuration from store
        config = await self.config_store.get_agent_config(agent_id)

        # 2. Apply session overrides
        if overrides:
            config = self._merge_config(config, overrides)

        # 3. Generate dynamic structured output schema
        schema_class = self._create_schema_from_config(config["schema_config"])

        # 4. Load prompt template
        prompt = await self._load_prompt(config["prompt_config"])

        # 5. Configure LLM
        llm = self._create_llm(config["llm_config"])

        # 6. Configure TTS
        tts = self._create_tts(config["tts_config"])

        # 7. Configure STT
        stt = self._create_stt(config.get("stt_config", {}))

        # 8. Configure turn detection
        turn_detection = self._create_turn_detector(config.get("pipeline_config", {}))

        # 9. Configure tracing (optional)
        tracer = None
        if config.get("tracing_config", {}).get("enabled", False):
            tracer = self._create_tracer(
                config["tracing_config"], session_id=session_id
            )

        # 10. Import and get agent class
        from agents.base_agent import BaseAgent

        # 11. Instantiate agent with all configurations
        agent = BaseAgent(
            room=room,
            user_id=user_id,
            session_id=session_id,
            llm=llm,
            tts=tts,
            stt=stt,
            turn_detection=turn_detection,
            prompt=prompt,
            schema_class=schema_class,
            tracer=tracer,
            config=config,
            user_details=user_details or {},
        )

        logger.info(
            f"Created agent instance: {agent_id} for session {session_id}, user {user_id}"
        )

        return agent

    def _create_schema_from_config(self, schema_config: dict) -> Type[TypedDict]:
        """
        Dynamically creates a TypedDict class from configuration.

        Args:
            schema_config: Schema configuration dictionary

        Returns:
            TypedDict class
        """
        schema_id = schema_config["schema_id"]

        # Check if already in registry
        if schema_id in self.schema_registry:
            return self.schema_registry[schema_id]

        annotations = {}

        for field_def in schema_config["fields"]:
            field_name = field_def["field_name"]
            field_type = self._parse_type(field_def["field_type"])
            description = field_def["description"]

            # Build annotated field with Pydantic Field
            constraints = field_def.get("constraints", {})

            from typing import Annotated

            annotations[field_name] = Annotated[
                field_type, Field(description=description, **constraints)
            ]

        # Generate dynamic TypedDict
        schema_name = f"DynamicSchema_{schema_id}"
        DynamicSchema = type(
            schema_name,
            (TypedDict,),
            {"__annotations__": annotations, "__total__": False},  # Allow partial
        )

        # Register in schema registry
        self.schema_registry[schema_id] = DynamicSchema

        logger.info(
            f"Created dynamic schema '{schema_name}' with {len(annotations)} fields"
        )

        return DynamicSchema

    def _parse_type(self, type_str: str) -> Type:
        """
        Parse string type representation to Python type.

        Args:
            type_str: Type string (e.g., "str", "int", "list[str]", "Literal['a','b']")

        Returns:
            Python type
        """
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list[str]": list[str],
            "list[int]": list[int],
            "list[float]": list[float],
            "dict": dict,
        }

        # Handle Literal types
        if type_str.startswith("Literal["):
            import ast

            # Extract literal values: Literal['a','b','c'] -> ('a', 'b', 'c')
            literal_content = type_str[8:-1]  # Remove "Literal[" and "]"
            # Parse as tuple
            try:
                values = ast.literal_eval(f"({literal_content})")
                # Create Literal type
                return Literal[values]
            except Exception as e:
                logger.error(f"Failed to parse Literal type '{type_str}': {e}")
                return str

        return type_map.get(type_str, str)

    async def _load_prompt(self, prompt_config: dict) -> str:
        """
        Load prompt from configuration.

        Args:
            prompt_config: Prompt configuration dictionary

        Returns:
            Prompt string
        """
        # For now, return instructions directly
        # In future, could support template variables, etc.
        return prompt_config.get("instructions", "")

    def _create_llm(self, llm_config: dict):
        """
        Create LLM instance from configuration.

        Args:
            llm_config: LLM configuration dictionary

        Returns:
            LLM instance
        """
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-5-nano")
        temperature = llm_config.get("temperature", 0.7)

        if provider == "openai":
            return openai.LLM(model=model, temperature=temperature)
        else:
            logger.warning(f"Unknown LLM provider '{provider}', defaulting to OpenAI")
            return openai.LLM(model=model, temperature=temperature)

    def _create_tts(self, tts_config: dict):
        """
        Create TTS instance from configuration.

        Args:
            tts_config: TTS configuration dictionary

        Returns:
            TTS instance
        """
        provider = tts_config.get("provider", "deepgram")

        if provider == "deepgram":
            return deepgram.TTS()
        elif provider == "openai":
            return openai.TTS()
        else:
            logger.warning(f"Unknown TTS provider '{provider}', defaulting to Deepgram")
            return deepgram.TTS()

    def _create_stt(self, stt_config: dict):
        """
        Create STT instance from configuration.

        Args:
            stt_config: STT configuration dictionary

        Returns:
            STT instance
        """
        provider = stt_config.get("provider", "deepgram")

        if provider == "deepgram":
            return deepgram.STT()
        else:
            logger.warning(f"Unknown STT provider '{provider}', defaulting to Deepgram")
            return deepgram.STT()

    def _create_turn_detector(self, pipeline_config: dict):
        """
        Create turn detector from configuration.

        Args:
            pipeline_config: Pipeline configuration dictionary

        Returns:
            Turn detector instance
        """
        # For now, always use MultilingualModel
        # Could be made configurable in the future
        return MultilingualModel()

    def _create_tracer(
        self, tracing_config: dict, session_id: str
    ) -> Optional[trace.Tracer]:
        """
        Create tracer instance from configuration.

        Args:
            tracing_config: Tracing configuration dictionary
            session_id: Session identifier for tracing metadata

        Returns:
            OpenTelemetry Tracer or None
        """
        if not tracing_config.get("enabled", False):
            return None

        provider = tracing_config.get("provider", "langfuse")

        if provider == "langfuse":
            # Setup Langfuse with OpenTelemetry
            trace_provider = setup_langfuse(
                metadata={"langfuse.session.id": session_id}
            )
            return trace.get_tracer(__name__)
        else:
            logger.warning(f"Unknown tracing provider '{provider}'")
            return None

    def _merge_config(self, base: dict, overrides: dict) -> dict:
        """
        Deep merge configuration with overrides.

        Args:
            base: Base configuration
            overrides: Override values

        Returns:
            Merged configuration
        """
        result = copy.deepcopy(base)

        def deep_merge(d1, d2):
            for key, value in d2.items():
                if (
                    key in d1
                    and isinstance(d1[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value

        deep_merge(result, overrides)
        return result
