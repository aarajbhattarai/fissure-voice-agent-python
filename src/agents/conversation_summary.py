"""
Conversation summary generation using LLM.

Generates summaries of conversation history after sessions end.
"""

import logging
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger("conversation-summary")


class ConversationSummarizer:
    """
    Generates conversation summaries using LLM.
    Configurable prompts and output format.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
    ):
        """
        Initialize summarizer.

        Args:
            llm_provider: LLM provider (currently only OpenAI supported)
            model: Model name
            api_key: API key (optional, uses env var if not provided)
        """
        self.llm_provider = llm_provider
        self.model = model

        if llm_provider == "openai":
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    async def generate_summary(
        self,
        transcript: str,
        user_data: Optional[dict] = None,
        session_metadata: Optional[dict] = None,
        custom_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate conversation summary.

        Args:
            transcript: Full conversation transcript
            user_data: User information for context
            session_metadata: Session metadata
            custom_prompt: Optional custom prompt template

        Returns:
            Dictionary with summary and metadata
        """
        # Build context
        context_parts = []

        if user_data:
            context_parts.append("USER INFORMATION:")
            context_parts.append(f"Name: {user_data.get('name', 'Unknown')}")
            context_parts.append(
                f"Email: {user_data.get('email', 'Not provided')}"
            )
            if user_data.get("institution"):
                context_parts.append(f"Institution: {user_data['institution']}")
            if user_data.get("field_of_study"):
                context_parts.append(f"Field of Study: {user_data['field_of_study']}")
            context_parts.append("")

        if session_metadata:
            context_parts.append("SESSION METADATA:")
            for key, value in session_metadata.items():
                context_parts.append(f"{key}: {value}")
            context_parts.append("")

        context = "\n".join(context_parts)

        # Use custom prompt or default
        if custom_prompt:
            prompt = custom_prompt.format(context=context, transcript=transcript)
        else:
            prompt = self._get_default_prompt(context, transcript)

        logger.info(f"Generating summary with model: {self.model}")

        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at summarizing conversations. Provide clear, concise, and insightful summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=1000,
            )

            summary_text = response.choices[0].message.content

            # Parse structured summary
            summary_data = self._parse_summary(summary_text)

            logger.info("Summary generated successfully")

            return {
                "summary": summary_text,
                "structured_summary": summary_data,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else None,
            }

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}", exc_info=True)
            raise

    def _get_default_prompt(self, context: str, transcript: str) -> str:
        """Get default summary prompt."""
        return f"""
{context}

CONVERSATION TRANSCRIPT:
{transcript}

Please provide a comprehensive summary of this conversation including:

1. **Overview**: Brief description of the conversation (2-3 sentences)

2. **Key Topics Discussed**:
   - Main topics covered
   - Important points raised

3. **User Insights**:
   - User's goals and objectives
   - User's concerns or questions
   - User's background (if discussed)

4. **Agent Performance**:
   - How well did the agent address user needs?
   - Were there any issues or areas for improvement?

5. **Outcomes**:
   - Decisions made or actions agreed upon
   - Next steps (if any)
   - Overall satisfaction indicators

6. **Key Quotes**: Notable statements from the conversation

7. **Recommendations**: Suggestions for follow-up or improvements

Please format your response clearly with the above sections.
"""

    def _parse_summary(self, summary_text: str) -> dict:
        """
        Parse summary text into structured data.

        Args:
            summary_text: Summary text from LLM

        Returns:
            Structured summary dictionary
        """
        # Simple parsing logic - can be enhanced with more sophisticated NLP
        sections = {}
        current_section = None
        current_content = []

        for line in summary_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if line is a section header
            if line.endswith(":") or line.startswith("**") or line.startswith("#"):
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                current_section = line.replace("**", "").replace("#", "").replace(":", "").strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    async def generate_quick_summary(
        self, transcript: str, max_length: int = 200
    ) -> str:
        """
        Generate a quick, short summary.

        Args:
            transcript: Conversation transcript
            max_length: Maximum summary length in characters

        Returns:
            Brief summary string
        """
        prompt = f"""
Provide a brief 2-3 sentence summary of this conversation:

{transcript}

Keep it under {max_length} characters and focus on the main point.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You provide concise summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=100,
            )

            summary = response.choices[0].message.content
            return summary[:max_length]

        except Exception as e:
            logger.error(f"Failed to generate quick summary: {e}")
            return "Summary generation failed."

    async def generate_structured_summary(
        self, transcript: str, schema: dict
    ) -> dict:
        """
        Generate summary with custom structured output schema.

        Args:
            transcript: Conversation transcript
            schema: JSON schema for output structure

        Returns:
            Structured summary matching schema
        """
        # This would use OpenAI's structured output feature
        # Similar to how we do it in the agent
        prompt = f"""
Analyze this conversation and extract information according to the provided schema:

CONVERSATION:
{transcript}

Provide structured output.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract structured information from conversations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            import json

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Failed to generate structured summary: {e}")
            raise


# =======================
# SUMMARY STORAGE
# =======================


class SummaryStorage:
    """Store and retrieve conversation summaries."""

    def __init__(self, db):
        """
        Initialize storage.

        Args:
            db: MongoDB database instance
        """
        self.summaries = db.conversation_summaries

    async def save_summary(
        self,
        session_id: str,
        user_id: str,
        summary_data: dict,
        conversation_history: Optional[dict] = None,
    ) -> str:
        """
        Save conversation summary.

        Args:
            session_id: Session identifier
            user_id: User identifier
            summary_data: Summary data from summarizer
            conversation_history: Optional full conversation history

        Returns:
            Inserted document ID
        """
        from datetime import datetime

        document = {
            "session_id": session_id,
            "user_id": user_id,
            "summary": summary_data["summary"],
            "structured_summary": summary_data.get("structured_summary", {}),
            "model": summary_data.get("model"),
            "tokens_used": summary_data.get("tokens_used"),
            "created_at": datetime.utcnow(),
        }

        if conversation_history:
            document["conversation_history"] = conversation_history

        result = await self.summaries.insert_one(document)
        return str(result.inserted_id)

    async def get_summary(self, session_id: str) -> Optional[dict]:
        """
        Get summary by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Summary document or None
        """
        summary = await self.summaries.find_one({"session_id": session_id})

        if summary:
            summary.pop("_id", None)

        return summary

    async def get_user_summaries(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        """
        Get all summaries for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of summaries to return

        Returns:
            List of summary documents
        """
        summaries = (
            await self.summaries.find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
            .to_list(limit)
        )

        for summary in summaries:
            summary.pop("_id", None)

        return summaries
