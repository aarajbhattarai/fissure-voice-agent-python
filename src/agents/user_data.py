"""
User data management models for agent sessions.

These data classes provide structured, type-safe user information management.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


# =======================
# PYDANTIC MODELS (Runtime Validation)
# =======================


class UserDetailsModel(BaseModel):
    """
    Pydantic model for user details with validation.
    Use this when you need runtime validation and API serialization.
    """

    user_id: str = Field(..., description="Unique user identifier")
    name: Optional[str] = Field(None, description="User's full name")
    email: Optional[str] = Field(None, description="User's email address")
    phone: Optional[str] = Field(None, description="User's phone number")

    # Interview-specific fields
    age: Optional[int] = Field(None, ge=0, le=150, description="User's age")
    nationality: Optional[str] = Field(None, description="User's nationality")
    education_level: Optional[str] = Field(None, description="Highest education level")
    institution: Optional[str] = Field(None, description="Educational institution")
    field_of_study: Optional[str] = Field(None, description="Field of study")

    # Metadata
    language_preference: str = Field("en", description="Preferred language code")
    timezone: Optional[str] = Field(None, description="User's timezone")
    custom_fields: dict[str, Any] = Field(default_factory=dict, description="Custom user data")

    # Tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("email")
    def validate_email(cls, v):
        """Validate email format."""
        if v and "@" not in v:
            raise ValueError("Invalid email format")
        return v

    @validator("language_preference")
    def validate_language(cls, v):
        """Validate language code."""
        allowed_languages = ["en", "es", "fr", "de", "zh", "ja", "ko"]
        if v not in allowed_languages:
            raise ValueError(f"Unsupported language: {v}")
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.dict()

    def update(self, **kwargs):
        """Update user details."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()


# =======================
# DATACLASS (Performance-Oriented)
# =======================


@dataclass
class UserData:
    """
    Dataclass for user information.
    Use this for better performance when validation is not critical.
    """

    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    # Interview-specific
    age: Optional[int] = None
    nationality: Optional[str] = None
    education_level: Optional[str] = None
    institution: Optional[str] = None
    field_of_study: Optional[str] = None

    # Preferences
    language_preference: str = "en"
    timezone: Optional[str] = None

    # Metadata
    custom_fields: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def update(self, **kwargs):
        """Update user data."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()

    @classmethod
    def from_dict(cls, data: dict) -> "UserData":
        """Create from dictionary."""
        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


# =======================
# SESSION DATA
# =======================


@dataclass
class SessionData:
    """
    Session-level data container.
    Holds user information and session state.
    """

    session_id: str
    user_data: UserData
    tenant_id: str
    agent_id: str

    # Session metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Session state
    status: str = "active"  # active, completed, aborted, error
    error_message: Optional[str] = None

    # Custom session data
    session_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert user_data to dict if it's an object
        if isinstance(self.user_data, (UserData, UserDetailsModel)):
            data["user_data"] = self.user_data.to_dict()
        return data

    def end_session(self, status: str = "completed", error: Optional[str] = None):
        """Mark session as ended."""
        self.ended_at = datetime.utcnow()
        self.status = status
        self.error_message = error

        if self.started_at:
            self.duration_seconds = (self.ended_at - self.started_at).total_seconds()

    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        """Create from dictionary."""
        # Handle user_data separately
        user_data_dict = data.pop("user_data", {})
        user_data = UserData.from_dict(user_data_dict)

        return cls(user_data=user_data, **data)


# =======================
# CONVERSATION TURN
# =======================


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""

    turn_id: str
    timestamp: datetime
    speaker: str  # "user" or "agent"
    message: str

    # Optional structured data (from LLM)
    structured_data: Optional[dict] = None

    # Audio metadata
    audio_duration_ms: Optional[float] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# =======================
# CONVERSATION HISTORY
# =======================


@dataclass
class ConversationHistory:
    """Complete conversation history for a session."""

    session_id: str
    user_id: str
    turns: list[ConversationTurn] = field(default_factory=list)

    # Summary
    summary: Optional[str] = None
    summary_generated_at: Optional[datetime] = None

    # Metadata
    total_turns: int = 0
    total_duration_seconds: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_turn(
        self,
        speaker: str,
        message: str,
        structured_data: Optional[dict] = None,
        audio_duration_ms: Optional[float] = None,
    ):
        """Add a conversation turn."""
        from uuid import uuid4

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            speaker=speaker,
            message=message,
            structured_data=structured_data,
            audio_duration_ms=audio_duration_ms,
        )

        self.turns.append(turn)
        self.total_turns += 1

    def get_user_messages(self) -> list[str]:
        """Get all user messages."""
        return [turn.message for turn in self.turns if turn.speaker == "user"]

    def get_agent_messages(self) -> list[str]:
        """Get all agent messages."""
        return [turn.message for turn in self.turns if turn.speaker == "agent"]

    def get_full_transcript(self) -> str:
        """Get full conversation transcript."""
        lines = []
        for turn in self.turns:
            lines.append(f"{turn.speaker.upper()}: {turn.message}")
        return "\n".join(lines)

    def set_summary(self, summary: str):
        """Set conversation summary."""
        self.summary = summary
        self.summary_generated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": self.summary,
            "summary_generated_at": self.summary_generated_at.isoformat() if self.summary_generated_at else None,
            "total_turns": self.total_turns,
            "total_duration_seconds": self.total_duration_seconds,
            "created_at": self.created_at.isoformat(),
        }


# =======================
# HELPER FUNCTIONS
# =======================


def create_user_data_from_dict(data: dict, use_validation: bool = False) -> UserData | UserDetailsModel:
    """
    Create user data from dictionary.

    Args:
        data: User data dictionary
        use_validation: If True, use Pydantic model with validation

    Returns:
        UserData or UserDetailsModel instance
    """
    if use_validation:
        return UserDetailsModel(**data)
    else:
        return UserData.from_dict(data)


def merge_user_data(base: dict, overrides: dict) -> dict:
    """
    Merge user data dictionaries.

    Args:
        base: Base user data
        overrides: Override values

    Returns:
        Merged dictionary
    """
    merged = {**base}

    for key, value in overrides.items():
        if key == "custom_fields" and key in merged:
            # Deep merge custom fields
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged
