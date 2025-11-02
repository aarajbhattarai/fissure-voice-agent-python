from typing import Any


def get_default_user_details() -> dict[str, Any]:
    """Get default user details for the interview."""
    return {
        "name": "Aaraj",
        "email": "john.smith@university.edu",
        "university": "MIT",
        "program": "Computer Science",
        "visa_type": "F-1",
        "interview_attempt": 1,
        "preparation_level": "intermediate",
    }
