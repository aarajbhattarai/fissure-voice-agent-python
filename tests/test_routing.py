"""
Tests for multi-agent routing logic.

Tests cover:
- Explicit agent_id routing
- Purpose-based routing
- User-attribute-based routing
- Default fallback routing
- Invalid routing scenarios
"""

import pytest
from unittest.mock import Mock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from multi_agent_worker import determine_agent_type


def test_explicit_agent_id_routing():
    """
    Test routing with explicit agent_id in metadata.

    Priority 1: Should use explicit agent_id when provided.
    """
    metadata = {
        "agent_id": "support-agent-v1",
        "purpose": "sales",  # Should be ignored
        "user_details": {"tier": "premium"},  # Should be ignored
    }

    result = determine_agent_type(metadata)

    assert result == "support-agent-v1"


def test_purpose_based_routing_support():
    """
    Test purpose-based routing for support.

    Priority 2: Should route based on purpose field.
    """
    metadata = {"purpose": "support", "user_details": {}}

    result = determine_agent_type(metadata)

    assert result == "support-agent-v1"


def test_purpose_based_routing_sales():
    """Test purpose-based routing for sales."""
    metadata = {"purpose": "sales", "user_details": {}}

    result = determine_agent_type(metadata)

    assert result == "sales-agent-v1"


def test_purpose_based_routing_onboarding():
    """Test purpose-based routing for onboarding."""
    metadata = {"purpose": "onboarding", "user_details": {}}

    result = determine_agent_type(metadata)

    assert result == "onboarding-agent-v1"


def test_purpose_based_routing_interview():
    """Test purpose-based routing for interview."""
    metadata = {"purpose": "interview", "user_details": {}}

    result = determine_agent_type(metadata)

    assert result == "pte-interview-agent-v1"


def test_purpose_case_insensitive():
    """Test that purpose routing is case-insensitive."""
    metadata_upper = {"purpose": "SUPPORT", "user_details": {}}
    metadata_mixed = {"purpose": "SuPpOrT", "user_details": {}}

    result_upper = determine_agent_type(metadata_upper)
    result_mixed = determine_agent_type(metadata_mixed)

    assert result_upper == "support-agent-v1"
    assert result_mixed == "support-agent-v1"


def test_user_attribute_new_user_routing():
    """
    Test user attribute-based routing for new users.

    Priority 3: Should route new users to onboarding.
    """
    metadata = {"user_details": {"is_new_user": True}}

    result = determine_agent_type(metadata)

    assert result == "onboarding-agent-v1"


def test_user_attribute_premium_tier_routing():
    """
    Test user attribute-based routing for premium users.

    Should route premium tier users to support.
    """
    metadata = {"user_details": {"tier": "premium"}}

    result = determine_agent_type(metadata)

    assert result == "support-agent-v1"


def test_default_fallback_routing():
    """
    Test default fallback routing.

    Priority 4: Should use default agent when no routing criteria match.
    """
    metadata = {}

    result = determine_agent_type(metadata)

    assert result == "support-agent-v1"


def test_empty_user_details_routing():
    """Test routing with empty user_details."""
    metadata = {"user_details": {}}

    result = determine_agent_type(metadata)

    assert result == "support-agent-v1"


def test_unknown_purpose_routing():
    """Test routing with unknown purpose falls back to default."""
    metadata = {"purpose": "unknown_purpose", "user_details": {}}

    result = determine_agent_type(metadata)

    assert result == "support-agent-v1"


def test_routing_priority_explicit_over_purpose():
    """
    Test that explicit agent_id has priority over purpose.

    Should use agent_id even when purpose is provided.
    """
    metadata = {
        "agent_id": "sales-agent-v1",
        "purpose": "support",  # Different purpose
        "user_details": {},
    }

    result = determine_agent_type(metadata)

    assert result == "sales-agent-v1"


def test_routing_priority_explicit_over_user_attributes():
    """
    Test that explicit agent_id has priority over user attributes.

    Should use agent_id even when user attributes would route differently.
    """
    metadata = {
        "agent_id": "sales-agent-v1",
        "user_details": {"is_new_user": True},  # Would normally route to onboarding
    }

    result = determine_agent_type(metadata)

    assert result == "sales-agent-v1"


def test_routing_priority_purpose_over_user_attributes():
    """
    Test that purpose has priority over user attributes.

    Should use purpose-based routing even when user attributes are present.
    """
    metadata = {
        "purpose": "sales",
        "user_details": {"is_new_user": True},  # Would normally route to onboarding
    }

    result = determine_agent_type(metadata)

    assert result == "sales-agent-v1"


def test_routing_with_multiple_user_attributes():
    """
    Test routing when multiple user attributes are present.

    Should use the first matching attribute (is_new_user checked first).
    """
    metadata = {
        "user_details": {"is_new_user": True, "tier": "premium"}  # Both present
    }

    result = determine_agent_type(metadata)

    # Should route to onboarding (is_new_user has priority in the code)
    assert result == "onboarding-agent-v1"


def test_routing_with_none_values():
    """Test routing handles None values gracefully."""
    metadata = {"purpose": None, "user_details": None}

    result = determine_agent_type(metadata)

    # Should use default fallback
    assert result == "support-agent-v1"


def test_routing_comprehensive_example():
    """
    Comprehensive test with realistic metadata.

    Verifies routing works correctly with full metadata structure.
    """
    metadata = {
        "user_id": "user-123",
        "tenant_id": "tenant-456",
        "purpose": "onboarding",
        "user_details": {
            "name": "John Doe",
            "email": "john@example.com",
            "is_new_user": False,
            "tier": "standard",
            "preferences": {"language": "en"},
        },
    }

    result = determine_agent_type(metadata)

    assert result == "onboarding-agent-v1"


@pytest.mark.parametrize(
    "metadata,expected_agent",
    [
        ({"agent_id": "custom-agent-v1"}, "custom-agent-v1"),
        ({"purpose": "support"}, "support-agent-v1"),
        ({"purpose": "sales"}, "sales-agent-v1"),
        ({"purpose": "onboarding"}, "onboarding-agent-v1"),
        ({"purpose": "interview"}, "pte-interview-agent-v1"),
        ({"user_details": {"is_new_user": True}}, "onboarding-agent-v1"),
        ({"user_details": {"tier": "premium"}}, "support-agent-v1"),
        ({}, "support-agent-v1"),
        ({"purpose": "invalid"}, "support-agent-v1"),
    ],
)
def test_routing_parametrized(metadata, expected_agent):
    """Parametrized test for various routing scenarios."""
    result = determine_agent_type(metadata)
    assert result == expected_agent


def test_routing_with_extra_metadata_fields():
    """
    Test that routing ignores extra metadata fields.

    Should work correctly even with additional fields present.
    """
    metadata = {
        "purpose": "sales",
        "user_details": {},
        "extra_field": "extra_value",
        "another_field": 12345,
        "nested": {"field": "value"},
    }

    result = determine_agent_type(metadata)

    assert result == "sales-agent-v1"
