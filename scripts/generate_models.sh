#!/bin/bash
# Script to generate Pydantic models from JSON Schema files using datamodel-code-generator

set -e

echo "Generating Pydantic models from JSON Schemas..."

# Create output directory
mkdir -p src/interview_agent/models

# Generate InterviewTurn model
echo "Generating InterviewTurn model..."
uv run datamodel-codegen \
  --input schemas/interview_turn.json \
  --output src/interview_agent/models/interview_turn.py \
  --output-model-type pydantic_v2.BaseModel \
  --field-constraints \
  --use-default \
  --use-default-kwarg \
  --use-schema-description \
  --use-field-description \
  --target-python-version 3.11 \
  --snake-case-field \
  --strict-types str int

# Generate SupportTurn model
echo "Generating SupportTurn model..."
uv run datamodel-codegen \
  --input schemas/support_turn.json \
  --output src/interview_agent/models/support_turn.py \
  --output-model-type pydantic_v2.BaseModel \
  --field-constraints \
  --use-default \
  --use-default-kwarg \
  --use-schema-description \
  --use-field-description \
  --target-python-version 3.11 \
  --snake-case-field \
  --strict-types str int

# Generate SalesTurn model
echo "Generating SalesTurn model..."
uv run datamodel-codegen \
  --input schemas/sales_turn.json \
  --output src/interview_agent/models/sales_turn.py \
  --output-model-type pydantic_v2.BaseModel \
  --field-constraints \
  --use-default \
  --use-default-kwarg \
  --use-schema-description \
  --use-field-description \
  --target-python-version 3.11 \
  --snake-case-field \
  --strict-types str int

# Create __init__.py for models package
echo "Creating models package __init__.py..."
cat > src/interview_agent/models/__init__.py << 'EOF'
"""
Auto-generated Pydantic models from JSON Schema.

Generated using datamodel-code-generator.
To regenerate, run: bash scripts/generate_models.sh
"""

from .interview_turn import InterviewTurn
from .support_turn import SupportTurn
from .sales_turn import SalesTurn

__all__ = ["InterviewTurn", "SupportTurn", "SalesTurn"]
EOF

echo "✅ Model generation complete!"
echo ""
echo "Generated models:"
echo "  - src/interview_agent/models/interview_turn.py"
echo "  - src/interview_agent/models/support_turn.py"
echo "  - src/interview_agent/models/sales_turn.py"
echo ""
echo "To use these models in your code:"
echo "  from interview_agent.models import InterviewTurn, SupportTurn, SalesTurn"
