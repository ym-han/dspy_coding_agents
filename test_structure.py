"""Test that the structure works correctly without API calls."""

from pydantic import BaseModel, Field
import dspy
from codex_dspy import ClaudeAgent


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0-1", ge=0, le=1)
    key_points: list[str]


class AnalyzeSignature(dspy.Signature):
    """Analyze sentiment."""
    text: str = dspy.InputField()
    analysis: SentimentAnalysis = dspy.OutputField()


def test_structure():
    """Test that ClaudeAgent initializes correctly."""
    print("Testing ClaudeAgent structure...")

    # Test Pydantic mode initialization
    agent = ClaudeAgent(AnalyzeSignature)

    print(f"✓ Agent created")
    print(f"  - Uses Pydantic mode: {agent._use_pydantic_mode}")
    print(f"  - Pydantic model: {agent._pydantic_model.__name__}")
    print(f"  - Output field: {agent.output_field}")

    # Test schema generation
    schema = agent._pydantic_model.model_json_schema()
    print(f"✓ Schema generated: {list(schema.get('properties', {}).keys())}")

    # Test prompt augmentation
    test_message = "I love this product!"
    augmented = agent._augment_prompt_with_schema(test_message, schema)
    print(f"✓ Prompt augmented (length: {len(augmented)} chars)")
    print(f"  Contains schema: {'schema' in augmented.lower()}")
    print(f"  Contains <response> tags: {'<response>' in augmented}")

    # Test JSON extraction with mock response
    mock_response = """
Here's my analysis:

<response>
{
  "sentiment": "positive",
  "confidence": 0.95,
  "key_points": ["love", "exceeded expectations"]
}
</response>

Hope that helps!
"""

    json_str = agent._extract_json_from_response(mock_response)
    print(f"✓ JSON extracted from mock response")

    # Test validation
    parsed = agent._pydantic_model.model_validate_json(json_str)
    print(f"✓ Pydantic validation successful")
    print(f"  - sentiment: {parsed.sentiment}")
    print(f"  - confidence: {parsed.confidence}")
    print(f"  - key_points: {parsed.key_points}")

    print("\n✓ All structure tests passed!")


if __name__ == "__main__":
    test_structure()
