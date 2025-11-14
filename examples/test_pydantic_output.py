"""Test ClaudeAgent with Pydantic structured output.

This example demonstrates using ClaudeAgent with Pydantic models for structured output.
Requires ANTHROPIC_API_KEY environment variable to be set.
"""

from pydantic import BaseModel, Field
import dspy
from codex_dspy import ClaudeAgent


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="The overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    key_points: list[str] = Field(description="Key points supporting the sentiment")


class AnalyzeSignature(dspy.Signature):
    """Analyze the sentiment of a text message."""
    text: str = dspy.InputField(desc="Text to analyze")
    analysis: SentimentAnalysis = dspy.OutputField(desc="Sentiment analysis result")


def test_basic_pydantic_output():
    """Test basic Pydantic output with ClaudeAgent."""
    print("Testing ClaudeAgent with Pydantic output...")

    # Create agent (working_directory not needed for Pydantic mode)
    agent = ClaudeAgent(AnalyzeSignature)

    # Test with a simple message
    test_text = "I absolutely love this product! It works perfectly and exceeded all my expectations."

    print(f"\nInput text: {test_text}")
    print("\nCalling ClaudeAgent...")

    result = agent(text=test_text)

    print(f"\nResult type: {type(result.analysis)}")
    print(f"Sentiment: {result.analysis.sentiment}")
    print(f"Confidence: {result.analysis.confidence}")
    print(f"Key points: {result.analysis.key_points}")

    # Verify it's a proper Pydantic model
    assert isinstance(result.analysis, SentimentAnalysis)
    assert hasattr(result.analysis, 'sentiment')
    assert hasattr(result.analysis, 'confidence')
    assert hasattr(result.analysis, 'key_points')

    # Check trace and metadata
    print(f"\nTrace: {result.trace}")
    if hasattr(result, 'usage'):
        print(f"Usage: {result.usage}")
    if hasattr(result, 'cost_usd'):
        print(f"Cost: ${result.cost_usd:.6f}")

    print("\n✓ Test passed!")


def test_complex_pydantic_model():
    """Test with a more complex nested Pydantic model."""

    class Entity(BaseModel):
        name: str = Field(description="Entity name")
        type: str = Field(description="Entity type (person, place, organization, etc.)")

    class TextAnalysis(BaseModel):
        summary: str = Field(description="Brief summary of the text")
        topics: list[str] = Field(description="Main topics discussed")
        entities: list[Entity] = Field(description="Named entities found in the text")
        word_count: int = Field(description="Approximate word count")

    class AnalyzeTextSignature(dspy.Signature):
        text: str = dspy.InputField()
        analysis: TextAnalysis = dspy.OutputField()

    print("\n\nTesting ClaudeAgent with complex nested Pydantic model...")

    agent = ClaudeAgent(AnalyzeTextSignature)

    test_text = """
    Apple Inc. announced today that CEO Tim Cook will visit Paris next month to meet with
    French President Emmanuel Macron. The meeting will focus on technology regulation and
    Apple's investments in Europe.
    """

    print(f"\nInput text: {test_text.strip()}")
    print("\nCalling ClaudeAgent...")

    result = agent(text=test_text)

    print(f"\nSummary: {result.analysis.summary}")
    print(f"Topics: {result.analysis.topics}")
    print(f"Entities:")
    for entity in result.analysis.entities:
        print(f"  - {entity.name} ({entity.type})")
    print(f"Word count: {result.analysis.word_count}")

    print("\n✓ Test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("ClaudeAgent Pydantic Output Tests")
    print("=" * 70)

    try:
        test_basic_pydantic_output()
        test_complex_pydantic_model()

        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
