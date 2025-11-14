"""Example usage of ClaudeAgent with string outputs.

Note: This demonstrates the minimal ClaudeAgent implementation which currently
supports string outputs only. Pydantic output support will be added in future versions.
"""

import dspy
from codex_dspy import ClaudeAgent


def example_1_string_output():
    """Example 1: Simple string output - ask agent about files."""
    print("=" * 60)
    print("Example 1: Claude - String Output")
    print("=" * 60)

    # Create signature with string input and output
    sig = dspy.Signature("message:str -> answer:str")

    # Create Claude agent (starts new session)
    agent = ClaudeAgent(
        sig,
        working_directory=".",
        permission_mode="bypassPermissions",  # Automated execution without prompts
        max_turns=5,  # Limit agent autonomy
    )

    # Call agent
    result = agent(message="Count to 3 and list those numbers.")

    # Access results
    print(f"\nAnswer: {result.answer}")
    print(f"\nSession ID: {agent.session_id}")
    print(f"\nCost: ${result.cost_usd:.4f}" if result.cost_usd else "\nCost: N/A")
    print(f"Turns: {result.num_turns}" if result.num_turns else "Turns: N/A")
    print(f"\nTrace items ({len(result.trace)}):")
    for item in result.trace:
        print(f"  - {item['type']}")

    # Multi-turn: continue same session
    print("\n" + "-" * 60)
    print("Continuing conversation...")
    print("-" * 60)

    result2 = agent(message="Now count to 5.")
    print(f"\nAnswer: {result2.answer}")
    print(f"Session ID (same): {agent.session_id}")
    print(f"Total turns: {result2.num_turns}")


def example_2_with_description():
    """Example 2: Using output field description to guide Claude."""
    print("\n" + "=" * 60)
    print("Example 2: Claude - Output Field with Description")
    print("=" * 60)

    # Create signature with description on output field
    class AnalysisSignature(dspy.Signature):
        """Analyze repository architecture."""

        message: str = dspy.InputField()
        analysis: str = dspy.OutputField(
            desc="A detailed analysis in markdown format with sections for: "
            "1) Architecture overview, 2) Key components, 3) Dependencies"
        )

    # Create agent
    agent = ClaudeAgent(
        AnalysisSignature,
        working_directory=".",
        permission_mode="bypassPermissions",
    )

    # The description will be appended to the message automatically
    result = agent(message="Provide a brief fake analysis (just make something up for testing)")

    print(f"\nAnalysis:\n{result.analysis}")
    print(f"\nSession ID: {agent.session_id}")


def example_3_system_prompt():
    """Example 3: Using system prompt to customize behavior."""
    print("\n" + "=" * 60)
    print("Example 3: Claude - Custom System Prompt")
    print("=" * 60)

    # Create signature
    sig = dspy.Signature("message:str -> answer:str")

    # Create agent with custom system prompt
    agent = ClaudeAgent(
        sig,
        working_directory=".",
        system_prompt="You are a helpful code review assistant. "
        "Always be concise and focus on potential improvements.",
        permission_mode="bypassPermissions",
    )

    result = agent(message="Give a brief fake code review (just make something up for testing)")

    print(f"\nReview:\n{result.answer}")
    print(f"\nSession ID: {agent.session_id}")


def example_4_fresh_vs_continued():
    """Example 4: Demonstrate fresh context vs continued conversation."""
    print("\n" + "=" * 60)
    print("Example 4: Claude - Fresh Context vs Continuation")
    print("=" * 60)

    sig = dspy.Signature("message:str -> answer:str")

    # First agent instance
    print("\nAgent 1 (first instance):")
    agent1 = ClaudeAgent(sig, working_directory=".", permission_mode="bypassPermissions")
    result1 = agent1(message="Remember this: the project uses DSPy")
    print(f"Response: {result1.answer[:100]}...")
    print(f"Session ID: {agent1.session_id}")

    # Continue same agent
    print("\nAgent 1 (continued):")
    result2 = agent1(message="What did I just tell you to remember?")
    print(f"Response: {result2.answer[:100]}...")
    print(f"Session ID: {agent1.session_id} (same)")

    # New agent instance - fresh context
    print("\nAgent 2 (new instance, fresh context):")
    agent2 = ClaudeAgent(sig, working_directory=".", permission_mode="bypassPermissions")
    result3 = agent2(message="What did I tell you to remember?")
    print(f"Response: {result3.answer[:100]}...")
    print(f"Session ID: {agent2.session_id} (different)")


if __name__ == "__main__":
    # Run all examples
    # Note: These examples require Claude Code CLI to be installed
    # They work with Claude Code subscriptions (no API key required)

    print("ClaudeAgent Basic Usage Examples")
    print("=" * 60)
    print("\nPrerequisites:")
    print("- Claude Code CLI installed (v2.0.0+)")
    print("- Authenticated with Claude Code (via subscription or ANTHROPIC_API_KEY)")
    print("=" * 60)

    try:
        example_1_string_output()
        example_2_with_description()
        example_3_system_prompt()
        example_4_fresh_vs_continued()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Claude Code CLI is installed and accessible.")
        print("Visit: https://github.com/anthropics/claude-agent-sdk-python")
