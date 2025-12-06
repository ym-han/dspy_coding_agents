"""
codex-dspy - DSPy module for OpenAI Codex SDK

Two-turn adapter pattern for keeping agents in-distribution.
"""


def main():
    print("codex-dspy: DSPy module for OpenAI Codex SDK")
    print()
    print("Features:")
    print("  - Multi-field signatures (any number of inputs/outputs)")
    print("  - Two-turn pattern (natural task + structured extraction)")
    print("  - Stateful threads (context preserved across calls)")
    print("  - BAML-style schemas for Pydantic models")
    print()
    print("Quick start:")
    print()
    print("  import dspy")
    print("  from codex_dspy import CodexAgent")
    print()
    print("  # Simple signature")
    print("  sig = dspy.Signature('message:str -> answer:str')")
    print("  agent = CodexAgent(sig, working_directory='.')")
    print("  result = agent(message='Hello!')")
    print()
    print("  # Multi-field signature with Pydantic")
    print("  from pydantic import BaseModel, Field")
    print()
    print("  class BugReport(BaseModel):")
    print("      severity: str = Field(description='Bug severity')")
    print("      description: str")
    print()
    print("  sig = dspy.Signature(")
    print("      'code: str, context: str -> bugs: list[BugReport], summary: str'")
    print("  )")
    print("  agent = CodexAgent(sig, working_directory='.')")
    print("  result = agent(code='...', context='...')")
    print("  print(result.bugs)    # list[BugReport]")
    print("  print(result.summary) # str")
    print()
    print("For more examples, see examples/basic_usage.py")
    print("Documentation: https://github.com/darinkishore/codex_dspy#readme")


if __name__ == "__main__":
    main()
