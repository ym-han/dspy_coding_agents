"""Example usage of CodexAgent with the two-turn pattern.

Demonstrates:
- Single-field signatures (simple)
- Multi-field signatures with Pydantic models
- Multi-turn conversations
- Inspecting execution trace
"""

from typing import Literal

from pydantic import BaseModel, Field

import dspy
from codex import Model, ModelReasoningEffort, SandboxMode

from codex_dspy import CodexAgent


def example_1_simple_string():
    """Example 1: Simple string input/output."""
    print("=" * 60)
    print("Example 1: Simple String Signature")
    print("=" * 60)

    sig = dspy.Signature("message:str -> answer:str")

    agent = CodexAgent(
        sig,
        working_directory=".",
        model=Model.GPT_5_1_CODEX_MAX,
        sandbox_mode=SandboxMode.READ_ONLY,
        model_reasoning_effort=ModelReasoningEffort.LOW,
    )

    result = agent(message="What files are in this directory? List the top 5.")

    print(f"\nAnswer: {result.answer}")
    print(f"\nThread ID: {agent.thread_id}")
    print(f"Usage: {result.usage}")
    print(f"Trace items: {len(result.trace)}")


def example_2_multi_field_pydantic():
    """Example 2: Multiple fields with Pydantic models."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Field Signature with Pydantic")
    print("=" * 60)

    class BugReport(BaseModel):
        severity: Literal["low", "medium", "high"] = Field(description="Bug severity")
        location: str = Field(description="File and line number")
        description: str = Field(description="What the bug does")

    # Multiple inputs AND outputs - pass custom types explicitly
    sig = dspy.Signature(
        "code: str, context: str -> bugs: list[BugReport], summary: str",
        "Analyze code for potential bugs",
        custom_types={"BugReport": BugReport},
    )

    agent = CodexAgent(
        sig,
        working_directory=".",
        model=Model.GPT_5_1_CODEX_MAX,
        sandbox_mode=SandboxMode.READ_ONLY,
        model_reasoning_effort=ModelReasoningEffort.LOW,
    )

    result = agent(
        code="""
def divide(a, b):
    return a / b

def get_item(items, index):
    return items[index]
""",
        context="These are utility functions in a production calculator module"
    )

    print(f"\nSummary: {result.summary}")
    print(f"\nBugs found ({len(result.bugs)}):")
    for bug in result.bugs:
        print(f"  [{bug.severity}] {bug.location}")
        print(f"    {bug.description}")


def example_3_multi_turn():
    """Example 3: Multi-turn conversation with context."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Turn Conversation")
    print("=" * 60)

    sig = dspy.Signature("request: str -> response: str")

    agent = CodexAgent(
        sig,
        working_directory=".",
        model=Model.GPT_5_1_CODEX_MAX,
        sandbox_mode=SandboxMode.READ_ONLY,
        model_reasoning_effort=ModelReasoningEffort.LOW,
    )

    # Turn 1
    result1 = agent(request="What Python files are in this project?")
    print(f"\nTurn 1 Response: {result1.response[:200]}...")

    # Turn 2 - has context from Turn 1
    result2 = agent(request="Which one has the most lines of code?")
    print(f"\nTurn 2 Response: {result2.response[:200]}...")

    # Same thread throughout
    print(f"\nThread ID (same for both): {agent.thread_id}")


def example_4_complex_analysis():
    """Example 4: Complex multi-output analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Complex Analysis with Multiple Outputs")
    print("=" * 60)

    class FileInfo(BaseModel):
        path: str = Field(description="File path")
        purpose: str = Field(description="What this file does")
        key_functions: list[str] = Field(description="Important functions/classes")

    class RepoAnalysis(BaseModel):
        architecture: str = Field(description="High-level architecture description")
        main_files: list[FileInfo] = Field(description="Key files in the project")
        tech_stack: list[str] = Field(description="Technologies used")

    sig = dspy.Signature(
        "directory: str, focus: str -> analysis: RepoAnalysis, recommendations: str",
        "Analyze repository structure and provide recommendations",
        custom_types={"RepoAnalysis": RepoAnalysis, "FileInfo": FileInfo},
    )

    agent = CodexAgent(
        sig,
        working_directory=".",
        model=Model.GPT_5_1_CODEX_MAX,
        sandbox_mode=SandboxMode.READ_ONLY,
        model_reasoning_effort=ModelReasoningEffort.LOW,
    )

    result = agent(
        directory="src/",
        focus="Understand the DSPy integration architecture"
    )

    print(f"\nArchitecture: {result.analysis.architecture}")
    print(f"\nTech Stack: {result.analysis.tech_stack}")
    print(f"\nKey Files ({len(result.analysis.main_files)}):")
    for f in result.analysis.main_files[:3]:
        print(f"  {f.path}: {f.purpose}")
    print(f"\nRecommendations: {result.recommendations[:300]}...")


def example_5_trace_inspection():
    """Example 5: Inspecting the execution trace."""
    print("\n" + "=" * 60)
    print("Example 5: Execution Trace Inspection")
    print("=" * 60)

    from codex import CommandExecutionItem, AgentMessageItem

    sig = dspy.Signature("task: str -> result: str")

    agent = CodexAgent(
        sig,
        working_directory=".",
        model=Model.GPT_5_1_CODEX_MAX,
        sandbox_mode=SandboxMode.READ_ONLY,
        model_reasoning_effort=ModelReasoningEffort.LOW,
    )

    result = agent(task="Count the number of Python files in this project")

    print(f"\nResult: {result.result}")
    print(f"\nExecution Trace ({len(result.trace)} items):")

    for item in result.trace:
        if isinstance(item, CommandExecutionItem):
            print(f"  [CMD] {item.command}")
            print(f"        Exit: {item.exit_code}")
        elif isinstance(item, AgentMessageItem):
            preview = item.text[:100] + "..." if len(item.text) > 100 else item.text
            print(f"  [MSG] {preview}")
        else:
            print(f"  [{item.type}] {item.id}")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Simple string", example_1_simple_string),
        "2": ("Multi-field Pydantic", example_2_multi_field_pydantic),
        "3": ("Multi-turn conversation", example_3_multi_turn),
        "4": ("Complex analysis", example_4_complex_analysis),
        "5": ("Trace inspection", example_5_trace_inspection),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            examples[choice][1]()
        else:
            print(f"Unknown example: {choice}")
            print(f"Available: {list(examples.keys())}")
    else:
        print("CodexAgent Examples")
        print("=" * 60)
        print("\nRun a specific example:")
        for key, (name, _) in examples.items():
            print(f"  python examples/basic_usage.py {key}  # {name}")
        print("\nOr run all:")
        print("  python examples/basic_usage.py all")

        if len(sys.argv) > 1 and sys.argv[1] == "all":
            for _, func in examples.values():
                func()
            print("\n" + "=" * 60)
            print("All examples completed!")
            print("=" * 60)
