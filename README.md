# CodexAgent - DSPy Module for OpenAI Codex SDK

A DSPy module that wraps the OpenAI Codex SDK with a signature-driven interface. Supports **multiple input/output fields** and uses a **two-turn pattern** that keeps agents "in distribution" during task execution.

## Features

- **Multi-field signatures** - Any number of input and output fields
- **Two-turn pattern** - Natural task execution + structured extraction
- **Stateful threads** - Each agent instance = one conversation thread
- **Smart schema handling** - BAML-style schemas for Pydantic, markers for strings
- **Rich outputs** - Get typed results + execution trace + token usage
- **DSPy-native** - Based on TwoStepAdapter and BAMLAdapter patterns

## Installation

```bash
# Install dependencies
uv sync

# For development (includes pytest, pre-commit)
uv sync --extra dev

# Ensure codex CLI is available
which codex
```

## Quick Start

### Simple Single-Field Signature

```python
import dspy
from codex_dspy import CodexAgent

sig = dspy.Signature('message:str -> answer:str')
agent = CodexAgent(sig, working_directory=".")

result = agent(message="What files are in this directory?")
print(result.answer)  # String response
print(result.trace)   # Execution trace
print(result.usage)   # Token counts
```

### Multi-Field Signature with Pydantic

```python
from typing import Literal
from pydantic import BaseModel, Field

class BugReport(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(description="Bug severity")
    location: str = Field(description="File and line number")
    description: str = Field(description="What the bug does")
    suggested_fix: str = Field(description="How to fix it")

# Multiple inputs AND outputs
sig = dspy.Signature(
    "code: str, context: str -> bugs: list[BugReport], summary: str",
    "Analyze code for bugs and provide a summary"
)

agent = CodexAgent(sig, working_directory=".")
result = agent(
    code="def divide(a, b): return a / b",
    context="Production calculator module"
)

print(result.summary)           # str
print(result.bugs)              # list[BugReport]
print(result.bugs[0].severity)  # Typed access!
```

## How It Works: Two-Turn Pattern

Unlike forcing JSON output during task execution (which pushes models out of distribution), CodexAgent uses a **two-turn pattern**:

### Turn 1: Natural Task Execution

The agent receives a natural prompt and does its work freely:

```
As input, you are provided with:
1. `code` (str): Source code to analyze
2. `context` (str): Additional context

Your task is to produce:
1. `bugs` (list[BugReport]): Bugs found in the code
2. `summary` (str): Overall analysis summary

Instructions: Analyze code for bugs and provide a summary

---

code: def divide(a, b): return a / b

context: Production calculator module
```

The agent reads files, runs commands, reasons naturally - no JSON pressure.

### Turn 2: Structured Extraction

After the task completes, the agent formats its findings:

```
Now provide your findings in the following format:

[[ ## bugs ## ]]
[
  {
    # Bug severity
    severity: "low" or "medium" or "high",
    # File and line number
    location: string,
    # What the bug does
    description: string,
    # How to fix it
    suggested_fix: string,
  }
]

[[ ## summary ## ]]
<string>

[[ ## completed ## ]]
```

This separation keeps the agent in-distribution during the actual work.

## API Reference

### CodexAgent

```python
class CodexAgent(dspy.Module):
    def __init__(
        self,
        signature: str | type[Signature],  # Any number of input/output fields
        working_directory: str,
        model: Optional[str] = None,
        sandbox_mode: Optional[SandboxMode] = None,
        skip_git_repo_check: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        codex_path_override: Optional[str] = None,
    )
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `signature` | `str \| type[Signature]` | DSPy signature with any number of input/output fields |
| `working_directory` | `str` | Directory where agent executes commands |
| `model` | `Optional[str]` | Model name (default: "gpt-5.1-codex-max") |
| `sandbox_mode` | `Optional[SandboxMode]` | `READ_ONLY`, `WORKSPACE_WRITE`, or `DANGER_FULL_ACCESS` |
| `skip_git_repo_check` | `bool` | Allow non-git directories |
| `api_key` | `Optional[str]` | OpenAI API key (falls back to `CODEX_API_KEY` env) |
| `base_url` | `Optional[str]` | API base URL (falls back to `OPENAI_BASE_URL` env) |
| `codex_path_override` | `Optional[str]` | Override path to codex binary |

#### Methods

##### `forward(**kwargs) -> Prediction`

Execute the agent with all input fields.

**Returns** a `Prediction` with:
- All output fields (typed according to signature)
- `trace` - `list[ThreadItem]` - Execution items (commands, files, etc.)
- `usage` - `Usage` - Token counts

### CodexAdapter

The adapter that formats prompts. You usually don't need to use this directly, but it's available:

```python
from codex_dspy import CodexAdapter

adapter = CodexAdapter()

# Format Turn 1 (task)
turn1 = adapter.format_turn1(signature, inputs)

# Format Turn 2 (extraction with BAML-style markers)
turn2 = adapter.format_turn2(signature)

# Or Turn 2 with JSON schema
turn2_json = adapter.format_turn2_json(signature)

# Parse [[ ## field ## ]] markers from response
parsed = adapter.parse(signature, completion)
```

## Usage Patterns

### Pattern 1: Multi-Turn Conversation

Each agent instance maintains a stateful thread:

```python
agent = CodexAgent(sig, working_directory=".")

# Turn 1 - agent does work
result1 = agent(code="...", context="...")
print(result1.summary)

# Turn 2 - has full context from Turn 1
result2 = agent(code="...", context="Now fix the bugs you found")
print(result2.summary)

# Same thread throughout
print(agent.thread_id)
```

### Pattern 2: Complex Multi-Field Analysis

```python
class SecurityAudit(BaseModel):
    vulnerabilities: list[Vulnerability]
    risk_score: float = Field(description="0-10 risk score")
    recommendations: list[str]

class TestCoverage(BaseModel):
    covered_functions: list[str]
    uncovered_functions: list[str]
    coverage_percent: float

sig = dspy.Signature(
    "codebase: str, focus_areas: list[str] -> "
    "security: SecurityAudit, tests: TestCoverage, report: str",
    "Perform security audit and test coverage analysis"
)

agent = CodexAgent(sig, working_directory="/path/to/project")
result = agent(
    codebase="src/",
    focus_areas=["authentication", "data validation"]
)

print(f"Risk Score: {result.security.risk_score}")
print(f"Coverage: {result.tests.coverage_percent}%")
print(f"Report:\n{result.report}")
```

### Pattern 3: Inspecting Execution Trace

```python
from codex import CommandExecutionItem, FileChangeItem

result = agent(code="...", context="Fix the bug")

# What commands ran?
commands = [item for item in result.trace if isinstance(item, CommandExecutionItem)]
for cmd in commands:
    print(f"$ {cmd.command}")
    print(f"  Exit: {cmd.exit_code}")

# What files changed?
files = [item for item in result.trace if isinstance(item, FileChangeItem)]
for f in files:
    for change in f.changes:
        print(f"  {change.kind}: {change.path}")
```

### Pattern 4: Safe Execution with Sandbox

```python
from codex import SandboxMode

# Read-only (safest - for analysis tasks)
agent = CodexAgent(sig, working_directory=".", sandbox_mode=SandboxMode.READ_ONLY)

# Can modify workspace (for fix/refactor tasks)
agent = CodexAgent(sig, working_directory=".", sandbox_mode=SandboxMode.WORKSPACE_WRITE)

# Full access (use with caution!)
agent = CodexAgent(sig, working_directory=".", sandbox_mode=SandboxMode.DANGER_FULL_ACCESS)
```

## Advanced Examples

### Code Review with Multiple Outputs

```python
class Issue(BaseModel):
    severity: Literal["critical", "high", "medium", "low"]
    file: str
    line: int
    description: str
    suggestion: str

class ReviewResult(BaseModel):
    approved: bool
    issues: list[Issue]

sig = dspy.Signature(
    "diff: str, guidelines: str -> review: ReviewResult, summary: str",
    "Review code changes against guidelines"
)

agent = CodexAgent(sig, working_directory=".", sandbox_mode=SandboxMode.READ_ONLY)

result = agent(
    diff=open("changes.diff").read(),
    guidelines="No hardcoded secrets. All functions must have docstrings."
)

if not result.review.approved:
    print("Review failed!")
    for issue in result.review.issues:
        print(f"  [{issue.severity}] {issue.file}:{issue.line}")
        print(f"    {issue.description}")
        print(f"    Suggestion: {issue.suggestion}")
```

### Repository Analysis Pipeline

```python
# Step 1: Gather stats
class RepoStats(BaseModel):
    total_files: int
    languages: dict[str, int]  # language -> file count
    largest_files: list[str]

stats_agent = CodexAgent(
    dspy.Signature("path: str -> stats: RepoStats"),
    working_directory="."
)
stats = stats_agent(path=".").stats

# Step 2: Architecture analysis (uses stats as context)
class Component(BaseModel):
    name: str
    responsibility: str
    dependencies: list[str]

arch_agent = CodexAgent(
    dspy.Signature("repo_info: str -> components: list[Component], diagram: str"),
    working_directory="."
)
arch = arch_agent(
    repo_info=f"Languages: {stats.languages}, Files: {stats.total_files}"
)

print("Components:")
for comp in arch.components:
    print(f"  {comp.name}: {comp.responsibility}")
print(f"\nDiagram:\n{arch.diagram}")
```

## Trace Item Types

| Type | Description |
|------|-------------|
| `AgentMessageItem` | Agent's text response |
| `ReasoningItem` | Agent's internal reasoning |
| `CommandExecutionItem` | Shell command execution |
| `FileChangeItem` | File modifications |
| `McpToolCallItem` | MCP tool invocation |
| `WebSearchItem` | Web search performed |
| `TodoListItem` | Task list created |
| `ErrorItem` | Error that occurred |

## Design Philosophy

### Why Two-Turn Pattern?

Traditional structured output forces the model to think about JSON formatting while doing complex agentic work. This is out-of-distribution - models are trained to reason naturally, then format at the end.

Our two-turn pattern:
1. **Turn 1**: Agent works naturally (reads files, runs commands, reasons)
2. **Turn 2**: Agent formats findings into structure (quick, focused)

This keeps the agent in-distribution during the actual work.

### Why Multi-Field Support?

Real agentic tasks often need:
- Multiple inputs (code + context + config)
- Multiple outputs (analysis + recommendations + confidence)

Single-field signatures force awkward workarounds. Multi-field signatures are declarative and type-safe.

### Why Stateful Threads?

Agents often need multi-turn context ("fix the bug" → "write tests for it"). Stateful threads make this natural. Want fresh context? Create a new agent instance.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Test Structure

Tests are co-located in `src/tests/unit/`. The pre-commit hook enforces this:

```
src/
├── codex_dspy/
│   ├── adapter.py
│   └── agent.py
└── tests/
    └── unit/
        └── test_adapter.py
```

## Contributing

Issues and PRs welcome!

## License

See LICENSE file.

## Related Documentation

- [Codex SDK API Reference](./docs/CODEX_SDK_API_SURFACE.md)
- [Codex Architecture](./docs/CODEX_ARCHITECTURE.md)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
