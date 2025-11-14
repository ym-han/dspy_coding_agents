# codex_dspy - DSPy Modules for Codex & Claude

DSPy modules that wrap agent SDKs with signature-driven interfaces. Each agent instance maintains a stateful conversation (thread/session), making them perfect for multi-turn agentic workflows.

## Supported Agents

- **CodexAgent** - OpenAI Codex SDK wrapper (two-turn pattern, native Pydantic support)
- **ClaudeAgent** - Anthropic Claude Agent SDK wrapper (string + Pydantic via prompt engineering)

## Features

### CodexAgent
- **Two-turn pattern** - Natural task execution + structured extraction
- **Stateful threads** - Each agent instance = one conversation thread
- **Typed outputs** - Pydantic models, primitives, lists - all work naturally
- **Execution trace** - Full visibility into commands, file changes, reasoning
- **DSPy-native** - Standard signatures, based on TwoStepAdapter patterns

### ClaudeAgent
- **Dual-mode architecture** - String and Pydantic outputs via claude-agent-sdk
- **Prompt engineering for Pydantic** - XML tags + JSON schema approach (~80-86% reliability)
- **Cost tracking** - USD cost tracking for both modes
- **Permission modes** - Control execution permissions
- **System prompts** - Customize agent behavior
- **No extra API requirements** - Uses existing Claude authentication

## Installation

```bash
# Install dependencies
uv sync

# For development (includes pytest, pre-commit)
uv sync --extra dev

# For CodexAgent: Ensure codex CLI is available
which codex

# For ClaudeAgent: Ensure Claude Code CLI is available
which claude  # Should be available after installing claude-agent-sdk
```

**Environment Variables:**
- `CODEX_API_KEY` or `OPENAI_API_KEY` - For CodexAgent
- `ANTHROPIC_API_KEY` - For ClaudeAgent

## Quick Start

### CodexAgent

```python
import dspy
from codex_dspy import CodexAgent

# Simple string signature
sig = dspy.Signature('task: str -> result: str')
agent = CodexAgent(sig, working_directory=".")

result = agent(task="What files are in this directory?")
print(result.result)  # String response
print(result.trace)   # Execution trace
print(result.usage)   # Token counts
```

### ClaudeAgent

```python
import dspy
from codex_dspy import ClaudeAgent

# Define signature
sig = dspy.Signature('message:str -> answer:str')

# Create agent
agent = ClaudeAgent(sig, working_directory=".")

result = agent(message="What files are in this directory?")
print(result.answer)     # String response
print(result.trace)      # Execution trace (dict-based)
print(result.cost_usd)   # Cost in USD
print(result.session_id) # Session ID for debugging
```

### CodexAgent - With Pydantic Models

```python
from typing import Literal
from pydantic import BaseModel, Field

class BugReport(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(description="Bug severity")
    location: str = Field(description="File and line number")
    description: str = Field(description="What the bug does")

sig = dspy.Signature(
    "code: str, context: str -> bugs: list[BugReport], summary: str",
    "Analyze code for bugs"
)

agent = CodexAgent(sig, working_directory=".")
result = agent(
    code="def divide(a, b): return a / b",
    context="Production calculator module"
)

print(result.summary)           # str
print(result.bugs)              # list[BugReport]
print(result.bugs[0].severity)  # Typed access
```

### ClaudeAgent - With Pydantic Models

```python
from pydantic import BaseModel, Field
from codex_dspy import ClaudeAgent

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0-1", ge=0, le=1)
    key_points: list[str]

class AnalyzeSignature(dspy.Signature):
    text: str = dspy.InputField()
    analysis: SentimentAnalysis = dspy.OutputField()

# working_directory optional for Pydantic mode
agent = ClaudeAgent(AnalyzeSignature)

result = agent(text="I love this product!")
print(result.analysis.sentiment)     # "positive"
print(result.analysis.confidence)    # 0.95
print(result.analysis.key_points)    # ["love", ...]
```

**Note:** CodexAgent uses native SDK support for Pydantic (~100% reliability). ClaudeAgent uses prompt engineering (~80-86% reliability).

## How It Works: Two-Turn Pattern (CodexAgent)

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

After the task completes, the agent formats its findings using TypeScript syntax (LLMs are heavily trained on TypeScript, making this format intuitive):

```
Respond with a TypeScript value matching this type:

```typescript
interface BugReport {
  /** Bug severity */
  severity: "low" | "medium" | "high";
  /** File and line number */
  location: string;
  /** What the bug does */
  description: string;
}

type Response = {
  /** Bugs found in the code */
  bugs: BugReport[];
  /** Overall analysis summary */
  summary: string;
};
```

This separation keeps the agent in-distribution during the actual work.

### Static Examples

You can provide static examples that show the LLM what good output looks like. These are defined on the signature and survive DSPy optimization (unlike few-shot demos which optimizers can replace):

```python
class BugReport(BaseModel):
    severity: Literal["low", "medium", "high"]
    location: str
    description: str

class CodeAnalysis(dspy.Signature):
    """Analyze code for bugs."""

    code: str = dspy.InputField()
    bugs: list[BugReport] = dspy.OutputField(desc="Bugs found")
    summary: str = dspy.OutputField(desc="Overall summary")

    # Static examples - shown in Turn 2 prompt
    class Examples:
        outputs = [
            {
                "bugs": [BugReport(severity="high", location="main.py:42", description="SQL injection")],
                "summary": "Found 1 critical security issue",
            },
            {
                "bugs": [],
                "summary": "No issues found",
            },
        ]
```

This renders in Turn 2 as:

```
Example outputs:
```typescript
// Example 1:
{
  bugs: [
    {
      severity: "high",
      location: "main.py:42",
      description: "SQL injection",
    },
  ],
  summary: "Found 1 critical security issue",
}

// Example 2:
{
  bugs: [],
  summary: "No issues found",
}
```

**Two types of examples:**

| Type | Location | Purpose | Survives Optimization? |
|------|----------|---------|------------------------|
| **Static** | `signature.Examples.outputs` | Format documentation - "here's what good output looks like" | Yes |
| **Dynamic** | `predictor.demos` | Few-shot learning for the task itself | No (optimizers replace) |

Static examples only appear in Turn 2 (extraction) since they demonstrate output format, not task execution.

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

# Format Turn 2 - TypeScript format (preferred)
turn2 = adapter.format_turn2_typescript(signature)

# Alternative Turn 2 formats:
turn2_markers = adapter.format_turn2(signature)      # BAML-style [[ ## field ## ]] markers
turn2_json = adapter.format_turn2_json(signature)   # JSON schema format

# Parse [[ ## field ## ]] markers from response
parsed = adapter.parse(signature, completion)
```

The TypeScript format (`format_turn2_typescript`) is preferred because:
- LLMs are heavily trained on TypeScript syntax
- JSDoc comments provide field descriptions naturally
- Output is parseable with `json5` (handles trailing commas, unquoted keys)

### ClaudeAgent

```python
class ClaudeAgent(BaseAgent):
    def __init__(
        self,
        signature: str | type[Signature],  # At least 1 input, exactly 1 output field
        working_directory: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        permission_mode: Optional[str] = None,  # "default", "acceptEdits", "plan", "bypassPermissions"
        allowed_tools: Optional[list[str]] = None,
        max_turns: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    )
```

#### Methods

##### `forward(**kwargs) -> Prediction`

**Returns** a `Prediction` with:
- Typed output field (string or Pydantic model, name from signature)
- `trace` - `list[dict]` - Converted message trace
- `session_id` - `str` - Session ID
- `cost_usd` - `float` - Total cost in USD
- `num_turns` - `int` - Number of turns

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

### Pattern 2: Complex Analysis

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

Tests are co-located under `src/tests/` and the pre-commit hook enforces the
two approved locations:

```
src/
├── codex_dspy/
│   ├── adapter.py
│   └── agent.py
└── tests/
    ├── unit/      # example-based tests
    └── property/  # hypothesis property tests
```

## Contributing

Issues and PRs welcome!

## License

See LICENSE file.

## Related Documentation

### CodexAgent
- [Codex SDK API Reference](./docs/CODEX_SDK_API_SURFACE.md)
- [Codex Architecture](./docs/CODEX_ARCHITECTURE.md)
- [Codex Quick Reference](./docs/CODEX_QUICK_REFERENCE.md)

### ClaudeAgent
- [Claude Integration Guide](./docs/CLAUDE_INTEGRATION.md)
- [Claude Basic Usage Example](./examples/claude_basic_usage.py)

### General
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Examples Directory](./examples/)
