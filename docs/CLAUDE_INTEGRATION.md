# Claude Integration Guide

This document explains how to use the ClaudeAgent module for integrating Anthropic's Claude Agent SDK with DSPy.

## Overview

`ClaudeAgent` is a DSPy module providing a signature-driven interface to Anthropic's Claude, with support for both string and structured (Pydantic) outputs.

**Features:**
- **Dual-mode architecture:** String and Pydantic outputs both via claude-agent-sdk
- **Prompt engineering for structured outputs** using XML tags + JSON schema (~80-86% reliability)
- **DSPy optimizer compatible** for both output modes
- **Stateful sessions** for multi-turn conversations
- **Type-safe** Pydantic validation for structured data
- **No additional API requirements** - uses existing Claude authentication

## Prerequisites

1. **API Key**
   - Set `ANTHROPIC_API_KEY` environment variable
   - Or pass `api_key` parameter to ClaudeAgent

2. **Python Dependencies**
   - Installed automatically with this package:
     - `claude-agent-sdk>=0.1.0`
     - `anyio>=4.0.0`
     - `httpx>=0.27.0`

3. **Claude Code CLI**
   - Install from: https://github.com/anthropics/claude-agent-sdk-python
   - Verify installation: `claude --version`

## Quick Start

### String Output Mode

```python
import dspy
from codex_dspy import ClaudeAgent

# Create signature
sig = dspy.Signature("message:str -> answer:str")

# Create agent (working_directory required for string mode)
agent = ClaudeAgent(
    sig,
    working_directory=".",
    permission_mode="default",
)

# Call agent
result = agent(message="What files are in this directory?")

print(result.answer)       # String response
print(result.session_id)   # Session ID
print(result.cost_usd)     # Cost in USD
print(result.trace)        # Execution trace
```

### Pydantic Output Mode (Structured Output)

```python
import dspy
from pydantic import BaseModel, Field
from codex_dspy import ClaudeAgent

# Define Pydantic model
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0-1 confidence score", ge=0, le=1)
    key_points: list[str] = Field(description="Supporting evidence")

# Create signature with Pydantic output
class AnalyzeSignature(dspy.Signature):
    text: str = dspy.InputField()
    analysis: SentimentAnalysis = dspy.OutputField()

# Create agent (working_directory optional for Pydantic mode)
agent = ClaudeAgent(AnalyzeSignature)

# Call agent
result = agent(text="I love this product! It exceeded my expectations.")

# Access structured output
print(result.analysis.sentiment)      # "positive"
print(result.analysis.confidence)     # 0.95
print(result.analysis.key_points)     # ["love", "exceeded expectations"]
print(result.usage)                   # Token usage stats
print(result.cost_usd)                # Estimated cost
```

## Architecture: Dual-Mode Operation

ClaudeAgent automatically selects the appropriate implementation based on your signature's output type:

### String Mode
**Triggered when:** Output type is `str`
**Method:** Direct string responses from Claude
**Best for:**
- Agent tasks requiring file operations, command execution
- Multi-turn conversations with context preservation
- Development/debugging with agent tools
- General-purpose text generation

**Characteristics:**
- Stateful: Multiple calls continue same session
- Requires `working_directory` parameter
- Returns basic cost in USD (no token counts)
- Access to agent tools (Bash, Read, Write, etc.)

### Pydantic Mode (Prompt Engineering)
**Triggered when:** Output type is a Pydantic `BaseModel`
**Method:** Prompt engineering with XML tags + JSON schema
**Best for:**
- Structured data extraction
- Type-safe outputs with validation
- DSPy optimizer compatibility
- API responses requiring consistent format

**How it works:**
1. JSON schema generated from Pydantic model
2. Prompt augmented with schema and `<response>` tag instructions
3. Claude wraps JSON in `<response>{json}</response>` tags
4. Response extracted via regex and validated with Pydantic

**Characteristics:**
- ~80-86% reliability (prompt-based, no API enforcement)
- Stateful: Maintains conversation context
- `working_directory` optional (defaults to ".")
- Returns cost in USD (same as string mode)
- Same agent tools available

### When to Use Which Mode

| Use Case | Recommended Mode |
|----------|------------------|
| Extract structured data | Pydantic |
| Multi-step agent workflows | Either |
| DSPy optimization loops | Pydantic |
| File operations | Either |
| Type-safe validation | Pydantic |
| Debugging with tools | String |
| Flexible text output | String |
| Interactive sessions | String |

## Key Features

### 1. Stateful Sessions

Each `ClaudeAgent` instance maintains one conversation session. Multiple `forward()` calls continue the same conversation:

```python
agent = ClaudeAgent(sig, working_directory=".")

# First call
result1 = agent(message="Remember this: the project uses DSPy")

# Second call - context preserved
result2 = agent(message="What did I just tell you?")
# Claude will remember "the project uses DSPy"
```

**Fresh Context:** Create a new instance for independent conversations:

```python
agent1 = ClaudeAgent(sig, working_directory=".")
agent1(message="First conversation")

agent2 = ClaudeAgent(sig, working_directory=".")  # Fresh context
agent2(message="Second conversation")
```

### 2. Output Field Descriptions

Use DSPy's `OutputField` to guide Claude's response format:

```python
class AnalysisSignature(dspy.Signature):
    message: str = dspy.InputField()
    analysis: str = dspy.OutputField(
        desc="Detailed markdown analysis with sections: "
        "1) Overview, 2) Key findings, 3) Recommendations"
    )

agent = ClaudeAgent(AnalysisSignature, working_directory=".")
result = agent(message="Analyze this codebase")
```

The description is automatically appended to the message, guiding Claude to produce the desired format.

### 3. Permission Modes

Control Claude's execution permissions:

```python
agent = ClaudeAgent(
    sig,
    working_directory=".",
    permission_mode="default",      # Ask for confirmation
    # permission_mode="acceptEdits",  # Auto-accept file edits
    # permission_mode="plan",         # Planning mode
    # permission_mode="bypassPermissions",  # Full access
)
```

### 4. System Prompts

Customize Claude's behavior with system prompts:

```python
agent = ClaudeAgent(
    sig,
    working_directory=".",
    system_prompt="You are a code review expert. "
    "Focus on security and performance issues.",
)
```

### 5. Tool Control

Restrict which tools Claude can use:

```python
agent = ClaudeAgent(
    sig,
    working_directory=".",
    allowed_tools=[
        "Bash",
        "Read",
        "Glob",
        # "Write",  # Excluded - no file writing
    ],
)
```

## Configuration Options

### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `signature` | `str \| type[Signature]` | DSPy signature (1 input, 1 output) | Required |
| `working_directory` | `str` | Directory for agent execution | Required |
| `model` | `str \| None` | Model name (e.g., "claude-sonnet-4-5") | Claude default |
| `system_prompt` | `str \| None` | Custom system prompt | None |
| `permission_mode` | `str \| None` | Permission level | "default" |
| `allowed_tools` | `list[str] \| None` | Allowed tool names | All tools |
| `max_turns` | `int \| None` | Max autonomous turns | Unlimited |
| `api_key` | `str \| None` | API key (falls back to env var) | None |
| `base_url` | `str \| None` | API base URL | None |

### Prediction Output

The `forward()` method returns a `Prediction` with:

| Field | Type | Description |
|-------|------|-------------|
| `{output_field}` | `str` | Response (name from signature) |
| `trace` | `list[dict]` | Execution trace items |
| `session_id` | `str \| None` | Claude session ID |
| `cost_usd` | `float \| None` | Total cost in USD |
| `num_turns` | `int \| None` | Number of turns |

**Note:** Unlike CodexAgent, Claude doesn't provide token counts, only total cost.

## Comparison with CodexAgent

### Similarities

- Both inherit from `BaseAgent`
- Same signature requirements (1 input, 1 output)
- Stateful conversations (thread/session)
- Support output field descriptions
- Return `Prediction` with trace and usage

### Differences

| Feature | CodexAgent | ClaudeAgent |
|---------|------------|-------------|
| **Output Types** | String & Pydantic | String & Pydantic |
| **Pydantic Method** | Native SDK (output_schema) | Prompt engineering (XML tags) |
| **Pydantic Reliability** | ~100% (OpenAI Structured Outputs) | ~80-86% (prompt-based) |
| **Usage Metrics** | Token counts | Cost in USD |
| **Trace Format** | Typed items (Codex SDK) | Dict-based |
| **Configuration** | Sandbox modes | Permission modes |
| **Stateful Sessions** | Yes (threads) | Yes (both modes) |
| **Async/Sync** | Sync | anyio-wrapped async |
| **Agent Tools** | Via Codex SDK | Via claude-agent-sdk (both modes) |

## Current Limitations

1. **Pydantic Mode Limitations**
   - Lower reliability than CodexAgent (80-86% vs ~100%)
   - Depends on prompt following (no API-level schema enforcement)
   - JSON extraction via regex (can fail on malformed responses)
   - May include extra text outside `<response>` tags

2. **Both Modes**
   - Simplified trace format (dict-based, not typed)
   - No token count metrics (only cost in USD)
   - Not all message types captured in trace (thinking blocks, some tool results)

3. **General Limitations**
   - No MCP server configuration exposed
   - No streaming support yet
   - Cost is estimated, may vary from actual billing

## Troubleshooting

### Claude CLI Not Found

```
Error: Claude Code CLI not found
```

**Solution:** Install Claude Code CLI and ensure it's in your PATH:
```bash
pip install claude-agent-sdk
# Verify installation
claude --version
```

### API Key Issues

```
Error: No API key found
```

**Solution:** Set environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or pass directly:
```python
agent = ClaudeAgent(sig, working_directory=".", api_key="your-key")
```

### Empty Response

```
ValueError: Claude returned empty response
```

**Possible causes:**
- Message too complex
- Permission denied for required operations
- Claude couldn't produce valid output

**Solution:** Simplify the message or adjust `permission_mode`.

## Future Enhancements

Planned for future versions:

- [x] **Pydantic output support** (✓ Complete - v0.1.0)
- [ ] Complete trace conversion (thinking blocks, tool results)
- [ ] Stateful sessions for Pydantic mode
- [ ] MCP server configuration (string mode)
- [ ] Hook system integration (string mode)
- [ ] Streaming support (both modes)
- [ ] Session resumption across instances
- [ ] Conversation history management (Pydantic mode)

## Examples

Complete examples are available in the `examples/` directory:

**String Mode:**
- [`examples/claude_basic_usage.py`](../examples/claude_basic_usage.py) - String outputs, system prompts, stateful sessions

**Pydantic Mode:**
- [`examples/test_pydantic_output.py`](../examples/test_pydantic_output.py) - Structured outputs with simple and nested Pydantic models

**Topics covered:**
1. Simple string output (string mode)
2. Structured data extraction (Pydantic mode)
3. Output field descriptions (both modes)
4. Custom system prompts (both modes)
5. Fresh vs continued contexts (string mode)
6. Nested Pydantic models (Pydantic mode)
7. Field validation (Pydantic mode)

## Resources

- [Claude Agent SDK Python](https://github.com/anthropics/claude-agent-sdk-python)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Project README](../README.md)

## Getting Help

- **Issues:** Report bugs or request features on GitHub
- **Discussions:** Join the DSPy community
- **Documentation:** Check the main README and code comments
