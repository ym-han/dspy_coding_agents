"""Multi-Agent DSPy Modules - Codex & Claude SDK Integration.

This package provides signature-driven interfaces to agent SDKs,
enabling stateful agentic workflows through DSPy signatures.

Supported Agents:
- CodexAgent: OpenAI Codex SDK wrapper
- CodexAdapter: Two-turn adapter for Codex workflows
- ClaudeAgent: Anthropic Claude Agent SDK wrapper
"""

from codex_dspy.adapter import CodexAdapter
from codex_dspy.agent import CodexAgent
from codex_dspy.claude_agent import ClaudeAgent
from codex_dspy.base import BaseAgent

__all__ = ["CodexAgent", "CodexAdapter", "ClaudeAgent", "BaseAgent"]
__version__ = "0.1.0"
