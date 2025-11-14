"""ClaudeAgent - DSPy module wrapping Claude Agent SDK.

This module provides a signature-driven interface to the Claude Agent SDK.
Each ClaudeAgent instance maintains a stateful session that accumulates context
across multiple forward() calls.

Supports both string outputs (via claude-agent-sdk) and Pydantic outputs
(via direct Anthropic API with tool-calling for maximum reliability).
"""

import json
import os
from typing import Any, Union, get_args, get_origin

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    query,
)
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature

from codex_dspy.base import BaseAgent, _is_pydantic_model, _is_pydantic_type


class ClaudeAgent(BaseAgent):
    """DSPy module for Claude Agent SDK integration.

    Supports two modes:
    1. String outputs: Direct string responses
    2. Pydantic outputs: Structured JSON with prompt engineering (~80-86% reliability)

    For Pydantic mode, uses prompt engineering with XML tags and JSON schema
    (similar to TypeScript driver-agent pattern). Claude is instructed to wrap
    JSON in <response> tags, and the response is parsed and validated.

    Args:
        signature: DSPy signature (at least 1 input field, exactly 1 output field)
        working_directory: Directory where Claude agent will execute commands (optional for Pydantic mode)
        model: Model to use (e.g., "claude-sonnet-4-5"). Defaults to "claude-sonnet-4-5-20250116".
        system_prompt: System prompt to guide agent behavior
        permission_mode: Permission level ("default", "acceptEdits", "plan", "bypassPermissions")
        allowed_tools: List of allowed tool names (e.g., ["mcp__server__tool"])
        max_turns: Maximum number of turns for agent autonomy
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
        base_url: API base URL (falls back to ANTHROPIC_BASE_URL env var)

    Examples:
        String output mode:
        >>> sig = dspy.Signature('message:str -> answer:str')
        >>> agent = ClaudeAgent(sig, working_directory=".")
        >>> result = agent(message="What files are in this directory?")
        >>> print(result.answer)  # str response

        Pydantic output mode (prompt engineering):
        >>> from pydantic import BaseModel
        >>> class Analysis(BaseModel):
        ...     sentiment: str
        ...     confidence: float
        >>> class AnalyzeSignature(dspy.Signature):
        ...     text: str = dspy.InputField()
        ...     analysis: Analysis = dspy.OutputField()
        >>> agent = ClaudeAgent(AnalyzeSignature)
        >>> result = agent(text="I love this!")
        >>> print(result.analysis.sentiment)  # Pydantic model
    """

    def __init__(
        self,
        signature: str | type[Signature],
        working_directory: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        permission_mode: str | None = None,
        allowed_tools: list[str] | None = None,
        max_turns: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        # Initialize base class (handles signature validation and field extraction)
        super().__init__(signature)

        # Store configuration
        self.model = model or "claude-sonnet-4-5-20250116"
        self.system_prompt = system_prompt
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        # Determine mode based on output type
        self._use_pydantic_mode = _is_pydantic_type(self.output_type)

        if self._use_pydantic_mode:
            # Pydantic mode: Use prompt engineering with schema
            if working_directory is None:
                working_directory = "."  # Default for Pydantic mode

            # Extract the actual Pydantic model from the type annotation
            self._pydantic_model = self._extract_pydantic_model(self.output_type)

            # Use claude-agent-sdk for both modes
            self.options = ClaudeAgentOptions(
                cwd=working_directory,
                system_prompt=system_prompt,
                permission_mode=permission_mode,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                continue_conversation=True,
            )
            self._session_id: str | None = None
        else:
            # String mode: Use claude-agent-sdk
            if working_directory is None:
                raise ValueError("working_directory is required for string output mode.")
            self.options = ClaudeAgentOptions(
                cwd=working_directory,
                system_prompt=system_prompt,
                permission_mode=permission_mode,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                continue_conversation=True,  # Enable multi-turn support
            )
            self._session_id: str | None = None

    def _extract_pydantic_model(self, annotation: Any) -> type:
        """Extract the Pydantic model class from type annotation.

        Handles both direct Pydantic models and Optional[PydanticModel].

        Args:
            annotation: Type annotation that contains a Pydantic model

        Returns:
            The Pydantic BaseModel class

        Raises:
            ValueError: If no Pydantic model found in annotation
        """
        if _is_pydantic_model(annotation):
            return annotation

        # Handle Optional[PydanticModel]
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            for arg in args:
                if arg is not type(None) and _is_pydantic_model(arg):
                    return arg

        raise ValueError(f"Could not extract Pydantic model from annotation: {annotation}")

    def _augment_prompt_with_schema(self, prompt: str, schema: dict[str, Any]) -> str:
        """Augment prompt with JSON schema instructions and XML tag wrapper.

        Similar to TypeScript driver-agent.ts approach.

        Args:
            prompt: Original prompt
            schema: JSON schema from Pydantic model

        Returns:
            Augmented prompt with schema and formatting instructions
        """
        return f"""{prompt}

You must respond with the JSON wrapped in <response> tags like this:
<response>{{raw JSON response}}</response>

The JSON must conform to this schema:
{json.dumps(schema, indent=2)}"""

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from <response></response> tags.

        Args:
            response: Claude's response text

        Returns:
            Extracted JSON string

        Raises:
            ValueError: If no JSON found in response tags
        """
        import re

        # Try to find JSON in <response> tags
        match = re.search(r"<response>\s*(\{.*?\})\s*</response>", response, re.DOTALL)
        if match:
            return match.group(1)

        # Fallback: Try to find raw JSON without tags
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError(
            f"No JSON found in response. Expected <response>{{json}}</response> tags.\n"
            f"Response preview: {response[:200]}..."
        )

    def forward(self, **kwargs) -> Prediction:
        """Execute agent with input message.

        Dispatches to either string mode or Pydantic mode (both use claude-agent-sdk).

        Args:
            **kwargs: Must contain the input field specified in signature

        Returns:
            Prediction with:
                - Typed output field (string or Pydantic model, name from signature)
                - trace: list[dict] - converted message trace
                - session_id: str - Session ID
                - cost_usd: float - Total cost in USD
                - num_turns: int - Number of turns

        Raises:
            ValueError: If Claude returns empty/invalid response or validation fails
        """
        message = self._format_input_message(kwargs)

        # Append output field description if present
        message = self._append_output_description(message)

        # For Pydantic mode, augment with schema
        if self._use_pydantic_mode:
            schema = self._pydantic_model.model_json_schema()
            message = self._augment_prompt_with_schema(message, schema)

        # Run async query in sync context
        return anyio.run(self._async_forward, message)

    async def _async_forward(self, message: str) -> Prediction:
        """Async implementation of forward() using claude-agent-sdk.

        Handles both string and Pydantic output modes.

        Args:
            message: Input message to send to Claude

        Returns:
            Prediction with typed output and metadata
        """
        # Collect messages
        text_responses = []
        thinking_blocks = []
        tool_uses = []
        final_result = None

        # Stream messages from Claude
        async for msg in query(prompt=message, options=self.options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_responses.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        thinking_blocks.append(block.thinking)
                    elif isinstance(block, ToolUseBlock):
                        tool_uses.append(
                            {
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
            elif isinstance(msg, ResultMessage):
                final_result = msg
                if msg.session_id:
                    self._session_id = msg.session_id

        # Get final response - prefer text blocks, fallback to result field
        final_response = "\n".join(text_responses) if text_responses else ""

        # If no text blocks, check if ResultMessage has a result
        if (
            not final_response
            and final_result
            and hasattr(final_result, "result")
            and final_result.result
        ):
            final_response = final_result.result

        if not final_response:
            raise ValueError("Claude returned empty response")

        # Build trace
        trace = self._build_trace(thinking_blocks, tool_uses, text_responses)

        # Handle Pydantic mode: extract and validate JSON
        if self._use_pydantic_mode:
            try:
                # Extract JSON from response
                json_str = self._extract_json_from_response(final_response)

                # Parse and validate with Pydantic
                parsed_output = self._pydantic_model.model_validate_json(json_str)

                # Return prediction with Pydantic model
                return Prediction(
                    **{self.output_field: parsed_output},
                    trace=trace,
                    session_id=self._session_id,
                    num_turns=final_result.num_turns if final_result else None,
                    cost_usd=final_result.total_cost_usd if final_result else None,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to parse Pydantic output from response.\n"
                    f"Error: {e}\n"
                    f"Response preview: {final_response[:500]}..."
                ) from e

        # String mode: return as-is
        return Prediction(
            **{self.output_field: final_response},
            trace=trace,
            session_id=self._session_id,
            num_turns=final_result.num_turns if final_result else None,
            cost_usd=final_result.total_cost_usd if final_result else None,
        )

    def _build_trace(
        self,
        thinking_blocks: list[str],
        tool_uses: list[dict],
        text_responses: list[str],
    ) -> list[dict]:
        """Convert Claude messages to trace items for observability.

        Args:
            thinking_blocks: List of thinking content
            tool_uses: List of tool use dictionaries
            text_responses: List of text responses

        Returns:
            List of trace item dictionaries
        """
        trace = []

        for thinking in thinking_blocks:
            trace.append(
                {
                    "type": "reasoning",
                    "content": thinking,
                }
            )

        for tool_use in tool_uses:
            trace.append(
                {
                    "type": "tool_call",
                    "name": tool_use.get("name"),
                    "input": tool_use.get("input"),
                }
            )

        for text in text_responses:
            trace.append(
                {
                    "type": "agent_message",
                    "text": text,
                }
            )

        return trace

    @property
    def session_id(self) -> str | None:
        """Get session ID for this agent instance.

        The session ID is assigned after the first forward() call.
        Useful for debugging and visibility into the conversation state.

        Returns:
            Session ID string, or None if no forward() calls have been made yet
        """
        return self._session_id
