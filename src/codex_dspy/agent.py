"""CodexAgent - DSPy module wrapping OpenAI Codex SDK.

This module provides a signature-driven interface to the Codex agent SDK.
Each CodexAgent instance maintains a stateful thread that accumulates context
across multiple forward() calls.

Uses a two-turn pattern:
- Turn 1: Natural task execution (agent does work)
- Turn 2: Structured output extraction (agent formats findings)
"""

import json
import re
import types
from typing import Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel

import dspy
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature

from codex import Codex, CodexOptions, SandboxMode, ThreadOptions, TurnOptions
from codex_dspy.adapter import CodexAdapter


def _combine_usage(usage1, usage2):
    """Combine token usage from two turns.

    Args:
        usage1: Usage from first turn (may be None)
        usage2: Usage from second turn (may be None)

    Returns:
        Combined usage with summed token counts, or whichever is not None
    """
    if usage1 is None:
        return usage2
    if usage2 is None:
        return usage1

    # Both exist - sum the token counts
    # Create a new usage-like object with combined counts
    from codex import Usage
    return Usage(
        input_tokens=(usage1.input_tokens or 0) + (usage2.input_tokens or 0),
        output_tokens=(usage1.output_tokens or 0) + (usage2.output_tokens or 0),
        cached_input_tokens=(usage1.cached_input_tokens or 0) + (usage2.cached_input_tokens or 0),
    )


def _strip_json_fences(text: str) -> str:
    """Strip markdown JSON fences from response if present.

    Handles:
        ```json\n{...}\n```
        ```\n{...}\n```
        {..} (no fences - returned as-is)
    """
    text = text.strip()

    # Pattern for ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r'^```(?:json)?\s*\n?(.*?)\n?```$', re.DOTALL)
    match = fence_pattern.match(text)
    if match:
        return match.group(1).strip()

    return text


def _is_all_str_outputs(signature: Signature) -> bool:
    """Check if all output fields are str or Optional[str]."""
    for field in signature.output_fields.values():
        annotation = field.annotation
        if annotation == str:
            continue
        origin = get_origin(annotation)
        # Handle both typing.Union and types.UnionType (PEP 604: str | None)
        if origin is Union or origin is types.UnionType:
            args = get_args(annotation)
            if len(args) == 2 and str in args and type(None) in args:
                continue
        return False
    return True


def _build_output_schema(signature: Signature) -> dict[str, Any]:
    """Build a combined JSON schema for all output fields."""
    properties = {}
    required = []

    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if annotation == str:
            properties[name] = {"type": "string"}
        elif hasattr(annotation, "model_json_schema"):
            # Pydantic model
            properties[name] = annotation.model_json_schema()
        else:
            # Fallback - try to get schema via pydantic TypeAdapter
            from pydantic import TypeAdapter
            properties[name] = TypeAdapter(annotation).json_schema()

        # Check if required (not Optional)
        # Handle both typing.Union and types.UnionType (PEP 604: str | None)
        origin = get_origin(annotation)
        is_optional = (origin is Union or origin is types.UnionType) and type(None) in get_args(annotation)
        if not is_optional:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema


class CodexAgent(dspy.Module):
    """DSPy module for Codex SDK integration.

    Creates a stateful agent where each instance maintains one conversation thread.
    Multiple forward() calls on the same instance continue the same conversation.

    Supports multiple input and output fields. Uses a two-turn pattern:
    - Turn 1: Agent receives task naturally and does work
    - Turn 2: Agent formats findings into structured output

    Args:
        signature: DSPy signature with any number of input/output fields
        working_directory: Directory where Codex agent will execute commands
        model: Model to use. Defaults to "gpt-5.1-codex-max".
        sandbox_mode: Execution sandbox level (READ_ONLY, WORKSPACE_WRITE, DANGER_FULL_ACCESS)
        skip_git_repo_check: Allow non-git directories as working_directory
        api_key: OpenAI API key (falls back to CODEX_API_KEY env var)
        base_url: API base URL (falls back to OPENAI_BASE_URL env var)
        codex_path_override: Override path to codex binary (for testing)

    Example with multiple fields:
        >>> class BugReport(BaseModel):
        ...     severity: str
        ...     description: str
        >>> sig = dspy.Signature(
        ...     "code: str, context: str -> bugs: list[BugReport], summary: str",
        ...     "Analyze code for bugs"
        ... )
        >>> agent = CodexAgent(sig, working_directory=".")
        >>> result = agent(code="def foo(): ...", context="Production code")
        >>> print(result.bugs)    # list[BugReport]
        >>> print(result.summary) # str
        >>> print(result.trace)   # execution trace
    """

    def __init__(
        self,
        signature: str | type[Signature],
        working_directory: str,
        model: Optional[str] = None,
        sandbox_mode: Optional[SandboxMode] = None,
        skip_git_repo_check: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        codex_path_override: Optional[str] = None,
    ):
        super().__init__()

        # Ensure signature is valid
        self.signature = ensure_signature(signature)

        # Validate: at least 1 input and 1 output field
        if len(self.signature.input_fields) < 1:
            raise ValueError(
                "CodexAgent requires at least 1 input field.\n"
                "Example: dspy.Signature('message:str -> answer:str')"
            )

        if len(self.signature.output_fields) < 1:
            raise ValueError(
                "CodexAgent requires at least 1 output field.\n"
                "Example: dspy.Signature('message:str -> answer:str')"
            )

        # Create adapter for formatting
        self.adapter = CodexAdapter()

        # Create Codex client
        self.client = Codex(
            options=CodexOptions(
                api_key=api_key,
                base_url=base_url,
                codex_path_override=codex_path_override,
            )
        )

        # Start thread (1 agent instance = 1 stateful thread)
        self.thread = self.client.start_thread(
            options=ThreadOptions(
                working_directory=working_directory,
                model=model,
                sandbox_mode=sandbox_mode,
                skip_git_repo_check=skip_git_repo_check,
            )
        )

    def forward(self, **kwargs) -> Prediction:
        """Execute agent with input fields.

        Args:
            **kwargs: Must contain all input fields specified in signature

        Returns:
            Prediction with:
                - All output fields (typed according to signature)
                - trace: list[ThreadItem] - chronological items (commands, files, etc.)
                - usage: Usage - token counts (input_tokens, cached_input_tokens, output_tokens)

        Raises:
            ValueError: If parsing fails for typed outputs
        """
        # Validate all input fields are provided
        for field_name in self.signature.input_fields:
            if field_name not in kwargs:
                raise ValueError(f"Missing required input field: {field_name}")

        # Turn 1: Natural task execution
        turn1_prompt = self.adapter.format_turn1(self.signature, kwargs)
        task_result = self.thread.run(turn1_prompt)

        # Check if we need structured output extraction
        if _is_all_str_outputs(self.signature):
            # All outputs are strings - parse from natural response
            # For single string output, just return the response
            if len(self.signature.output_fields) == 1:
                output_name = list(self.signature.output_fields.keys())[0]
                return Prediction(
                    **{output_name: task_result.final_response},
                    trace=task_result.items,
                    usage=task_result.usage,
                )
            else:
                # Multiple string outputs - need extraction turn
                turn2_prompt = self.adapter.format_turn2(self.signature)
                extract_result = self.thread.run(turn2_prompt)
                parsed = self.adapter.parse(self.signature, extract_result.final_response)

                return Prediction(
                    **parsed,
                    trace=task_result.items + extract_result.items,
                    usage=_combine_usage(task_result.usage, extract_result.usage),
                )
        else:
            # Need structured output - Turn 2 with JSON schema
            turn2_prompt = self.adapter.format_turn2_json(self.signature)
            output_schema = _build_output_schema(self.signature)
            turn_options = TurnOptions(output_schema=output_schema)

            extract_result = self.thread.run(turn2_prompt, turn_options)

            # Parse JSON response (strip fences if present)
            try:
                json_str = _strip_json_fences(extract_result.final_response)
                raw_output = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON response: {e}\n"
                    f"Response: {extract_result.final_response[:500]}"
                ) from e

            # Convert to typed outputs
            parsed_outputs = {}
            for name, field in self.signature.output_fields.items():
                value = raw_output.get(name)
                annotation = field.annotation

                if value is None:
                    parsed_outputs[name] = None
                elif get_origin(annotation) is list:
                    # Check for list[PydanticModel] - must check BEFORE direct model check
                    inner_type = get_args(annotation)[0] if get_args(annotation) else None
                    if inner_type and hasattr(inner_type, "model_validate") and isinstance(value, list):
                        parsed_outputs[name] = [inner_type.model_validate(v) for v in value]
                    else:
                        parsed_outputs[name] = value
                elif hasattr(annotation, "model_validate"):
                    # Direct Pydantic model
                    if isinstance(value, dict):
                        parsed_outputs[name] = annotation.model_validate(value)
                    else:
                        parsed_outputs[name] = value
                else:
                    parsed_outputs[name] = value

            return Prediction(
                **parsed_outputs,
                trace=task_result.items + extract_result.items,
                usage=_combine_usage(task_result.usage, extract_result.usage),
            )

    @property
    def thread_id(self) -> Optional[str]:
        """Get thread ID for this agent instance.

        The thread ID is assigned after the first forward() call.
        Useful for debugging and visibility into the conversation state.

        Returns:
            Thread ID string, or None if no forward() calls have been made yet
        """
        return self.thread.id
