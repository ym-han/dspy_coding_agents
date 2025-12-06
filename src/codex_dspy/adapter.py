"""CodexAdapter - Two-turn adapter for agentic workflows.

Turn 1: Natural task prompt (agent does work)
Turn 2: Structured output extraction (agent formats findings)

Based on DSPy's TwoStepAdapter and BAMLAdapter patterns.
"""

import inspect
import json
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


# --- Schema Rendering (from BAML) ---

def _render_type_str(
    annotation: Any,
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Render a type annotation into a simplified, human-readable string.

    Examples:
        str -> "string"
        int -> "int"
        list[Bug] -> "[\n  { ... }\n]"
        Literal["a", "b"] -> '"a" or "b"'
        Optional[str] -> "string or null"
    """
    # Primitives
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"

    # Pydantic models
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_simplified_schema(annotation, indent, seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Optional[T] or T | None (handles both typing.Union and types.UnionType)
    if origin is Union or origin is types.UnionType:
        non_none_args = [arg for arg in args if arg is not type(None)]
        type_render = " or ".join([_render_type_str(arg, indent) for arg in non_none_args])
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    # Literal["a", "b", ...]
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    # list[T]
    if origin is list:
        inner_type = args[0] if args else Any
        # Direct Pydantic model
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            inner_schema = _build_simplified_schema(inner_type, indent + 1, seen_models)
            current_indent = "  " * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        # list[Model | None] - Optional Pydantic model
        inner_origin = get_origin(inner_type)
        if inner_origin is Union or inner_origin is types.UnionType:
            inner_args = get_args(inner_type)
            non_none = [a for a in inner_args if a is not type(None)]
            if len(non_none) == 1 and inspect.isclass(non_none[0]) and issubclass(non_none[0], BaseModel):
                inner_schema = _build_simplified_schema(non_none[0], indent + 1, seen_models)
                current_indent = "  " * indent
                return f"[\n{inner_schema},  // or null\n{current_indent}]"
        # Other list types (primitives, nested lists, etc.)
        return f"{_render_type_str(inner_type, indent)}[]"

    # dict[K, V]
    if origin is dict:
        key_type = _render_type_str(args[0], indent) if args else "string"
        val_type = _render_type_str(args[1], indent) if len(args) > 1 else "any"
        return f"dict[{key_type}, {val_type}]"

    # Fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _build_simplified_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Build a simplified, human-readable schema from a Pydantic model.

    Example output:
        {
          # Bug severity level
          severity: "low" or "medium" or "high",
          # Where in the code
          location: string,
        }
    """
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        return f"{pydantic_model.__name__} (recursive)"

    seen_models = seen_models | {pydantic_model}

    lines = []
    current_indent = "  " * indent
    next_indent = "  " * (indent + 1)

    lines.append(f"{current_indent}{{")

    fields = pydantic_model.model_fields
    if not fields:
        lines.append(f"{next_indent}# No fields defined")

    for name, field in fields.items():
        # Add description as comment
        if field.description:
            lines.append(f"{next_indent}# {field.description}")

        rendered_type = _render_type_str(field.annotation, indent=indent + 1, seen_models=seen_models)
        lines.append(f"{next_indent}{name}: {rendered_type},")

    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


# --- Field Description (from TwoStepAdapter) ---

def get_annotation_name(annotation: Any) -> str:
    """Get a human-readable name for a type annotation."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)

    if origin is Literal:
        args_str = ", ".join(f'"{a}"' if isinstance(a, str) else str(a) for a in args)
        return f"Literal[{args_str}]"

    args_str = ", ".join(get_annotation_name(a) for a in args)
    origin_name = getattr(origin, "__name__", str(origin))
    return f"{origin_name}[{args_str}]"


def format_field_description(fields: dict[str, FieldInfo]) -> str:
    """Format field descriptions as a numbered list.

    Example:
        1. `context` (CodeContext): The code to analyze
        2. `user_request` (str): What to look for
    """
    descriptions = []
    for idx, (name, field) in enumerate(fields.items(), 1):
        type_name = get_annotation_name(field.annotation)
        desc = f": {field.description}" if field.description else ""
        descriptions.append(f"{idx}. `{name}` ({type_name}){desc}")
    return "\n".join(descriptions)


# --- CodexAdapter ---

class CodexAdapter:
    """Two-turn adapter for Codex agentic workflows.

    Turn 1 (format_turn1): Natural task prompt
        - Describes input and output fields
        - Shows input values
        - Includes task instructions
        - Agent works naturally, no structured output required

    Turn 2 (format_turn2): Structured extraction
        - BAML-style schemas for each output field
        - Agent formats its findings into the structure
    """

    def format_turn1(
        self,
        signature,  # DSPy Signature
        inputs: dict[str, Any],
    ) -> str:
        """Format the task turn prompt.

        Agent receives this and does its work naturally.
        Output fields are declared (so agent knows the goal) but not structured.
        """
        parts = []

        # Input field descriptions
        if signature.input_fields:
            parts.append("As input, you are provided with:")
            parts.append(format_field_description(signature.input_fields))
            parts.append("")

        # Output field descriptions (declare the goal)
        if signature.output_fields:
            parts.append("Your task is to produce:")
            parts.append(format_field_description(signature.output_fields))
            parts.append("")

        # Task instructions from signature
        if signature.instructions:
            parts.append(f"Instructions: {signature.instructions}")
            parts.append("")

        # Separator
        parts.append("---")
        parts.append("")

        # Input values
        for name, field in signature.input_fields.items():
            if name in inputs:
                value = inputs[name]
                # Format Pydantic models as JSON
                if isinstance(value, BaseModel):
                    formatted = value.model_dump_json(indent=2)
                elif isinstance(value, (dict, list)):
                    formatted = json.dumps(value, indent=2, ensure_ascii=False)
                else:
                    formatted = str(value)
                parts.append(f"{name}: {formatted}")
                parts.append("")

        return "\n".join(parts).strip()

    def format_turn2(self, signature) -> str:
        """Format the extraction turn prompt.

        Agent receives this after completing the task.
        Uses BAML-style schemas to request structured output.
        """
        parts = []
        parts.append("Now provide your findings in the following format:")
        parts.append("")

        for name, field in signature.output_fields.items():
            parts.append(f"[[ ## {name} ## ]]")

            annotation = field.annotation
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                # Pydantic model - show simplified schema
                schema = _build_simplified_schema(annotation, indent=0)
                parts.append(schema)
            elif get_origin(annotation) is list:
                # list[T] - show array schema
                inner = get_args(annotation)[0] if get_args(annotation) else Any
                if inspect.isclass(inner) and issubclass(inner, BaseModel):
                    # list[Model]
                    inner_schema = _build_simplified_schema(inner, indent=1)
                    parts.append(f"[\n{inner_schema}\n]")
                else:
                    # Check for list[Model | None]
                    inner_origin = get_origin(inner)
                    if inner_origin is Union or inner_origin is types.UnionType:
                        inner_args = get_args(inner)
                        non_none = [a for a in inner_args if a is not type(None)]
                        if len(non_none) == 1 and inspect.isclass(non_none[0]) and issubclass(non_none[0], BaseModel):
                            inner_schema = _build_simplified_schema(non_none[0], indent=1)
                            parts.append(f"[\n{inner_schema},  // or null\n]")
                            continue
                    # Other list types
                    parts.append(f"{_render_type_str(inner)}[]")
            else:
                # Primitive or other type
                parts.append(f"<{_render_type_str(annotation)}>")

            parts.append("")

        parts.append("[[ ## completed ## ]]")
        return "\n".join(parts)

    def format_turn2_json(self, signature) -> str:
        """Alternative: request JSON output for Turn 2.

        Use this if you want to use output_schema with the LLM
        instead of parsing [[ ## field ## ]] markers.
        """
        parts = []
        parts.append("Now provide your findings as JSON with the following structure:")
        parts.append("")
        parts.append("```json")
        parts.append("{")

        field_lines = []
        for name, field in signature.output_fields.items():
            schema = _render_type_str(field.annotation, indent=1)
            desc = f"  // {field.description}" if field.description else ""
            field_lines.append(f'  "{name}": {schema}{desc}')

        parts.append(",\n".join(field_lines))
        parts.append("}")
        parts.append("```")

        return "\n".join(parts)

    def parse(self, signature, completion: str) -> dict[str, Any]:
        """Parse [[ ## field ## ]] markers from completion.

        Returns a dict mapping field names to their string values.
        Caller is responsible for type conversion (e.g., JSON parsing for Pydantic).
        """
        import re

        field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")

        sections = [(None, [])]
        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                header = match.group(1)
                remaining = line[match.end():].strip()
                sections.append((header, [remaining] if remaining else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {}
        for name, value in sections:
            if name and name in signature.output_fields and name not in fields:
                if name == "completed":
                    continue
                fields[name] = value

        return fields
