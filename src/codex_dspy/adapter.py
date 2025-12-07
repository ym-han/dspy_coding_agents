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


# --- TypeScript Conversion ---

def _is_optional_type(annotation: Any) -> bool:
    """Check if a type annotation is optional (Union with None)."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(annotation)
    return False


def _ts_type(annotation: Any, seen: set[type] | None = None) -> str:
    """Convert Python type annotation to TypeScript type string."""
    seen = seen or set()

    # Primitives
    if annotation is str:
        return "string"
    if annotation is int or annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is type(None):
        return "null"

    # Pydantic model - just use the name (interface defined separately)
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation.__name__

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional / Union
    if origin is Union or origin is types.UnionType:
        parts = [_ts_type(a, seen) for a in args]
        return " | ".join(parts)

    # Literal
    if origin is Literal:
        return " | ".join(f'"{a}"' if isinstance(a, str) else str(a).lower() for a in args)

    # list / Array
    if origin is list:
        inner = _ts_type(args[0], seen) if args else "any"
        # Wrap union types in parens for array
        if " | " in inner:
            return f"Array<{inner}>"
        return f"{inner}[]"

    # dict / Record
    if origin is dict:
        key_type = _ts_type(args[0], seen) if args else "string"
        val_type = _ts_type(args[1], seen) if len(args) > 1 else "any"
        return f"Record<{key_type}, {val_type}>"

    # Fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return "any"


def _collect_models(annotation: Any, collected: set[type] | None = None) -> set[type]:
    """Recursively collect all Pydantic models referenced in a type annotation."""
    if collected is None:
        collected = set()

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        if annotation not in collected:
            collected.add(annotation)
            # Recurse into model fields
            for field in annotation.model_fields.values():
                _collect_models(field.annotation, collected)
        return collected

    origin = get_origin(annotation)
    args = get_args(annotation)

    if args:
        for arg in args:
            _collect_models(arg, collected)

    return collected


def pydantic_to_typescript(models: list[type[BaseModel]] | type[BaseModel]) -> str:
    """Convert Pydantic models to TypeScript interfaces.

    Args:
        models: A single model or list of models to convert.
                Recursively includes all referenced models.

    Returns:
        TypeScript interface definitions as a string.
    """
    if not isinstance(models, list):
        models = [models]

    # Collect all referenced models
    all_models: set[type] = set()
    for model in models:
        _collect_models(model, all_models)

    # Sort for deterministic output (dependencies first would be ideal, but alphabetical is fine)
    sorted_models = sorted(all_models, key=lambda m: m.__name__)

    interfaces = []
    for model in sorted_models:
        lines = [f"interface {model.__name__} {{"]

        for name, field in model.model_fields.items():
            # JSDoc comment for description
            if field.description:
                lines.append(f"  /** {field.description} */")

            # Check if optional (not required by Pydantic)
            is_optional = not field.is_required()

            ts_type = _ts_type(field.annotation)
            # Remove null from type if we're marking as optional with ?
            if is_optional and " | null" in ts_type:
                ts_type = ts_type.replace(" | null", "")

            optional_marker = "?" if is_optional else ""
            lines.append(f"  {name}{optional_marker}: {ts_type};")

        lines.append("}")
        interfaces.append("\n".join(lines))

    return "\n\n".join(interfaces)


def value_to_typescript(value: Any, indent: int = 0) -> str:
    """Convert a Python value to TypeScript literal syntax.

    Handles Pydantic models, dicts, lists, and primitives.
    """
    prefix = "  " * indent

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        # Escape quotes and use double quotes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'

    if isinstance(value, BaseModel):
        return value_to_typescript(value.model_dump(), indent)

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        items = list(value.items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ","  # trailing comma
            val_str = value_to_typescript(v, indent + 1)
            # Handle multi-line values
            if "\n" in val_str:
                lines.append(f"{prefix}  {k}: {val_str}{comma}")
            else:
                lines.append(f"{prefix}  {k}: {val_str}{comma}")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        # Check if simple list (all primitives on one line)
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
            items = [value_to_typescript(v, 0) for v in value]
            return f"[{', '.join(items)}]"
        # Complex list - multi-line
        lines = ["["]
        for i, v in enumerate(value):
            comma = "," if i < len(value) - 1 else ","
            val_str = value_to_typescript(v, indent + 1)
            lines.append(f"{prefix}  {val_str}{comma}")
        lines.append(f"{prefix}]")
        return "\n".join(lines)

    # Fallback
    return str(value)


# --- Schema Rendering (from BAML) - kept for backwards compat ---

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

    def format_turn2_typescript(self, signature) -> str:
        """Format Turn 2 using TypeScript interfaces.

        This is the preferred format:
        - Uses real TypeScript syntax (LLMs know it well)
        - JSDoc comments for field descriptions
        - Optional fields marked with ?
        - Includes static examples from signature if defined

        Expected output from LLM: TypeScript object literal (parseable with json5)
        """
        parts = []
        parts.append("Respond with a TypeScript value matching this type:")
        parts.append("")
        parts.append("```typescript")

        # Collect all Pydantic models from output fields
        models_to_render = []
        for field in signature.output_fields.values():
            if inspect.isclass(field.annotation) and issubclass(field.annotation, BaseModel):
                models_to_render.append(field.annotation)
            else:
                # Check for models inside generics (list[Model], etc.)
                _collect_models(field.annotation, set())  # warm up
                for model in _collect_models(field.annotation):
                    if model not in models_to_render:
                        models_to_render.append(model)

        # Render TypeScript interfaces
        if models_to_render:
            parts.append(pydantic_to_typescript(models_to_render))
            parts.append("")

        # Build the Response type from output fields
        parts.append("type Response = {")
        for name, field in signature.output_fields.items():
            if field.description:
                parts.append(f"  /** {field.description} */")
            ts_type = _ts_type(field.annotation)
            optional_marker = "?" if _is_optional_type(field.annotation) else ""
            parts.append(f"  {name}{optional_marker}: {ts_type};")
        parts.append("};")
        parts.append("```")

        # Add static examples if defined on signature
        examples = getattr(signature, 'Examples', None)
        if examples:
            output_examples = getattr(examples, 'outputs', None)
            if output_examples:
                parts.append("")
                if len(output_examples) == 1:
                    parts.append("Example output:")
                    parts.append("```typescript")
                    parts.append(value_to_typescript(output_examples[0]))
                    parts.append("```")
                else:
                    parts.append("Example outputs:")
                    parts.append("```typescript")
                    for i, ex in enumerate(output_examples):
                        parts.append(f"// Example {i + 1}:")
                        parts.append(value_to_typescript(ex))
                        if i < len(output_examples) - 1:
                            parts.append("")
                    parts.append("```")

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
