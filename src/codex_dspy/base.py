"""Base agent class for multi-agent DSPy integration.

This module provides shared infrastructure for wrapping agent SDKs in DSPy modules.
"""

from typing import Any, Union, get_args, get_origin

import dspy
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature


def _is_str_type(annotation: Any) -> bool:
    """Check if annotation is str or Optional[str].

    Args:
        annotation: Type annotation to check

    Returns:
        True if annotation is str, Optional[str], or Union[str, None]
    """
    if annotation is str:
        return True

    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # Check for Optional[str] which is Union[str, None]
        if len(args) == 2 and str in args and type(None) in args:
            return True

    return False


def _is_pydantic_model(cls: Any) -> bool:
    """Check if a class is a Pydantic BaseModel.

    Args:
        cls: Class to check

    Returns:
        True if cls is a Pydantic BaseModel subclass
    """
    try:
        from pydantic import BaseModel

        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except (TypeError, ImportError):
        return False


def _is_pydantic_type(annotation: Any) -> bool:
    """Check if type annotation is a Pydantic model or Optional[PydanticModel].

    Args:
        annotation: Type annotation to check

    Returns:
        True if annotation is a Pydantic BaseModel or Optional[BaseModel]
    """
    # Check if it's directly a Pydantic model
    if _is_pydantic_model(annotation):
        return True

    # Check for Optional[PydanticModel]
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        return any(_is_pydantic_model(arg) for arg in args if arg is not type(None))

    return False


class BaseAgent(dspy.Module):
    """Base class for agent DSPy modules.

    Provides common signature validation and field extraction logic.
    Concrete agent implementations (Codex, Claude, etc.) inherit from this class.
    """

    def __init__(self, signature: str | type[Signature]):
        """Initialize base agent with signature validation.

        Args:
            signature: DSPy signature (must have at least 1 input and exactly 1 output field)

        Raises:
            ValueError: If signature doesn't have at least 1 input and exactly 1 output field
        """
        super().__init__()

        # Ensure signature is valid
        self.signature = ensure_signature(signature)

        # Validate: at least 1 input field, exactly 1 output field
        if len(self.signature.input_fields) < 1:
            raise ValueError(
                f"{self.__class__.__name__} requires at least 1 input field.\n"
                f"Example: dspy.Signature('message:str -> answer:str')"
            )

        if len(self.signature.output_fields) != 1:
            output_fields = list(self.signature.output_fields.keys())
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 output field, "
                f"got {len(output_fields)}: {output_fields}\n"
                f"Example: dspy.Signature('message:str -> answer:str')"
            )

        # Extract output field name and type
        self.output_field = next(iter(self.signature.output_fields.keys()))
        self.output_field_info = self.signature.output_fields[self.output_field]
        self.output_type = self.output_field_info.annotation

    def _format_input_message(self, kwargs: dict[str, Any]) -> str:
        """Format all input fields into a single prompt string.

        Single-field signatures: returns the value directly.
        Multi-field signatures: renders each field as a labelled block.

        Args:
            kwargs: Keyword arguments from forward() call

        Returns:
            Formatted input string ready to send to the agent
        """
        input_fields = self.signature.input_fields
        if len(input_fields) == 1:
            field_name = next(iter(input_fields))
            return str(kwargs[field_name])

        parts = []
        for name, field_info in input_fields.items():
            desc = (field_info.json_schema_extra or {}).get("desc", "")
            label = f"{name} ({desc})" if desc and desc != f"${{{name}}}" else name
            parts.append(f"[{label}]\n{kwargs[name]}")
        return "\n\n".join(parts)

    def _append_output_description(self, message: str) -> str:
        """Append output field description to message if present.

        Skips DSPy's default placeholder (e.g., "${answer}").

        Args:
            message: Original input message

        Returns:
            Message with optional output description appended
        """
        output_desc = (self.output_field_info.json_schema_extra or {}).get("desc")
        # Skip if desc is just DSPy's default placeholder
        if output_desc and output_desc != f"${{{self.output_field}}}":
            return f"{message}\n\nPlease produce the following output: {output_desc}"
        return message

    def forward(self, **kwargs) -> Prediction:
        """Execute agent with input message.

        Concrete implementations must:
        1. Format inputs via _format_input_message()
        2. Optionally append output description via _append_output_description()
        3. Call underlying agent SDK
        4. Parse response according to output type
        5. Return Prediction with typed output + metadata

        Args:
            **kwargs: Must contain the input field specified in signature

        Returns:
            Prediction with typed output field and metadata

        Raises:
            ValueError: If parsing fails for typed output
        """
        raise NotImplementedError
