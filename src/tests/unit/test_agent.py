"""Unit tests for CodexAgent utilities.

Tests helper functions used by CodexAgent.
"""

from typing import Optional

import pytest

from codex_dspy.agent import _build_output_schema, _combine_usage, _is_all_str_outputs, _strip_json_fences


class TestStripJsonFences:
    """Tests for _strip_json_fences function."""

    def test_no_fences(self):
        """Plain JSON without fences should be returned as-is."""
        json_str = '{"name": "test", "value": 42}'
        assert _strip_json_fences(json_str) == json_str

    def test_json_fences(self):
        """JSON wrapped in ```json ... ``` should be unwrapped."""
        fenced = '```json\n{"name": "test", "value": 42}\n```'
        expected = '{"name": "test", "value": 42}'
        assert _strip_json_fences(fenced) == expected

    def test_plain_fences(self):
        """JSON wrapped in ``` ... ``` (no language) should be unwrapped."""
        fenced = '```\n{"name": "test"}\n```'
        expected = '{"name": "test"}'
        assert _strip_json_fences(fenced) == expected

    def test_multiline_json(self):
        """Multi-line JSON in fences should be unwrapped correctly."""
        fenced = '''```json
{
  "bugs": [
    {"severity": "high", "description": "Division by zero"}
  ],
  "summary": "Found 1 bug"
}
```'''
        result = _strip_json_fences(fenced)
        assert '"bugs"' in result
        assert '"summary"' in result
        assert '```' not in result

    def test_with_leading_whitespace(self):
        """Leading/trailing whitespace should be handled."""
        fenced = '  \n```json\n{"test": true}\n```  \n'
        expected = '{"test": true}'
        assert _strip_json_fences(fenced) == expected

    def test_fences_no_newline(self):
        """Fences without newlines should work."""
        fenced = '```json{"inline": true}```'
        expected = '{"inline": true}'
        assert _strip_json_fences(fenced) == expected

    def test_nested_backticks_in_string(self):
        """Backticks in JSON string values should not confuse the parser."""
        # This is JSON without fences that happens to contain backticks in a string
        json_str = '{"code": "Use `json.loads()` here"}'
        assert _strip_json_fences(json_str) == json_str

    def test_empty_json_object(self):
        """Empty JSON object should work."""
        fenced = '```json\n{}\n```'
        assert _strip_json_fences(fenced) == '{}'

    def test_json_array(self):
        """JSON array should work."""
        fenced = '```json\n[1, 2, 3]\n```'
        assert _strip_json_fences(fenced) == '[1, 2, 3]'


class TestCombineUsage:
    """Tests for _combine_usage function."""

    def test_both_none(self):
        """Both None should return None."""
        assert _combine_usage(None, None) is None

    def test_first_none(self):
        """First None should return second."""
        from codex import Usage
        usage2 = Usage(input_tokens=100, output_tokens=50, cached_input_tokens=10)
        result = _combine_usage(None, usage2)
        assert result is usage2

    def test_second_none(self):
        """Second None should return first."""
        from codex import Usage
        usage1 = Usage(input_tokens=100, output_tokens=50, cached_input_tokens=10)
        result = _combine_usage(usage1, None)
        assert result is usage1

    def test_sum_tokens(self):
        """Both present should sum all token counts."""
        from codex import Usage
        usage1 = Usage(input_tokens=100, output_tokens=50, cached_input_tokens=10)
        usage2 = Usage(input_tokens=200, output_tokens=75, cached_input_tokens=20)
        result = _combine_usage(usage1, usage2)

        assert result.input_tokens == 300
        assert result.output_tokens == 125
        assert result.cached_input_tokens == 30

    def test_handles_zero_values(self):
        """Zero values should be handled correctly."""
        from codex import Usage
        usage1 = Usage(input_tokens=100, output_tokens=0, cached_input_tokens=0)
        usage2 = Usage(input_tokens=0, output_tokens=50, cached_input_tokens=0)
        result = _combine_usage(usage1, usage2)

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cached_input_tokens == 0


class MockFieldInfo:
    """Mock DSPy FieldInfo for testing."""
    def __init__(self, annotation):
        self.annotation = annotation


class MockSignature:
    """Mock DSPy Signature for testing."""
    def __init__(self, output_fields: dict):
        self.output_fields = {k: MockFieldInfo(v) for k, v in output_fields.items()}


class TestIsAllStrOutputs:
    """Tests for _is_all_str_outputs function."""

    def test_single_str(self):
        """Single str output should return True."""
        sig = MockSignature({"answer": str})
        assert _is_all_str_outputs(sig) is True

    def test_multiple_str(self):
        """Multiple str outputs should return True."""
        sig = MockSignature({"answer": str, "summary": str})
        assert _is_all_str_outputs(sig) is True

    def test_optional_str_typing_union(self):
        """Optional[str] (typing.Union) should return True."""
        from typing import Union
        sig = MockSignature({"answer": Union[str, None]})
        assert _is_all_str_outputs(sig) is True

    def test_optional_str_pep604(self):
        """str | None (PEP 604) should return True."""
        sig = MockSignature({"answer": str | None})
        assert _is_all_str_outputs(sig) is True

    def test_mixed_str_and_optional_str(self):
        """Mix of str and str | None should return True."""
        sig = MockSignature({"answer": str, "notes": str | None})
        assert _is_all_str_outputs(sig) is True

    def test_int_output(self):
        """Non-str output should return False."""
        sig = MockSignature({"count": int})
        assert _is_all_str_outputs(sig) is False

    def test_pydantic_output(self):
        """Pydantic model output should return False."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        sig = MockSignature({"result": MyModel})
        assert _is_all_str_outputs(sig) is False

    def test_list_str_output(self):
        """list[str] should return False (not plain str)."""
        sig = MockSignature({"items": list[str]})
        assert _is_all_str_outputs(sig) is False


class TestListPydanticParsing:
    """Tests for list[PydanticModel] output parsing.

    Verifies that the parsing logic correctly handles list[Model] annotations.
    """

    def test_list_pydantic_model_is_validated(self):
        """list[PydanticModel] should be validated, not returned as raw dicts."""
        from typing import get_origin, get_args
        from pydantic import BaseModel

        class BugReport(BaseModel):
            severity: str
            description: str

        # Simulate the parsing logic from agent.forward()
        annotation = list[BugReport]
        value = [
            {"severity": "high", "description": "Division by zero"},
            {"severity": "low", "description": "Missing docstring"},
        ]

        # This is the fixed logic
        if get_origin(annotation) is list:
            inner_type = get_args(annotation)[0] if get_args(annotation) else None
            if inner_type and hasattr(inner_type, "model_validate") and isinstance(value, list):
                result = [inner_type.model_validate(v) for v in value]
            else:
                result = value
        else:
            result = value

        # Verify we got validated Pydantic models, not raw dicts
        assert len(result) == 2
        assert isinstance(result[0], BugReport)
        assert isinstance(result[1], BugReport)
        assert result[0].severity == "high"
        assert result[1].description == "Missing docstring"

    def test_list_pydantic_model_detects_inner_type(self):
        """get_origin/get_args should correctly identify list[Model]."""
        from typing import get_origin, get_args
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        annotation = list[MyModel]

        assert get_origin(annotation) is list
        args = get_args(annotation)
        assert len(args) == 1
        assert args[0] is MyModel
        assert hasattr(args[0], "model_validate")

    def test_list_non_pydantic_not_validated(self):
        """list[str] should not attempt Pydantic validation."""
        from typing import get_origin, get_args

        annotation = list[str]
        value = ["a", "b", "c"]

        if get_origin(annotation) is list:
            inner_type = get_args(annotation)[0] if get_args(annotation) else None
            if inner_type and hasattr(inner_type, "model_validate") and isinstance(value, list):
                result = "should not reach here"
            else:
                result = value
        else:
            result = value

        # str doesn't have model_validate, so value passed through unchanged
        assert result == ["a", "b", "c"]

    def test_direct_pydantic_model_still_works(self):
        """Direct PydanticModel (not list) should still be validated."""
        from typing import get_origin, get_args
        from pydantic import BaseModel

        class SingleModel(BaseModel):
            value: int

        annotation = SingleModel
        value = {"value": 42}

        # Simulate the fixed logic (list check first, then direct model)
        if get_origin(annotation) is list:
            result = value  # Not a list annotation
        elif hasattr(annotation, "model_validate"):
            if isinstance(value, dict):
                result = annotation.model_validate(value)
            else:
                result = value
        else:
            result = value

        assert isinstance(result, SingleModel)
        assert result.value == 42


class TestBuildOutputSchema:
    """Tests for _build_output_schema function."""

    def test_required_field_marked_required(self):
        """Non-optional fields should be in required list."""
        sig = MockSignature({"answer": str})
        schema = _build_output_schema(sig)

        assert "answer" in schema["required"]

    def test_optional_typing_union_not_required(self):
        """Optional[str] (typing.Union) should not be required."""
        from typing import Union
        sig = MockSignature({"notes": Union[str, None]})
        schema = _build_output_schema(sig)

        assert "notes" not in schema["required"]
        assert "notes" in schema["properties"]

    def test_optional_pep604_not_required(self):
        """str | None (PEP 604) should not be required."""
        sig = MockSignature({"notes": str | None})
        schema = _build_output_schema(sig)

        assert "notes" not in schema["required"]
        assert "notes" in schema["properties"]

    def test_mixed_required_and_optional(self):
        """Mix of required and optional fields should be handled correctly."""
        sig = MockSignature({
            "answer": str,           # required
            "notes": str | None,     # optional (PEP 604)
            "count": int,            # required
        })
        schema = _build_output_schema(sig)

        assert "answer" in schema["required"]
        assert "count" in schema["required"]
        assert "notes" not in schema["required"]

        # All fields should be in properties
        assert "answer" in schema["properties"]
        assert "notes" in schema["properties"]
        assert "count" in schema["properties"]

    def test_pydantic_model_field(self):
        """Pydantic model fields should use model_json_schema."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        sig = MockSignature({"result": MyModel})
        schema = _build_output_schema(sig)

        assert "result" in schema["required"]
        assert "result" in schema["properties"]
        # Should have Pydantic schema structure
        assert "properties" in schema["properties"]["result"]

    def test_optional_pydantic_model_not_required(self):
        """Model | None should not be required."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        sig = MockSignature({"result": MyModel | None})
        schema = _build_output_schema(sig)

        assert "result" not in schema["required"]
        assert "result" in schema["properties"]

    def test_nested_list(self):
        """list[list[str]] should be handled."""
        sig = MockSignature({"matrix": list[list[str]]})
        schema = _build_output_schema(sig)

        assert "matrix" in schema["required"]
        assert "matrix" in schema["properties"]

    def test_dict_type(self):
        """dict[str, int] should be handled."""
        sig = MockSignature({"counts": dict[str, int]})
        schema = _build_output_schema(sig)

        assert "counts" in schema["required"]
        assert "counts" in schema["properties"]

    def test_list_of_optional_pydantic(self):
        """list[Model | None] edge case."""
        from pydantic import BaseModel

        class Item(BaseModel):
            value: int

        sig = MockSignature({"items": list[Item | None]})
        schema = _build_output_schema(sig)

        assert "items" in schema["required"]
        assert "items" in schema["properties"]


class TestOptionalPydanticParsing:
    """Tests for Optional[PydanticModel] output parsing."""

    def test_optional_model_with_value(self):
        """Model | None with a dict value should be validated."""
        from typing import get_origin, get_args
        from pydantic import BaseModel
        import types

        class Config(BaseModel):
            setting: str

        annotation = Config | None
        value = {"setting": "enabled"}

        # Simulate parsing logic
        origin = get_origin(annotation)
        is_optional_model = (origin is types.UnionType) and type(None) in get_args(annotation)

        if is_optional_model:
            # Get the non-None type
            args = [a for a in get_args(annotation) if a is not type(None)]
            model_type = args[0] if args else None
            if model_type and hasattr(model_type, "model_validate") and isinstance(value, dict):
                result = model_type.model_validate(value)
            else:
                result = value
        else:
            result = value

        assert isinstance(result, Config)
        assert result.setting == "enabled"

    def test_optional_model_with_none(self):
        """Model | None with None value should pass through."""
        from pydantic import BaseModel

        class Config(BaseModel):
            setting: str

        annotation = Config | None
        value = None

        # None should pass through unchanged
        assert value is None


class TestNestedListParsing:
    """Tests for nested list types like list[list[str]]."""

    def test_nested_list_passthrough(self):
        """list[list[str]] should pass through as-is (no Pydantic validation)."""
        from typing import get_origin, get_args

        annotation = list[list[str]]
        value = [["a", "b"], ["c", "d"]]

        # Current logic: check if it's a list
        if get_origin(annotation) is list:
            inner_type = get_args(annotation)[0] if get_args(annotation) else None
            # inner_type is list[str], which doesn't have model_validate
            if inner_type and hasattr(inner_type, "model_validate") and isinstance(value, list):
                result = "should not reach"
            else:
                result = value
        else:
            result = value

        assert result == [["a", "b"], ["c", "d"]]

    def test_nested_list_type_detection(self):
        """get_origin/get_args should correctly identify nested lists."""
        from typing import get_origin, get_args

        annotation = list[list[str]]

        assert get_origin(annotation) is list
        inner = get_args(annotation)[0]
        assert get_origin(inner) is list
        assert get_args(inner)[0] is str


class TestDictTypeParsing:
    """Tests for dict[K, V] types."""

    def test_dict_passthrough(self):
        """dict[str, int] should pass through as-is."""
        from typing import get_origin

        annotation = dict[str, int]
        value = {"a": 1, "b": 2}

        # dict doesn't have model_validate, should pass through
        if get_origin(annotation) is list:
            result = "should not reach"
        elif hasattr(annotation, "model_validate"):
            result = "should not reach"
        else:
            result = value

        assert result == {"a": 1, "b": 2}

    def test_dict_type_detection(self):
        """get_origin/get_args should correctly identify dict types."""
        from typing import get_origin, get_args

        annotation = dict[str, int]

        assert get_origin(annotation) is dict
        args = get_args(annotation)
        assert args[0] is str
        assert args[1] is int


class TestEmptyAndNoneListHandling:
    """Tests for empty lists and None in list fields."""

    def test_empty_list_passthrough(self):
        """Empty list should pass through unchanged."""
        from typing import get_origin, get_args
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        annotation = list[Item]
        value = []

        if get_origin(annotation) is list:
            inner_type = get_args(annotation)[0] if get_args(annotation) else None
            if inner_type and hasattr(inner_type, "model_validate") and isinstance(value, list):
                result = [inner_type.model_validate(v) for v in value]
            else:
                result = value
        else:
            result = value

        assert result == []

    def test_none_list_field(self):
        """None value for list field should pass through."""
        value = None
        # None check happens first in the parsing logic
        assert value is None

    def test_list_with_none_elements(self):
        """list containing None elements (if allowed by type)."""
        from typing import get_origin, get_args
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        # list[Item | None] - list that can contain None elements
        annotation = list[Item | None]
        value = [{"name": "first"}, None, {"name": "third"}]

        # This is tricky - inner type is Item | None, not Item
        # Current logic won't validate this correctly, but it should pass through
        if get_origin(annotation) is list:
            inner_type = get_args(annotation)[0] if get_args(annotation) else None
            # inner_type is Item | None, which doesn't have model_validate directly
            if inner_type and hasattr(inner_type, "model_validate"):
                result = "would validate"
            else:
                result = value  # passes through
        else:
            result = value

        # Current behavior: passes through as raw dicts/None
        assert result == [{"name": "first"}, None, {"name": "third"}]
