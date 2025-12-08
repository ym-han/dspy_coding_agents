"""Unit tests for CodexAgent utilities.

Tests helper functions used by CodexAgent.
"""

from typing import Optional

import pytest

from codex_dspy.agent import (
    _build_output_schema,
    _combine_usage,
    _ensure_additional_properties_false,
    _is_all_str_outputs,
    _parse_output_value,
    _strip_json_fences,
)


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

    def test_defs_hoisted_to_root(self):
        """$defs from nested models should be hoisted to root level."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            inner: Inner

        sig = MockSignature({"result": Outer})
        schema = _build_output_schema(sig)

        # $defs should be at root, not buried in properties
        assert "$defs" in schema
        assert "Inner" in schema["$defs"]
        # Should not be in the property schema
        assert "$defs" not in schema["properties"]["result"]

    def test_defs_hoisted_from_list_of_models(self):
        """$defs from list[Model] should be hoisted to root."""
        from pydantic import BaseModel

        class BugReport(BaseModel):
            severity: str
            description: str

        sig = MockSignature({"bugs": list[BugReport]})
        schema = _build_output_schema(sig)

        # $defs should be at root level
        assert "$defs" in schema
        assert "BugReport" in schema["$defs"]


class TestParseOutputValue:
    """TDD tests for _parse_output_value function.

    These tests define the DESIRED behavior for output parsing.
    """

    # --- Optional[Model] (Model | None) ---

    def test_optional_model_with_dict_validates(self):
        """Model | None with a dict value should validate to Model instance."""
        from pydantic import BaseModel

        class Config(BaseModel):
            setting: str

        result = _parse_output_value({"setting": "enabled"}, Config | None)

        assert isinstance(result, Config)
        assert result.setting == "enabled"

    def test_optional_model_with_none_passes_through(self):
        """Model | None with None value should return None."""
        from pydantic import BaseModel

        class Config(BaseModel):
            setting: str

        result = _parse_output_value(None, Config | None)

        assert result is None

    def test_optional_model_typing_union(self):
        """Optional[Model] using typing.Union should also work."""
        from typing import Union
        from pydantic import BaseModel

        class Config(BaseModel):
            setting: str

        result = _parse_output_value({"setting": "test"}, Union[Config, None])

        assert isinstance(result, Config)
        assert result.setting == "test"

    # --- list[list[str]] nested generics ---

    def test_nested_list_passes_through(self):
        """list[list[str]] should pass through unchanged."""
        result = _parse_output_value([["a", "b"], ["c", "d"]], list[list[str]])

        assert result == [["a", "b"], ["c", "d"]]

    def test_nested_list_empty(self):
        """Empty nested list should pass through."""
        result = _parse_output_value([], list[list[str]])

        assert result == []

    # --- dict[str, T] ---

    def test_dict_passes_through(self):
        """dict[str, int] should pass through unchanged."""
        result = _parse_output_value({"a": 1, "b": 2}, dict[str, int])

        assert result == {"a": 1, "b": 2}

    def test_dict_empty(self):
        """Empty dict should pass through."""
        result = _parse_output_value({}, dict[str, int])

        assert result == {}

    # --- Empty list and None handling ---

    def test_empty_list_of_models(self):
        """Empty list[Model] should return empty list."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        result = _parse_output_value([], list[Item])

        assert result == []

    def test_none_value_any_type(self):
        """None value should always return None regardless of annotation."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        assert _parse_output_value(None, str) is None
        assert _parse_output_value(None, list[str]) is None
        assert _parse_output_value(None, Item) is None
        assert _parse_output_value(None, list[Item]) is None

    # --- list[Model | None] - the tricky one ---

    def test_list_of_optional_models_validates_dicts(self):
        """list[Model | None] should validate dicts to Models, keep Nones."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        value = [{"name": "first"}, None, {"name": "third"}]
        result = _parse_output_value(value, list[Item | None])

        assert len(result) == 3
        assert isinstance(result[0], Item)
        assert result[0].name == "first"
        assert result[1] is None
        assert isinstance(result[2], Item)
        assert result[2].name == "third"

    def test_list_of_optional_models_all_none(self):
        """list[Model | None] with all Nones should preserve them."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        result = _parse_output_value([None, None], list[Item | None])

        assert result == [None, None]

    def test_list_of_optional_models_typing_union(self):
        """list[Union[Model, None]] should also work."""
        from typing import Union
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        value = [{"name": "test"}, None]
        result = _parse_output_value(value, list[Union[Item, None]])

        assert isinstance(result[0], Item)
        assert result[1] is None

    # --- Optional list of models: list[Model] | None ---

    def test_optional_list_of_models_validates_when_present(self):
        """list[Model] | None should validate list elements when value is not None."""
        from pydantic import BaseModel

        class BugReport(BaseModel):
            severity: str
            description: str

        value = [{"severity": "high", "description": "SQL injection"}]
        result = _parse_output_value(value, list[BugReport] | None)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], BugReport)
        assert result[0].severity == "high"

    def test_optional_list_of_models_none_passes_through(self):
        """list[Model] | None should return None when value is None."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        result = _parse_output_value(None, list[Item] | None)
        assert result is None

    def test_optional_list_of_models_typing_union(self):
        """Union[list[Model], None] should also validate list elements."""
        from typing import Union
        from pydantic import BaseModel

        class Task(BaseModel):
            title: str

        value = [{"title": "Task 1"}, {"title": "Task 2"}]
        result = _parse_output_value(value, Union[list[Task], None])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, Task) for r in result)

    # --- Existing behavior (regression tests) ---

    def test_direct_model_validates(self):
        """Direct Pydantic model should validate dict."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        result = _parse_output_value({"name": "Alice", "age": 30}, Person)

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_list_of_models_validates(self):
        """list[Model] should validate each dict to Model."""
        from pydantic import BaseModel

        class Item(BaseModel):
            value: int

        value = [{"value": 1}, {"value": 2}, {"value": 3}]
        result = _parse_output_value(value, list[Item])

        assert len(result) == 3
        assert all(isinstance(r, Item) for r in result)
        assert [r.value for r in result] == [1, 2, 3]

    def test_primitive_passes_through(self):
        """Primitive types should pass through unchanged."""
        assert _parse_output_value("hello", str) == "hello"
        assert _parse_output_value(42, int) == 42
        assert _parse_output_value(3.14, float) == 3.14
        assert _parse_output_value(True, bool) is True

    def test_list_of_primitives_passes_through(self):
        """list[str] should pass through unchanged."""
        result = _parse_output_value(["a", "b", "c"], list[str])

        assert result == ["a", "b", "c"]


class TestEnsureAdditionalPropertiesFalse:
    """Tests for _ensure_additional_properties_false function.

    This is critical - the OpenAI API rejects schemas without
    additionalProperties: false on ALL object types.
    """

    def test_adds_to_root_object(self):
        """Root object should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        _ensure_additional_properties_false(schema)

        assert schema["additionalProperties"] is False

    def test_adds_to_nested_properties(self):
        """Nested objects in properties should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        _ensure_additional_properties_false(schema)

        assert schema["additionalProperties"] is False
        assert schema["properties"]["user"]["additionalProperties"] is False

    def test_adds_to_defs(self):
        """Objects in $defs should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {"result": {"$ref": "#/$defs/MyModel"}},
            "$defs": {
                "MyModel": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                }
            },
        }
        _ensure_additional_properties_false(schema)

        assert schema["$defs"]["MyModel"]["additionalProperties"] is False

    def test_adds_to_array_items(self):
        """Objects in array items should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            },
        }
        _ensure_additional_properties_false(schema)

        assert schema["properties"]["items"]["items"]["additionalProperties"] is False

    def test_adds_to_anyof(self):
        """Objects in anyOf should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "anyOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                        {"type": "null"},
                    ]
                }
            },
        }
        _ensure_additional_properties_false(schema)

        assert schema["properties"]["result"]["anyOf"][0]["additionalProperties"] is False

    def test_adds_to_oneof(self):
        """Objects in oneOf should get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "variant": {
                    "oneOf": [
                        {"type": "object", "properties": {"x": {"type": "integer"}}},
                        {"type": "object", "properties": {"y": {"type": "integer"}}},
                    ]
                }
            },
        }
        _ensure_additional_properties_false(schema)

        assert schema["properties"]["variant"]["oneOf"][0]["additionalProperties"] is False
        assert schema["properties"]["variant"]["oneOf"][1]["additionalProperties"] is False

    def test_deeply_nested(self):
        """Deeply nested objects should all get additionalProperties: false."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {"value": {"type": "string"}},
                                }
                            },
                        }
                    },
                }
            },
        }
        _ensure_additional_properties_false(schema)

        assert schema["additionalProperties"] is False
        level1 = schema["properties"]["level1"]
        assert level1["additionalProperties"] is False
        level2 = level1["properties"]["level2"]
        assert level2["additionalProperties"] is False
        level3 = level2["properties"]["level3"]
        assert level3["additionalProperties"] is False

    def test_preserves_existing_false(self):
        """Existing additionalProperties: false should be preserved."""
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        _ensure_additional_properties_false(schema)

        assert schema["additionalProperties"] is False

    def test_overwrites_existing_true(self):
        """Existing additionalProperties: true should be overwritten to false."""
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
        _ensure_additional_properties_false(schema)

        assert schema["additionalProperties"] is False

    def test_non_object_unchanged(self):
        """Non-object types should not get additionalProperties."""
        schema = {"type": "string"}
        _ensure_additional_properties_false(schema)

        assert "additionalProperties" not in schema

    def test_array_type_unchanged(self):
        """Array type itself should not get additionalProperties."""
        schema = {"type": "array", "items": {"type": "string"}}
        _ensure_additional_properties_false(schema)

        assert "additionalProperties" not in schema

    def test_handles_non_dict(self):
        """Non-dict input should not raise."""
        _ensure_additional_properties_false("not a dict")
        _ensure_additional_properties_false(None)
        _ensure_additional_properties_false(42)

    def test_real_pydantic_schema(self):
        """Test with actual Pydantic-generated schema structure."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            inner: Inner

        schema = Outer.model_json_schema()
        _ensure_additional_properties_false(schema)

        # Root should have it
        assert schema["additionalProperties"] is False
        # $defs.Inner should have it
        assert schema["$defs"]["Inner"]["additionalProperties"] is False
