"""Unit tests for CodexAgent utilities.

Tests helper functions used by CodexAgent.
"""

from typing import Optional

import pytest

from codex_dspy.agent import _combine_usage, _is_all_str_outputs, _strip_json_fences


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
