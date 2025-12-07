"""Unit tests for CodexAdapter.

Tests formatting of Turn 1 (task) and Turn 2 (extraction) prompts,
as well as parsing of structured responses.
"""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from codex_dspy.adapter import (
    CodexAdapter,
    _build_simplified_schema,
    _render_type_str,
    format_field_description,
    get_annotation_name,
)


# --- Test Fixtures: Pydantic Models ---


class SimpleModel(BaseModel):
    name: str
    age: int


class ModelWithDescriptions(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    active: bool = Field(description="Whether the account is active")


class NestedAddress(BaseModel):
    street: str
    city: str
    country: Literal["US", "CA", "UK"]


class PersonWithAddress(BaseModel):
    name: str = Field(description="Person's name")
    address: NestedAddress = Field(description="Home address")


class BugReport(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(description="Bug severity level")
    location: str = Field(description="File and line number")
    description: str = Field(description="What the bug does")
    suggested_fix: str | None = Field(default=None, description="How to fix it")


class AnalysisResult(BaseModel):
    bugs: list[BugReport] = Field(description="List of bugs found")
    summary: str = Field(description="Overall summary")
    confidence: float = Field(description="Confidence score 0-1")


# --- Mock Signature for Testing ---


class MockFieldInfo:
    """Mock DSPy FieldInfo for testing."""

    def __init__(self, annotation, description=None):
        self.annotation = annotation
        self.description = description


class MockSignature:
    """Mock DSPy Signature for testing."""

    def __init__(self, input_fields: dict, output_fields: dict, instructions: str = ""):
        self.input_fields = {k: MockFieldInfo(v[0], v[1]) for k, v in input_fields.items()}
        self.output_fields = {k: MockFieldInfo(v[0], v[1]) for k, v in output_fields.items()}
        self.instructions = instructions


# --- Tests for Type Rendering ---


class TestRenderTypeStr:
    """Tests for _render_type_str function."""

    def test_primitive_str(self):
        assert _render_type_str(str) == "string"

    def test_primitive_int(self):
        assert _render_type_str(int) == "int"

    def test_primitive_float(self):
        assert _render_type_str(float) == "float"

    def test_primitive_bool(self):
        assert _render_type_str(bool) == "boolean"

    def test_literal_strings(self):
        result = _render_type_str(Literal["low", "medium", "high"])
        assert '"low"' in result
        assert '"medium"' in result
        assert '"high"' in result
        assert " or " in result

    def test_optional_str(self):
        result = _render_type_str(str | None)
        assert "string" in result
        assert "null" in result

    def test_list_of_primitives(self):
        result = _render_type_str(list[str])
        assert result == "string[]"

    def test_list_of_pydantic_model(self):
        result = _render_type_str(list[SimpleModel])
        assert "[" in result
        assert "name: string," in result
        assert "age: int," in result
        assert "]" in result

    def test_nested_list(self):
        """list[list[str]] should render correctly."""
        result = _render_type_str(list[list[str]])
        assert "string[][]" in result

    def test_dict_type(self):
        """dict[str, int] should render as dict."""
        result = _render_type_str(dict[str, int])
        # dict types fall back to type name
        assert "dict" in result.lower()

    def test_optional_pydantic_model(self):
        """Model | None should render with null option."""
        result = _render_type_str(SimpleModel | None)
        assert "null" in result

    def test_list_of_optional(self):
        """list[str | None] should render correctly."""
        result = _render_type_str(list[str | None])
        # Inner type is str | None
        assert "[]" in result


class TestBuildSimplifiedSchema:
    """Tests for _build_simplified_schema function."""

    def test_simple_model(self):
        schema = _build_simplified_schema(SimpleModel)
        assert "{" in schema
        assert "}" in schema
        assert "name: string," in schema
        assert "age: int," in schema

    def test_model_with_descriptions(self):
        schema = _build_simplified_schema(ModelWithDescriptions)
        assert "# Full name of the person" in schema
        assert "# Age in years" in schema
        assert "# Whether the account is active" in schema
        assert "name: string," in schema
        assert "age: int," in schema
        assert "active: boolean," in schema

    def test_nested_model(self):
        schema = _build_simplified_schema(PersonWithAddress)
        assert "# Person's name" in schema
        assert "# Home address" in schema
        assert "address:" in schema
        assert "street: string," in schema
        assert "city: string," in schema
        assert '"US" or "CA" or "UK"' in schema

    def test_model_with_optional_field(self):
        schema = _build_simplified_schema(BugReport)
        assert "suggested_fix:" in schema
        assert "or null" in schema

    def test_model_with_list_field(self):
        schema = _build_simplified_schema(AnalysisResult)
        assert "bugs:" in schema
        assert "[" in schema
        assert "severity:" in schema


# --- Tests for Field Description ---


class TestFormatFieldDescription:
    """Tests for format_field_description function."""

    def test_single_field(self):
        fields = {"name": MockFieldInfo(str, "The user's name")}
        result = format_field_description(fields)
        assert "1. `name` (str): The user's name" in result

    def test_multiple_fields(self):
        fields = {
            "context": MockFieldInfo(str, "Code context"),
            "request": MockFieldInfo(str, "What to look for"),
        }
        result = format_field_description(fields)
        assert "1. `context` (str): Code context" in result
        assert "2. `request` (str): What to look for" in result

    def test_pydantic_model_field(self):
        fields = {"report": MockFieldInfo(BugReport, "The bug report")}
        result = format_field_description(fields)
        assert "`report`" in result
        assert "BugReport" in result
        assert "The bug report" in result

    def test_field_without_description(self):
        fields = {"data": MockFieldInfo(str, None)}
        result = format_field_description(fields)
        assert "1. `data` (str)" in result


class TestGetAnnotationName:
    """Tests for get_annotation_name function."""

    def test_simple_type(self):
        assert get_annotation_name(str) == "str"
        assert get_annotation_name(int) == "int"

    def test_pydantic_model(self):
        assert get_annotation_name(BugReport) == "BugReport"

    def test_literal(self):
        result = get_annotation_name(Literal["a", "b"])
        assert "Literal" in result
        assert '"a"' in result
        assert '"b"' in result

    def test_list(self):
        result = get_annotation_name(list[str])
        assert "list" in result
        assert "str" in result


# --- Tests for CodexAdapter ---


class TestCodexAdapterFormatTurn1:
    """Tests for CodexAdapter.format_turn1 method."""

    def test_simple_signature(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"message": (str, "User message")},
            output_fields={"answer": (str, "Response")},
            instructions="Answer the question",
        )

        result = adapter.format_turn1(sig, {"message": "Hello world"})

        assert "As input, you are provided with:" in result
        assert "`message`" in result
        assert "Your task is to produce:" in result
        assert "`answer`" in result
        assert "Instructions: Answer the question" in result
        assert "message: Hello world" in result

    def test_multiple_input_fields(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={
                "code": (str, "Source code"),
                "context": (str, "Additional context"),
            },
            output_fields={"analysis": (str, "Analysis result")},
        )

        result = adapter.format_turn1(sig, {"code": "def foo(): pass", "context": "Python"})

        assert "code:" in result
        assert "def foo(): pass" in result
        assert "context:" in result
        assert "Python" in result

    def test_pydantic_input(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"person": (PersonWithAddress, "Person info")},
            output_fields={"greeting": (str, "Greeting message")},
        )

        person = PersonWithAddress(
            name="John",
            address=NestedAddress(street="123 Main", city="NYC", country="US"),
        )

        result = adapter.format_turn1(sig, {"person": person})

        # Should be JSON formatted
        assert '"name": "John"' in result
        assert '"street": "123 Main"' in result

    def test_no_instructions(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"y": (str, None)},
            instructions="",
        )

        result = adapter.format_turn1(sig, {"x": "test"})

        assert "Instructions:" not in result


class TestCodexAdapterFormatTurn2:
    """Tests for CodexAdapter.format_turn2 method."""

    def test_simple_output(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"answer": (str, "The answer")},
        )

        result = adapter.format_turn2(sig)

        assert "Now provide your findings" in result
        assert "[[ ## answer ## ]]" in result
        assert "[[ ## completed ## ]]" in result

    def test_pydantic_output(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"report": (BugReport, "Bug report")},
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## report ## ]]" in result
        assert "severity:" in result
        assert "location:" in result
        assert "description:" in result

    def test_multiple_outputs(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={
                "bugs": (list[BugReport], "Bugs found"),
                "summary": (str, "Summary"),
            },
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## bugs ## ]]" in result
        assert "[[ ## summary ## ]]" in result
        assert "[[ ## completed ## ]]" in result

    def test_list_primitive_not_double_bracketed(self):
        """list[str] should render as string[], not string[][]."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"items": (list[str], "List of items")},
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## items ## ]]" in result
        # Should be string[], NOT string[][]
        assert "string[]" in result
        assert "string[][]" not in result

    def test_list_int_not_double_bracketed(self):
        """list[int] should render as int[], not int[][]."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"numbers": (list[int], "List of numbers")},
        )

        result = adapter.format_turn2(sig)

        assert "int[]" in result
        assert "int[][]" not in result


class TestCodexAdapterFormatTurn2Json:
    """Tests for CodexAdapter.format_turn2_json method."""

    def test_json_format(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={
                "answer": (str, "The answer"),
                "confidence": (float, "Confidence score"),
            },
        )

        result = adapter.format_turn2_json(sig)

        assert "JSON" in result
        assert '"answer"' in result
        assert '"confidence"' in result
        assert "```json" in result


class TestCodexAdapterParse:
    """Tests for CodexAdapter.parse method."""

    def test_parse_single_field(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"answer": (str, None)},
        )

        completion = """
[[ ## answer ## ]]
This is the answer to your question.

[[ ## completed ## ]]
"""

        result = adapter.parse(sig, completion)

        assert "answer" in result
        assert result["answer"] == "This is the answer to your question."

    def test_parse_multiple_fields(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={
                "summary": (str, None),
                "details": (str, None),
            },
        )

        completion = """
[[ ## summary ## ]]
Brief summary here.

[[ ## details ## ]]
More detailed explanation
spanning multiple lines.

[[ ## completed ## ]]
"""

        result = adapter.parse(sig, completion)

        assert result["summary"] == "Brief summary here."
        assert "More detailed explanation" in result["details"]
        assert "spanning multiple lines." in result["details"]

    def test_parse_ignores_unknown_fields(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"answer": (str, None)},
        )

        completion = """
[[ ## thinking ## ]]
Some internal reasoning...

[[ ## answer ## ]]
The actual answer.

[[ ## completed ## ]]
"""

        result = adapter.parse(sig, completion)

        assert "answer" in result
        assert "thinking" not in result

    def test_parse_handles_inline_content(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"value": (str, None)},
        )

        completion = "[[ ## value ## ]] inline content\n[[ ## completed ## ]]"

        result = adapter.parse(sig, completion)

        assert result["value"] == "inline content"


class TestEdgeCaseTypes:
    """Tests for edge case type handling across the adapter."""

    def test_format_turn2_optional_model(self):
        """Model | None output should render correctly in turn2."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"config": (SimpleModel | None, "Optional config")},
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## config ## ]]" in result
        # Should show the model schema with null option
        assert "[[ ## completed ## ]]" in result

    def test_format_turn2_nested_list(self):
        """list[list[str]] should render correctly in turn2."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"matrix": (list[list[str]], "2D string matrix")},
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## matrix ## ]]" in result
        assert "string[][]" in result
        # Should NOT be string[][][] (triple brackets)
        assert "string[][][]" not in result

    def test_format_turn2_dict_type(self):
        """dict[str, int] should be handled in turn2."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"counts": (dict[str, int], "Word counts")},
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## counts ## ]]" in result
        assert "[[ ## completed ## ]]" in result

    def test_format_turn2_json_nested_list(self):
        """list[list[str]] should render correctly in turn2_json."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"matrix": (list[list[str]], "2D matrix")},
        )

        result = adapter.format_turn2_json(sig)

        assert '"matrix"' in result
        assert "```json" in result

    def test_format_turn2_json_dict_type(self):
        """dict[str, int] should render correctly in turn2_json."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"counts": (dict[str, int], "Counts")},
        )

        result = adapter.format_turn2_json(sig)

        assert '"counts"' in result
        assert "```json" in result

    def test_format_turn2_list_optional_model(self):
        """list[Model | None] should render with schema and null comment."""
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"items": (list[SimpleModel | None], "Items with gaps")},
        )

        result = adapter.format_turn2(sig)

        assert "[[ ## items ## ]]" in result
        # Should show array with model schema
        assert "[" in result
        assert "name:" in result
        assert "age:" in result
        # Should indicate null is allowed
        assert "null" in result
        # Should NOT be the broken format with }[]
        assert "} or null[]" not in result

    def test_render_type_str_list_optional_model(self):
        """list[Model | None] should render with null comment, not broken brackets."""
        result = _render_type_str(list[SimpleModel | None])

        # Should show the model schema in array format
        assert "[" in result
        assert "name:" in result
        # Should indicate null is allowed
        assert "null" in result
        # Should NOT have the broken }[] format
        assert "} or null[]" not in result


# --- TypeScript Conversion Tests ---

from codex_dspy.adapter import (
    _ts_type,
    _collect_models,
    pydantic_to_typescript,
    value_to_typescript,
)


class TestTsType:
    """Tests for _ts_type function."""

    def test_primitives(self):
        assert _ts_type(str) == "string"
        assert _ts_type(int) == "number"
        assert _ts_type(float) == "number"
        assert _ts_type(bool) == "boolean"

    def test_optional(self):
        result = _ts_type(str | None)
        assert "string" in result
        assert "null" in result
        assert "|" in result

    def test_list(self):
        assert _ts_type(list[str]) == "string[]"
        assert _ts_type(list[int]) == "number[]"

    def test_list_of_optional(self):
        result = _ts_type(list[str | None])
        assert "Array<" in result  # Uses Array<> for union types
        assert "string" in result
        assert "null" in result

    def test_dict(self):
        result = _ts_type(dict[str, int])
        assert "Record<" in result
        assert "string" in result
        assert "number" in result

    def test_literal(self):
        from typing import Literal
        result = _ts_type(Literal["a", "b", "c"])
        assert '"a"' in result
        assert '"b"' in result
        assert '"c"' in result

    def test_pydantic_model(self):
        result = _ts_type(SimpleModel)
        assert result == "SimpleModel"


class TestCollectModels:
    """Tests for _collect_models function."""

    def test_single_model(self):
        result = _collect_models(SimpleModel)
        assert SimpleModel in result

    def test_nested_model(self):
        result = _collect_models(BugReport)
        assert BugReport in result
        # BugReport has no nested models in our test fixtures

    def test_list_of_model(self):
        result = _collect_models(list[SimpleModel])
        assert SimpleModel in result

    def test_empty_set_bug_fixed(self):
        """Regression test: empty set should work (was falsy bug)."""
        result = set()
        _collect_models(SimpleModel, result)
        assert SimpleModel in result


class TestPydanticToTypescript:
    """Tests for pydantic_to_typescript function."""

    def test_simple_model(self):
        result = pydantic_to_typescript(SimpleModel)
        assert "interface SimpleModel" in result
        assert "name:" in result or "name?:" in result
        assert "string" in result

    def test_includes_jsdoc(self):
        result = pydantic_to_typescript(BugReport)
        # BugReport has descriptions
        assert "/**" in result
        assert "*/" in result

    def test_optional_fields(self):
        result = pydantic_to_typescript(SimpleModel)
        # age has no default so should NOT be optional
        # But this depends on model definition


class TestValueToTypescript:
    """Tests for value_to_typescript function."""

    def test_primitives(self):
        assert value_to_typescript(None) == "null"
        assert value_to_typescript(True) == "true"
        assert value_to_typescript(False) == "false"
        assert value_to_typescript(42) == "42"
        assert value_to_typescript(3.14) == "3.14"
        assert value_to_typescript("hello") == '"hello"'

    def test_string_escaping(self):
        assert value_to_typescript('say "hi"') == '"say \\"hi\\""'
        assert value_to_typescript("line1\nline2") == '"line1\\nline2"'

    def test_simple_list(self):
        result = value_to_typescript([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_simple_dict(self):
        result = value_to_typescript({"a": 1})
        assert "a:" in result
        assert "1" in result

    def test_pydantic_model(self):
        model = SimpleModel(name="test", age=25)
        result = value_to_typescript(model)
        assert "name:" in result
        assert '"test"' in result
        assert "age:" in result
        assert "25" in result

    def test_nested_structure(self):
        data = {"items": [{"name": "a"}, {"name": "b"}]}
        result = value_to_typescript(data)
        assert "items:" in result
        assert '"a"' in result
        assert '"b"' in result


class TestFormatTurn2Typescript:
    """Tests for CodexAdapter.format_turn2_typescript method."""

    def test_basic_output(self):
        adapter = CodexAdapter()
        sig = MockSignature(
            input_fields={"x": (str, None)},
            output_fields={"result": (SimpleModel, "The result")},
        )
        result = adapter.format_turn2_typescript(sig)

        assert "```typescript" in result
        assert "interface SimpleModel" in result
        assert "type Response" in result
        assert "result:" in result
        assert result.count("```") >= 2  # Opening and closing fences

    def test_includes_examples(self):
        adapter = CodexAdapter()

        class SigWithExamples:
            output_fields = {"result": MockFieldInfo(SimpleModel, "The result")}

            class Examples:
                outputs = [SimpleModel(name="test", age=30)]

        result = adapter.format_turn2_typescript(SigWithExamples)

        assert "Example output:" in result
        assert '"test"' in result
        assert "30" in result

    def test_multiple_examples(self):
        adapter = CodexAdapter()

        class SigWithMultipleExamples:
            output_fields = {"result": MockFieldInfo(SimpleModel, "The result")}

            class Examples:
                outputs = [
                    SimpleModel(name="first", age=1),
                    SimpleModel(name="second", age=2),
                ]

        result = adapter.format_turn2_typescript(SigWithMultipleExamples)

        assert "Example outputs:" in result
        assert "// Example 1:" in result
        assert "// Example 2:" in result
        assert '"first"' in result
        assert '"second"' in result


class MockFieldInfo:
    """Mock for testing."""
    def __init__(self, annotation, description=None):
        self.annotation = annotation
        self.description = description
