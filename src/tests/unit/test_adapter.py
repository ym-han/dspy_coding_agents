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
