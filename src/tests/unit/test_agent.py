"""Unit tests for CodexAgent utilities.

Tests helper functions used by CodexAgent.
"""

import pytest

from codex_dspy.agent import _strip_json_fences


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
