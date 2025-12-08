"""Unit tests for codex module parsing functions.

Tests JSON → typed dataclass conversions for thread items and events.
"""

import pytest

from codex import (
    AgentMessageItem,
    CommandExecutionItem,
    CommandExecutionStatus,
    ErrorItem,
    FileChangeItem,
    FileUpdateChange,
    McpToolCallItem,
    McpToolCallStatus,
    PatchApplyStatus,
    PatchChangeKind,
    ReasoningItem,
    TodoItem,
    TodoListItem,
    WebSearchItem,
)
from codex.events import (
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemUpdatedEvent,
    ThreadErrorEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
    Usage,
    parse_thread_event,
)
from codex.exceptions import CodexError
from codex.items import parse_thread_item


class TestParseThreadItem:
    """Tests for parse_thread_item function."""

    def test_agent_message(self):
        """AgentMessageItem should parse correctly."""
        payload = {"type": "agent_message", "id": "msg_1", "text": "Hello world"}
        item = parse_thread_item(payload)

        assert isinstance(item, AgentMessageItem)
        assert item.id == "msg_1"
        assert item.text == "Hello world"
        assert item.type == "agent_message"

    def test_reasoning(self):
        """ReasoningItem should parse correctly."""
        payload = {"type": "reasoning", "id": "reason_1", "text": "Thinking..."}
        item = parse_thread_item(payload)

        assert isinstance(item, ReasoningItem)
        assert item.id == "reason_1"
        assert item.text == "Thinking..."

    def test_command_execution(self):
        """CommandExecutionItem should parse with all fields."""
        payload = {
            "type": "command_execution",
            "id": "cmd_1",
            "command": "ls -la",
            "aggregated_output": "file1.txt\nfile2.txt",
            "status": "completed",
            "exit_code": 0,
        }
        item = parse_thread_item(payload)

        assert isinstance(item, CommandExecutionItem)
        assert item.id == "cmd_1"
        assert item.command == "ls -la"
        assert item.aggregated_output == "file1.txt\nfile2.txt"
        assert item.status == CommandExecutionStatus.COMPLETED
        assert item.exit_code == 0

    def test_command_execution_failed(self):
        """CommandExecutionItem with failed status."""
        payload = {
            "type": "command_execution",
            "id": "cmd_2",
            "command": "false",
            "aggregated_output": "",
            "status": "failed",
            "exit_code": 1,
        }
        item = parse_thread_item(payload)

        assert item.status == CommandExecutionStatus.FAILED
        assert item.exit_code == 1

    def test_command_execution_declined(self):
        """CommandExecutionItem with declined status."""
        payload = {
            "type": "command_execution",
            "id": "cmd_3",
            "command": "rm -rf /",
            "aggregated_output": "",
            "status": "declined",
        }
        item = parse_thread_item(payload)

        assert item.status == CommandExecutionStatus.DECLINED
        assert item.exit_code is None

    def test_file_change(self):
        """FileChangeItem should parse with changes list."""
        payload = {
            "type": "file_change",
            "id": "fc_1",
            "changes": [
                {"path": "src/main.py", "kind": "update"},
                {"path": "src/new.py", "kind": "add"},
                {"path": "src/old.py", "kind": "delete"},
            ],
            "status": "completed",
        }
        item = parse_thread_item(payload)

        assert isinstance(item, FileChangeItem)
        assert len(item.changes) == 3
        assert isinstance(item.changes[0], FileUpdateChange)
        assert item.changes[0].path == "src/main.py"
        assert item.changes[0].kind == PatchChangeKind.UPDATE
        assert item.changes[1].kind == PatchChangeKind.ADD
        assert item.changes[2].kind == PatchChangeKind.DELETE
        assert item.status == PatchApplyStatus.COMPLETED

    def test_mcp_tool_call_in_progress(self):
        """McpToolCallItem in progress state."""
        payload = {
            "type": "mcp_tool_call",
            "id": "mcp_1",
            "server": "my-server",
            "tool": "search",
            "arguments": {"query": "test"},
            "status": "in_progress",
        }
        item = parse_thread_item(payload)

        assert isinstance(item, McpToolCallItem)
        assert item.server == "my-server"
        assert item.tool == "search"
        assert item.arguments == {"query": "test"}
        assert item.status == McpToolCallStatus.IN_PROGRESS
        assert item.result is None
        assert item.error is None

    def test_mcp_tool_call_completed_with_result(self):
        """McpToolCallItem completed with result."""
        payload = {
            "type": "mcp_tool_call",
            "id": "mcp_2",
            "server": "db-server",
            "tool": "query",
            "arguments": "SELECT * FROM users",
            "status": "completed",
            "result": {
                "content": [{"type": "text", "text": "Found 5 rows"}],
                "structured_content": {"rows": 5},
            },
        }
        item = parse_thread_item(payload)

        assert item.status == McpToolCallStatus.COMPLETED
        assert item.result is not None
        assert len(item.result.content) == 1
        assert item.result.content[0].type == "text"
        assert item.result.structured_content == {"rows": 5}

    def test_mcp_tool_call_failed_with_error(self):
        """McpToolCallItem failed with error."""
        payload = {
            "type": "mcp_tool_call",
            "id": "mcp_3",
            "server": "api-server",
            "tool": "fetch",
            "arguments": None,
            "status": "failed",
            "error": {"message": "Connection refused"},
        }
        item = parse_thread_item(payload)

        assert item.status == McpToolCallStatus.FAILED
        assert item.error is not None
        assert item.error.message == "Connection refused"

    def test_web_search(self):
        """WebSearchItem should parse correctly."""
        payload = {"type": "web_search", "id": "ws_1", "query": "python async"}
        item = parse_thread_item(payload)

        assert isinstance(item, WebSearchItem)
        assert item.query == "python async"

    def test_error_item(self):
        """ErrorItem should parse correctly."""
        payload = {"type": "error", "id": "err_1", "message": "Something went wrong"}
        item = parse_thread_item(payload)

        assert isinstance(item, ErrorItem)
        assert item.message == "Something went wrong"

    def test_todo_list(self):
        """TodoListItem should parse with items."""
        payload = {
            "type": "todo_list",
            "id": "todo_1",
            "items": [
                {"text": "Fix bug", "completed": False},
                {"text": "Write tests", "completed": True},
            ],
        }
        item = parse_thread_item(payload)

        assert isinstance(item, TodoListItem)
        assert len(item.items) == 2
        assert isinstance(item.items[0], TodoItem)
        assert item.items[0].text == "Fix bug"
        assert item.items[0].completed is False
        assert item.items[1].completed is True

    def test_unsupported_type_raises(self):
        """Unknown type should raise CodexError."""
        payload = {"type": "unknown_type", "id": "x"}

        with pytest.raises(CodexError, match="Unsupported item type"):
            parse_thread_item(payload)

    def test_missing_type_raises(self):
        """Missing type field should raise CodexError."""
        payload = {"id": "x", "text": "hello"}

        with pytest.raises(CodexError, match="Expected string for type"):
            parse_thread_item(payload)

    def test_missing_id_raises(self):
        """Missing id field should raise CodexError."""
        payload = {"type": "agent_message", "text": "hello"}

        with pytest.raises(CodexError, match="Expected string for id"):
            parse_thread_item(payload)


class TestParseThreadEvent:
    """Tests for parse_thread_event function."""

    def test_thread_started(self):
        """ThreadStartedEvent should parse correctly."""
        payload = {"type": "thread.started", "thread_id": "thread_123"}
        event = parse_thread_event(payload)

        assert isinstance(event, ThreadStartedEvent)
        assert event.thread_id == "thread_123"
        assert event.type == "thread.started"

    def test_turn_started(self):
        """TurnStartedEvent should parse correctly."""
        payload = {"type": "turn.started"}
        event = parse_thread_event(payload)

        assert isinstance(event, TurnStartedEvent)
        assert event.type == "turn.started"

    def test_turn_completed(self):
        """TurnCompletedEvent should parse with usage."""
        payload = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 20,
            },
        }
        event = parse_thread_event(payload)

        assert isinstance(event, TurnCompletedEvent)
        assert isinstance(event.usage, Usage)
        assert event.usage.input_tokens == 100
        assert event.usage.output_tokens == 50
        assert event.usage.cached_input_tokens == 20

    def test_turn_failed(self):
        """TurnFailedEvent should parse with error message."""
        payload = {
            "type": "turn.failed",
            "error": {"message": "Rate limit exceeded"},
        }
        event = parse_thread_event(payload)

        assert isinstance(event, TurnFailedEvent)
        assert event.error.message == "Rate limit exceeded"

    def test_item_started(self):
        """ItemStartedEvent should parse with nested item."""
        payload = {
            "type": "item.started",
            "item": {"type": "agent_message", "id": "msg_1", "text": "Starting..."},
        }
        event = parse_thread_event(payload)

        assert isinstance(event, ItemStartedEvent)
        assert isinstance(event.item, AgentMessageItem)
        assert event.item.text == "Starting..."

    def test_item_updated(self):
        """ItemUpdatedEvent should parse with nested item."""
        payload = {
            "type": "item.updated",
            "item": {
                "type": "command_execution",
                "id": "cmd_1",
                "command": "ls",
                "aggregated_output": "file1.txt",
                "status": "in_progress",
            },
        }
        event = parse_thread_event(payload)

        assert isinstance(event, ItemUpdatedEvent)
        assert isinstance(event.item, CommandExecutionItem)
        assert event.item.status == CommandExecutionStatus.IN_PROGRESS

    def test_item_completed(self):
        """ItemCompletedEvent should parse with nested item."""
        payload = {
            "type": "item.completed",
            "item": {"type": "reasoning", "id": "r_1", "text": "Done thinking"},
        }
        event = parse_thread_event(payload)

        assert isinstance(event, ItemCompletedEvent)
        assert isinstance(event.item, ReasoningItem)

    def test_error_event(self):
        """ThreadErrorEvent should parse correctly."""
        payload = {"type": "error", "message": "Connection lost"}
        event = parse_thread_event(payload)

        assert isinstance(event, ThreadErrorEvent)
        assert event.message == "Connection lost"

    def test_unsupported_event_type_raises(self):
        """Unknown event type should raise CodexError."""
        payload = {"type": "unknown.event"}

        with pytest.raises(CodexError, match="Unsupported event type"):
            parse_thread_event(payload)
