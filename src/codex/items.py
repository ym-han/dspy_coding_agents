from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast

from .exceptions import CodexError
from .types import JsonObject, JsonValue


class CommandExecutionStatus(StrEnum):
    """Status of a command execution."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DECLINED = "declined"


class PatchChangeKind(StrEnum):
    """Type of file change in a patch."""

    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


class PatchApplyStatus(StrEnum):
    """Status of a patch application."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class McpToolCallStatus(StrEnum):
    """Status of an MCP tool call."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class McpContentBlock:
    """A content block from an MCP tool result."""

    type: str
    data: JsonObject  # The raw content block data (always a JSON object)


@dataclass(frozen=True, slots=True)
class McpToolCallResult:
    """Result payload returned by the MCP server for successful calls."""

    content: Sequence[McpContentBlock]
    structured_content: JsonValue | None = None


@dataclass(frozen=True, slots=True)
class McpToolCallError:
    """Error reported for failed MCP tool calls."""

    message: str


@dataclass(frozen=True, slots=True)
class CommandExecutionItem:
    """A command executed by the agent."""

    type: Literal["command_execution"] = field(default="command_execution", init=False)
    id: str
    command: str
    aggregated_output: str
    status: CommandExecutionStatus
    exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class FileUpdateChange:
    """A single file change within a patch."""

    path: str
    kind: PatchChangeKind


@dataclass(frozen=True, slots=True)
class FileChangeItem:
    """A set of file changes by the agent."""

    type: Literal["file_change"] = field(default="file_change", init=False)
    id: str
    changes: Sequence[FileUpdateChange]
    status: PatchApplyStatus


@dataclass(frozen=True, slots=True)
class McpToolCallItem:
    """A call to an MCP tool."""

    type: Literal["mcp_tool_call"] = field(default="mcp_tool_call", init=False)
    id: str
    server: str
    tool: str
    arguments: JsonValue  # Can be any JSON-serializable value
    status: McpToolCallStatus
    result: McpToolCallResult | None = None
    error: McpToolCallError | None = None


@dataclass(frozen=True, slots=True)
class AgentMessageItem:
    """Response from the agent (natural language or JSON for structured output)."""

    type: Literal["agent_message"] = field(default="agent_message", init=False)
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningItem:
    """Agent's reasoning summary."""

    type: Literal["reasoning"] = field(default="reasoning", init=False)
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class WebSearchItem:
    """A web search request."""

    type: Literal["web_search"] = field(default="web_search", init=False)
    id: str
    query: str


@dataclass(frozen=True, slots=True)
class ErrorItem:
    """A non-fatal error surfaced as an item."""

    type: Literal["error"] = field(default="error", init=False)
    id: str
    message: str


@dataclass(frozen=True, slots=True)
class TodoItem:
    """An item in the agent's to-do list."""

    text: str
    completed: bool


@dataclass(frozen=True, slots=True)
class TodoListItem:
    """The agent's running to-do list."""

    type: Literal["todo_list"] = field(default="todo_list", init=False)
    id: str
    items: Sequence[TodoItem]


ThreadItem = (
    AgentMessageItem
    | ReasoningItem
    | CommandExecutionItem
    | FileChangeItem
    | McpToolCallItem
    | WebSearchItem
    | TodoListItem
    | ErrorItem
)


def _ensure_str(value: JsonValue, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise CodexError(f"Expected string for {field_name}")


def _ensure_sequence(value: JsonValue, field_name: str) -> Sequence[JsonValue]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return cast("Sequence[JsonValue]", value)
    raise CodexError(f"Expected sequence for {field_name}")


def _parse_changes(values: Iterable[JsonValue]) -> list[FileUpdateChange]:
    changes: list[FileUpdateChange] = []
    for value in values:
        if not isinstance(value, dict):
            raise CodexError("Invalid file change entry")
        path = _ensure_str(value.get("path"), "path")
        kind = _ensure_str(value.get("kind"), "kind")
        try:
            enum_kind = PatchChangeKind(kind)
        except ValueError as exc:
            raise CodexError(f"Unsupported file change kind: {kind}") from exc
        changes.append(FileUpdateChange(path=path, kind=enum_kind))
    return changes


def _parse_todos(values: Iterable[JsonValue]) -> list[TodoItem]:
    todos: list[TodoItem] = []
    for value in values:
        if not isinstance(value, dict):
            raise CodexError("Invalid todo entry")
        text = _ensure_str(value.get("text"), "text")
        completed = bool(value.get("completed", False))
        todos.append(TodoItem(text=text, completed=completed))
    return todos


def parse_thread_item(payload: JsonObject) -> ThreadItem:
    """Parse a JSON object into a ThreadItem."""
    type_name = _ensure_str(payload.get("type"), "type")
    item_id = _ensure_str(payload.get("id"), "id")

    if type_name == "agent_message":
        text = _ensure_str(payload.get("text"), "text")
        return AgentMessageItem(id=item_id, text=text)

    if type_name == "reasoning":
        text = _ensure_str(payload.get("text"), "text")
        return ReasoningItem(id=item_id, text=text)

    if type_name == "command_execution":
        command = _ensure_str(payload.get("command"), "command")
        aggregated_output = _ensure_str(payload.get("aggregated_output"), "aggregated_output")
        status_str = _ensure_str(payload.get("status"), "status")
        try:
            status = CommandExecutionStatus(status_str)
        except ValueError as exc:
            raise CodexError(f"Unsupported command execution status: {status_str}") from exc
        exit_code = payload.get("exit_code")
        exit_value = int(exit_code) if isinstance(exit_code, int) else None
        return CommandExecutionItem(
            id=item_id,
            command=command,
            aggregated_output=aggregated_output,
            status=status,
            exit_code=exit_value,
        )

    if type_name == "file_change":
        changes_raw = _ensure_sequence(payload.get("changes"), "changes")
        status_str = _ensure_str(payload.get("status"), "status")
        try:
            change_status = PatchApplyStatus(status_str)
        except ValueError as exc:
            raise CodexError(f"Unsupported file change status: {status_str}") from exc
        changes = _parse_changes(changes_raw)
        return FileChangeItem(id=item_id, changes=changes, status=change_status)

    if type_name == "mcp_tool_call":
        server = _ensure_str(payload.get("server"), "server")
        tool = _ensure_str(payload.get("tool"), "tool")
        arguments = payload.get("arguments")  # Can be any JSON value
        status_str = _ensure_str(payload.get("status"), "status")
        try:
            call_status = McpToolCallStatus(status_str)
        except ValueError as exc:
            raise CodexError(f"Unsupported MCP tool call status: {status_str}") from exc

        # Parse optional result
        result: McpToolCallResult | None = None
        result_payload = payload.get("result")
        if result_payload is not None and isinstance(result_payload, dict):
            content_raw = result_payload.get("content", [])
            content_blocks: list[McpContentBlock] = []
            if isinstance(content_raw, list):
                for block in content_raw:
                    if isinstance(block, dict):
                        block_type = block.get("type", "unknown")
                        content_blocks.append(
                            McpContentBlock(
                                type=str(block_type),
                                data=block,
                            )
                        )
            structured = result_payload.get("structured_content")
            result = McpToolCallResult(content=content_blocks, structured_content=structured)

        # Parse optional error
        error: McpToolCallError | None = None
        error_payload = payload.get("error")
        if error_payload is not None and isinstance(error_payload, dict):
            error_message = error_payload.get("message", "")
            error = McpToolCallError(message=str(error_message))

        return McpToolCallItem(
            id=item_id,
            server=server,
            tool=tool,
            arguments=arguments,
            status=call_status,
            result=result,
            error=error,
        )

    if type_name == "web_search":
        query = _ensure_str(payload.get("query"), "query")
        return WebSearchItem(id=item_id, query=query)

    if type_name == "error":
        message = _ensure_str(payload.get("message"), "message")
        return ErrorItem(id=item_id, message=message)

    if type_name == "todo_list":
        todos_raw = _ensure_sequence(payload.get("items"), "items")
        todos = _parse_todos(todos_raw)
        return TodoListItem(id=item_id, items=todos)

    raise CodexError(f"Unsupported item type: {type_name}")
