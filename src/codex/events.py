from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .exceptions import CodexError
from .items import ThreadItem, parse_thread_item
from .types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage statistics for a turn."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ThreadError:
    """Fatal error emitted by the stream."""

    message: str


@dataclass(frozen=True, slots=True)
class ThreadStartedEvent:
    """Emitted when a new thread is started."""

    type: Literal["thread.started"] = field(default="thread.started", init=False)
    thread_id: str


@dataclass(frozen=True, slots=True)
class TurnStartedEvent:
    """Emitted when a turn is started by sending a new prompt."""

    type: Literal["turn.started"] = field(default="turn.started", init=False)


@dataclass(frozen=True, slots=True)
class TurnCompletedEvent:
    """Emitted when a turn is completed."""

    type: Literal["turn.completed"] = field(default="turn.completed", init=False)
    usage: Usage


@dataclass(frozen=True, slots=True)
class TurnFailedEvent:
    """Indicates that a turn failed with an error."""

    type: Literal["turn.failed"] = field(default="turn.failed", init=False)
    error: ThreadError


@dataclass(frozen=True, slots=True)
class ItemStartedEvent:
    """Emitted when a new item is added to the thread."""

    type: Literal["item.started"] = field(default="item.started", init=False)
    item: ThreadItem


@dataclass(frozen=True, slots=True)
class ItemUpdatedEvent:
    """Emitted when an item is updated."""

    type: Literal["item.updated"] = field(default="item.updated", init=False)
    item: ThreadItem


@dataclass(frozen=True, slots=True)
class ItemCompletedEvent:
    """Signals that an item has reached a terminal state."""

    type: Literal["item.completed"] = field(default="item.completed", init=False)
    item: ThreadItem


@dataclass(frozen=True, slots=True)
class ThreadErrorEvent:
    """Unrecoverable error emitted by the event stream."""

    type: Literal["error"] = field(default="error", init=False)
    message: str


ThreadEvent = (
    ThreadStartedEvent
    | TurnStartedEvent
    | TurnCompletedEvent
    | TurnFailedEvent
    | ItemStartedEvent
    | ItemUpdatedEvent
    | ItemCompletedEvent
    | ThreadErrorEvent
)


def _ensure_dict(payload: JsonValue) -> JsonObject:
    if isinstance(payload, dict):
        return payload
    raise CodexError("Event payload must be an object")


def _ensure_str(value: JsonValue, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise CodexError(f"Expected string for {field_name}")


def _ensure_int(value: JsonValue, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CodexError(f"Expected integer for {field_name}")


def _parse_usage(payload: JsonValue) -> Usage:
    data = _ensure_dict(payload)
    return Usage(
        input_tokens=_ensure_int(data.get("input_tokens"), "input_tokens"),
        cached_input_tokens=_ensure_int(data.get("cached_input_tokens"), "cached_input_tokens"),
        output_tokens=_ensure_int(data.get("output_tokens"), "output_tokens"),
    )


def parse_thread_event(payload: JsonObject) -> ThreadEvent:
    """Parse a JSON object into a ThreadEvent."""
    type_name = _ensure_str(payload.get("type"), "type")

    if type_name == "thread.started":
        thread_id = _ensure_str(payload.get("thread_id"), "thread_id")
        return ThreadStartedEvent(thread_id=thread_id)

    if type_name == "turn.started":
        return TurnStartedEvent()

    if type_name == "turn.completed":
        usage = _parse_usage(payload.get("usage"))
        return TurnCompletedEvent(usage=usage)

    if type_name == "turn.failed":
        error_payload = _ensure_dict(payload.get("error"))
        message = _ensure_str(error_payload.get("message"), "error.message")
        return TurnFailedEvent(error=ThreadError(message=message))

    if type_name in {"item.started", "item.updated", "item.completed"}:
        item_data = payload.get("item")
        if not isinstance(item_data, dict):
            raise CodexError("item must be an object")
        item = parse_thread_item(item_data)
        if type_name == "item.started":
            return ItemStartedEvent(item=item)
        if type_name == "item.updated":
            return ItemUpdatedEvent(item=item)
        return ItemCompletedEvent(item=item)

    if type_name == "error":
        message = _ensure_str(payload.get("message"), "message")
        return ThreadErrorEvent(message=message)

    raise CodexError(f"Unsupported event type: {type_name}")
