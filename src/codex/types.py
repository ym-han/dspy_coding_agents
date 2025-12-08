"""Type definitions for JSON values and other common types."""

from __future__ import annotations

# Python 3.12+ recursive type alias for JSON values
# This properly types any JSON-serializable value
type JsonPrimitive = str | int | float | bool | None
type JsonArray = list[JsonValue]
type JsonObject = dict[str, JsonValue]
type JsonValue = JsonPrimitive | JsonArray | JsonObject

__all__ = [
    "JsonArray",
    "JsonObject",
    "JsonPrimitive",
    "JsonValue",
]
