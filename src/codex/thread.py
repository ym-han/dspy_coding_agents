from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from .config import CodexOptions, ThreadOptions, TurnOptions
from .events import (
    ItemCompletedEvent,
    ThreadErrorEvent,
    ThreadEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    Usage,
    parse_thread_event,
)
from .exceptions import JsonParseError, ThreadRunError
from .exec import CodexExec, ExecArgs
from .items import AgentMessageItem, ThreadItem
from .schema import prepare_schema_file


class TextInput(TypedDict):
    """A text input to send to the agent."""

    type: Literal["text"]
    text: str


class LocalImageInput(TypedDict):
    """A local image input to send to the agent."""

    type: Literal["local_image"]
    path: str


UserInput = TextInput | LocalImageInput
Input = str | Sequence[UserInput]


def _normalize_input(input_value: Input) -> tuple[str, list[str]]:
    """Normalize input to prompt string and images list."""
    if isinstance(input_value, str):
        return input_value, []

    prompt_parts: list[str] = []
    images: list[str] = []
    for item in input_value:
        if item.get("type") == "text":
            prompt_parts.append(item.get("text", ""))
        elif item.get("type") == "local_image":
            images.append(item.get("path", ""))

    return "\n\n".join(prompt_parts), images


@dataclass(frozen=True, slots=True)
class ThreadRunResult:
    """Result of a completed turn."""

    items: tuple[ThreadItem, ...]
    final_response: str
    usage: Usage | None


@dataclass(frozen=True, slots=True)
class ThreadStream:
    """Streaming events from a turn."""

    events: Iterator[ThreadEvent]

    def __iter__(self) -> Iterator[ThreadEvent]:
        return self.events


class Thread:
    """A conversation thread with the Codex agent."""

    def __init__(
        self,
        exec_client: CodexExec,
        codex_options: CodexOptions,
        thread_options: ThreadOptions,
        thread_id: str | None = None,
    ) -> None:
        self._exec = exec_client
        self._codex_options = codex_options
        self._thread_options = thread_options
        self._id = thread_id

    @property
    def id(self) -> str | None:
        """Thread ID, populated after the first turn starts."""
        return self._id

    def run_streamed(
        self, input_value: Input, turn_options: TurnOptions | None = None
    ) -> ThreadStream:
        """Run a turn and stream events as they are produced."""
        events = self._stream_events(input_value, turn_options)
        return ThreadStream(events=events)

    def run(self, input_value: Input, turn_options: TurnOptions | None = None) -> ThreadRunResult:
        """Run a turn and return the completed result."""
        final_response = ""
        items: list[ThreadItem] = []
        usage: Usage | None = None
        failure_message: str | None = None

        for event in self._stream_events(input_value, turn_options):
            if isinstance(event, ThreadErrorEvent):
                raise ThreadRunError(event.message)
            if isinstance(event, TurnFailedEvent):
                failure_message = event.error.message
                break
            if isinstance(event, TurnCompletedEvent):
                usage = event.usage
            if isinstance(event, ItemCompletedEvent):
                item = event.item
                items.append(item)
                if isinstance(item, AgentMessageItem):
                    final_response = item.text

        if failure_message is not None:
            raise ThreadRunError(failure_message)

        return ThreadRunResult(items=tuple(items), final_response=final_response, usage=usage)

    def _stream_events(
        self,
        input_value: Input,
        turn_options: TurnOptions | None,
    ) -> Iterator[ThreadEvent]:
        turn = turn_options or TurnOptions()
        prompt, images = _normalize_input(input_value)
        with prepare_schema_file(turn.output_schema) as schema_file:
            exec_args = ExecArgs(
                input=prompt,
                base_url=self._codex_options.base_url,
                api_key=self._codex_options.api_key,
                thread_id=self._id,
                images=images if images else None,
                model=self._thread_options.model,
                sandbox_mode=self._thread_options.sandbox_mode,
                working_directory=self._thread_options.working_directory,
                additional_directories=self._thread_options.additional_directories,
                skip_git_repo_check=self._thread_options.skip_git_repo_check,
                output_schema_path=str(schema_file.path) if schema_file.path else None,
                model_reasoning_effort=self._thread_options.model_reasoning_effort,
                network_access_enabled=self._thread_options.network_access_enabled,
                web_search_enabled=self._thread_options.web_search_enabled,
                approval_policy=self._thread_options.approval_policy,
            )
            command = tuple(self._exec.build_command(exec_args))
            for line in self._exec.run_lines(exec_args):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise JsonParseError(line, command) from error

                event = parse_thread_event(payload)
                if isinstance(event, ThreadStartedEvent):
                    self._id = event.thread_id
                yield event
