from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import BaseModel as PydanticBaseModel

    SchemaInput = Mapping[str, Any] | type[PydanticBaseModel] | PydanticBaseModel
else:
    SchemaInput = Mapping[str, Any]


class Model(StrEnum):
    """Supported Codex models.

    Default model is GPT_5_1_CODEX_MAX.
    """

    # Production models (recommended)
    GPT_5_1_CODEX_MAX = "gpt-5.1-codex-max"
    GPT_5_1_CODEX = "gpt-5.1-codex"
    GPT_5_1_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT_5_1 = "gpt-5.1"

    # Deprecated models (still functional)
    GPT_5_CODEX = "gpt-5-codex"
    GPT_5_CODEX_MINI = "gpt-5-codex-mini"
    GPT_5 = "gpt-5"

    # Other supported models
    O3 = "o3"
    O4_MINI = "o4-mini"
    CODEX_MINI_LATEST = "codex-mini-latest"
    GPT_4_1 = "gpt-4.1"
    GPT_4O = "gpt-4o"


# Default model constant
DEFAULT_MODEL = Model.GPT_5_1_CODEX_MAX


class ApprovalMode(StrEnum):
    """Command approval policy."""

    NEVER = "never"
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    UNTRUSTED = "untrusted"


class SandboxMode(StrEnum):
    """Sandbox policy for file system access."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class ModelReasoningEffort(StrEnum):
    """Reasoning effort level for model inference."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    X_HIGH = "x-high"


class ModelVerbosity(StrEnum):
    """Output verbosity for GPT-5 models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class CodexOptions:
    """Options for the Codex client."""

    codex_path_override: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ThreadOptions:
    """Options for a conversation thread."""

    model: str | None = None
    sandbox_mode: SandboxMode | None = None
    working_directory: str | None = None
    skip_git_repo_check: bool = False
    model_reasoning_effort: ModelReasoningEffort | None = None
    network_access_enabled: bool | None = None
    web_search_enabled: bool | None = None
    approval_policy: ApprovalMode | None = None
    additional_directories: Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class TurnOptions:
    """Options for a single turn in a thread."""

    output_schema: SchemaInput | None = None
