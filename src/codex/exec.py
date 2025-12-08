from __future__ import annotations

import io
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import Thread

from .config import ApprovalMode, ModelReasoningEffort, SandboxMode
from .discovery import find_codex_binary
from .exceptions import ExecExitError, SpawnError

INTERNAL_ORIGINATOR_ENV = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
PYTHON_SDK_ORIGINATOR = "codex_sdk_py"


@dataclass(frozen=True, slots=True)
class ExecArgs:
    """Arguments for executing Codex CLI."""

    input: str
    base_url: str | None = None
    api_key: str | None = None
    thread_id: str | None = None
    images: Sequence[str] | None = None
    model: str | None = None
    sandbox_mode: SandboxMode | None = None
    working_directory: str | None = None
    additional_directories: Sequence[str] | None = None
    skip_git_repo_check: bool = False
    output_schema_path: str | None = None
    model_reasoning_effort: ModelReasoningEffort | None = None
    network_access_enabled: bool | None = None
    web_search_enabled: bool | None = None
    approval_policy: ApprovalMode | None = None


class CodexExec:
    """Executes the Codex CLI binary."""

    def __init__(
        self,
        executable_override: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._binary = find_codex_binary(executable_override)
        self._env_override = env

    def build_command(self, args: ExecArgs) -> list[str]:
        """Build the CLI command from arguments."""
        command = [str(self._binary), "exec", "--experimental-json"]

        if args.model:
            command.extend(["--model", args.model])
        if args.sandbox_mode:
            command.extend(["--sandbox", args.sandbox_mode.value])
        if args.working_directory:
            command.extend(["--cd", args.working_directory])
        if args.additional_directories:
            for dir_path in args.additional_directories:
                command.extend(["--add-dir", dir_path])
        if args.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if args.output_schema_path:
            command.extend(["--output-schema", args.output_schema_path])
        if args.model_reasoning_effort:
            command.extend(
                ["--config", f'model_reasoning_effort="{args.model_reasoning_effort.value}"']
            )
        if args.network_access_enabled is not None:
            value = "true" if args.network_access_enabled else "false"
            command.extend(["--config", f"sandbox_workspace_write.network_access={value}"])
        if args.web_search_enabled is not None:
            value = "true" if args.web_search_enabled else "false"
            command.extend(["--config", f"features.web_search_request={value}"])
        if args.approval_policy:
            command.extend(["--config", f'approval_policy="{args.approval_policy.value}"'])
        if args.images:
            for image_path in args.images:
                command.extend(["--image", image_path])
        if args.thread_id:
            command.extend(["resume", args.thread_id])

        return command

    def run_lines(self, args: ExecArgs) -> Iterator[str]:
        """Execute the command and yield stdout lines."""
        command = self.build_command(args)

        if self._env_override is not None:
            env = dict(self._env_override)
        else:
            env = os.environ.copy()
        env.setdefault(INTERNAL_ORIGINATOR_ENV, PYTHON_SDK_ORIGINATOR)
        if args.base_url:
            env["OPENAI_BASE_URL"] = args.base_url
        if args.api_key:
            env["CODEX_API_KEY"] = args.api_key

        stderr_buffer: list[str] = []

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                env=env,
            )
        except OSError as error:  # pragma: no cover - exercised indirectly
            raise SpawnError(command, error) from error

        if not process.stdin or not process.stdout:
            process.kill()
            raise SpawnError(command, OSError("Missing stdio pipes"))

        stderr_thread: Thread | None = None
        if process.stderr:

            def _drain_stderr(pipe: io.TextIOBase, buffer: list[str]) -> None:
                while True:
                    try:
                        chunk = pipe.readline()
                    except ValueError:
                        break
                    if chunk == "":
                        break
                    buffer.append(chunk)

            stderr_thread = Thread(
                target=_drain_stderr,
                args=(process.stderr, stderr_buffer),
                daemon=True,
            )
            stderr_thread.start()

        try:
            process.stdin.write(args.input)
            process.stdin.close()

            for line in iter(process.stdout.readline, ""):
                yield line.rstrip("\n")

            return_code = process.wait()
            if stderr_thread is not None:
                stderr_thread.join()

            stderr_output = "".join(stderr_buffer)
            if return_code != 0:
                raise ExecExitError(tuple(command), return_code, stderr_output)
        finally:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
            if process.stderr and not process.stderr.closed:
                try:
                    process.stderr.close()
                except ValueError:
                    pass
            if stderr_thread is not None and stderr_thread.is_alive():
                stderr_thread.join(timeout=0.1)
            returncode = process.poll()
            if returncode is None:
                process.kill()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.wait()
