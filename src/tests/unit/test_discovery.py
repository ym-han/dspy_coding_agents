"""Unit tests for codex binary discovery."""

from pathlib import Path
from unittest.mock import patch

import pytest

from codex.discovery import find_codex_binary
from codex.exceptions import UnsupportedPlatformError


class TestFindCodexBinary:
    """Tests for find_codex_binary function."""

    def test_override_returns_override_path(self):
        """Explicit override should be returned directly."""
        result = find_codex_binary("/custom/path/to/codex")

        assert result == Path("/custom/path/to/codex")

    def test_finds_codex_in_path(self):
        """Should find codex in PATH before checking vendor."""
        with patch("codex.discovery.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/codex"
            result = find_codex_binary()

        assert result == Path("/usr/bin/codex")
        mock_which.assert_called_once_with("codex")

    def test_falls_back_to_vendor_when_not_in_path(self):
        """Should fall back to vendor path when codex not in PATH."""
        with patch("codex.discovery.shutil.which") as mock_which:
            mock_which.return_value = None
            result = find_codex_binary()

        # Should be a path to vendor directory
        assert "vendor" in str(result)
        assert result.name == "codex"

    def test_path_check_before_vendor(self):
        """PATH should be checked before vendor fallback."""
        call_order = []

        def track_which(name):
            call_order.append("which")
            return "/found/codex"

        with patch("codex.discovery.shutil.which", side_effect=track_which):
            find_codex_binary()

        assert call_order == ["which"]

    def test_override_skips_path_check(self):
        """Override should skip PATH check entirely."""
        with patch("codex.discovery.shutil.which") as mock_which:
            find_codex_binary("/my/codex")

        mock_which.assert_not_called()


class TestDetectTarget:
    """Tests for platform detection."""

    def test_linux_x86_64(self):
        """Linux x86_64 should return correct target."""
        with (
            patch("codex.discovery.sys.platform", "linux"),
            patch("codex.discovery.platform.machine", return_value="x86_64"),
        ):
            from codex.discovery import _detect_target

            assert _detect_target() == "x86_64-unknown-linux-musl"

    def test_linux_aarch64(self):
        """Linux aarch64 should return correct target."""
        with (
            patch("codex.discovery.sys.platform", "linux"),
            patch("codex.discovery.platform.machine", return_value="aarch64"),
        ):
            from codex.discovery import _detect_target

            assert _detect_target() == "aarch64-unknown-linux-musl"

    def test_darwin_x86_64(self):
        """macOS x86_64 should return correct target."""
        with (
            patch("codex.discovery.sys.platform", "darwin"),
            patch("codex.discovery.platform.machine", return_value="x86_64"),
        ):
            from codex.discovery import _detect_target

            assert _detect_target() == "x86_64-apple-darwin"

    def test_darwin_arm64(self):
        """macOS arm64 should return correct target."""
        with (
            patch("codex.discovery.sys.platform", "darwin"),
            patch("codex.discovery.platform.machine", return_value="arm64"),
        ):
            from codex.discovery import _detect_target

            assert _detect_target() == "aarch64-apple-darwin"

    def test_unsupported_platform_raises(self):
        """Unsupported platform should raise UnsupportedPlatformError."""
        with (
            patch("codex.discovery.sys.platform", "freebsd"),
            patch("codex.discovery.platform.machine", return_value="x86_64"),
        ):
            from codex.discovery import _detect_target

            with pytest.raises(UnsupportedPlatformError):
                _detect_target()
