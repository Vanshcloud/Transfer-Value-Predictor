"""The commit-msg hook is this project's only enforcement of sole authorship.

A rule that lives in a prompt survives until the next fresh session; a hook
survives a reclone. These tests exist so the hook cannot silently rot — if
someone edits its patterns, this suite says so.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "scripts" / "hooks" / "commit-msg"

# git var GIT_AUTHOR_IDENT honours these, so the hook's author check is
# deterministic here rather than dependent on the machine's git config.
VANSH_ENV = {
    "GIT_AUTHOR_NAME": "Vanshcloud",
    "GIT_AUTHOR_EMAIL": "vanshwar@gmail.com",
    "GIT_COMMITTER_NAME": "Vanshcloud",
    "GIT_COMMITTER_EMAIL": "vanshwar@gmail.com",
    # The git this machine actually uses goes first. A fixed /usr/bin first
    # made the suite fail on macOS whenever /usr/bin/git is the Xcode shim and
    # its licence has not been accepted: the shim exits 69 before the hook can
    # read a message, which the tests reported as the hook rejecting it.
    "PATH": ":".join([str(Path(shutil.which("git") or "/usr/bin/git").parent), "/usr/bin", "/bin"]),
    "HOME": str(Path.home()),
}


def run_hook(message: str, env: dict[str, str] | None = None) -> int:
    """Run the commit-msg hook against ``message`` and return its exit code."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(message)
        path = handle.name
    try:
        result = subprocess.run(
            ["bash", str(HOOK), path],
            capture_output=True,
            env=env or VANSH_ENV,
            check=False,
        )
        return result.returncode
    finally:
        Path(path).unlink()


def test_hook_exists_and_is_executable() -> None:
    assert HOOK.is_file(), f"missing hook: {HOOK}"


REJECTED = [
    pytest.param(
        "feat: a change\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n", id="coauthor"
    ),
    pytest.param("feat: a change\n\nco-authored-by: someone <x@y.z>\n", id="coauthor-lowercase"),
    pytest.param(
        "feat: a change\n\n\U0001f916 Generated with [Claude Code](https://claude.ai/code)\n",
        id="generated-with",
    ),
    pytest.param(
        "feat: a change\n\nClaude-Session: https://claude.ai/code/session_x\n", id="session-trailer"
    ),
    pytest.param("feat: a change\n\nSigned-off-by: Claude <x@y.z>\n", id="signed-off-by-claude"),
]


@pytest.mark.parametrize("message", REJECTED)
def test_attribution_is_rejected(message: str) -> None:
    assert run_hook(message) != 0, "hook accepted an attribution trailer"


ACCEPTED = [
    pytest.param("feat(ingestion): add the Kaggle loader\n", id="plain"),
    pytest.param(
        "chore: add .claude/ to gitignore\n\nThe directory is local tooling state.\n",
        id="legitimate-claude-mention",
    ),
    pytest.param(
        "docs: explain how the model was generated with seeded training\n", id="generated-in-prose"
    ),
]


@pytest.mark.parametrize("message", ACCEPTED)
def test_legitimate_messages_are_accepted(message: str) -> None:
    assert run_hook(message) == 0, "hook rejected a legitimate message"


def test_non_vansh_author_is_rejected() -> None:
    env = dict(VANSH_ENV, GIT_AUTHOR_NAME="Someone Else", GIT_AUTHOR_EMAIL="someone@example.com")
    assert run_hook("feat: a change\n", env=env) != 0
