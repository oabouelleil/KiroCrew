"""The Git panel routes answer from Jujutsu where a project directory is a jj checkout.

Real jj and git (skipped when either is absent) behind the real routes, with the
Dev Fleet runner swapped for a plain subprocess so the sandbox and the trusted
``jj`` lookup are out of the picture: those belong to ``runtime._run_cmd``, not to
the revsets and templates under test.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.apps.builtins.dev_fleet import runtime
from kiro_crew.dashboard.handlers import api_project_git_log, api_project_git_status
from kiro_crew.subprocess_utf8 import UTF8_TEXT

pytestmark = pytest.mark.skipif(
    not (shutil.which("jj") and shutil.which("git")), reason="jj and git are required"
)

_TIMEOUT_S = 30


class _Slot:
    def __init__(self, project: str) -> None:
        self.project = project


class _State:
    def __init__(self, project: str) -> None:
        self._slots = {"s0": _Slot(project)}


def _app(project: Path) -> web.Application:
    app = web.Application()
    app["state"] = _State(str(project))
    app.router.add_get("/api/project/git/status", api_project_git_status)
    app.router.add_get("/api/project/git/log", api_project_git_log)
    return app


@pytest.fixture(autouse=True)
def _unsandboxed_git(monkeypatch):
    from kiro_crew.dashboard.handlers import files as files_mod

    monkeypatch.setattr(
        files_mod,
        "sandboxed_spawn_argv",
        lambda argv, mode="standard", **kw: (list(argv), dict(kw.get("env") or os.environ), None),
    )


@pytest.fixture(autouse=True)
def _mock_sel():
    with patch("kiro_crew.dashboard.handlers.sel") as m:
        m.return_value = MagicMock()
        yield


@pytest.fixture
def env(tmp_path, monkeypatch) -> dict:
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "JJ_USER": "t",
        "JJ_EMAIL": "t@example.com",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }

    async def plain_run_cmd(cmd, *, cwd=None, timeout=30, **_kw):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        return proc.returncode, out.decode(), err.decode()

    monkeypatch.setattr(runtime, "_run_cmd", plain_run_cmd)
    # The handler's git runs inherit os.environ; stop its upward repo discovery
    # at the test tree and give it an identity.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    return env


def _run(cwd: Path, env: dict, *argv: str) -> None:
    done = subprocess.run(
        list(argv), cwd=cwd, env=env, capture_output=True, timeout=_TIMEOUT_S, **UTF8_TEXT
    )
    assert done.returncode == 0, f"{argv}: {done.stderr}"


@pytest.fixture
def colocated(tmp_path, env) -> Path:
    """A colocated jj checkout: ``trunk`` holds a committed ``a.txt``, ``@`` is empty.

    Git's HEAD is detached, which is where jj leaves it as soon as it moves ``@``.
    """
    main = tmp_path / "main"
    main.mkdir()
    _run(main, env, "git", "init", "-q", "-b", "trunk")
    (main / "a.txt").write_text("a\n", encoding="utf-8")
    _run(main, env, "git", "add", "a.txt")
    _run(main, env, "git", "commit", "-qm", "init")
    _run(main, env, "jj", "git", "init", "--colocate")
    _run(main, env, "jj", "new", "trunk")
    _run(main, env, "git", "checkout", "-q", "--detach")
    return main


def _workspace_without_git(main: Path, env: dict, path: Path) -> Path:
    """A secondary workspace with only ``.jj``.

    jj 0.45 colocates a secondary workspace as a git worktree unless the user's
    config says otherwise; this is the shape that config produces.
    """
    _run(main, env, "jj", "workspace", "add", str(path))
    (path / ".git").unlink(missing_ok=True)
    return path


async def _get(project: Path, route: str) -> tuple[int, dict]:
    async with TestClient(TestServer(_app(project))) as client:
        resp = await client.get(f"/api/project/git/{route}", params={"path": str(project)})
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_colocated_checkout_shows_its_bookmark_not_a_detached_head(colocated):
    (colocated / "a.txt").write_text("changed\n", encoding="utf-8")

    status, body = await _get(colocated, "status")

    assert status == 200
    assert body["branch"] == "trunk"
    assert [(f["path"], f["status"]) for f in body["files"]] == [("a.txt", "M")]


@pytest.mark.asyncio
async def test_colocated_log_comes_from_jj_and_marks_the_head(colocated, env):
    _run(colocated, env, "jj", "describe", "-m", "wip: second")

    status, body = await _get(colocated, "log")

    assert status == 200
    assert [c["message"] for c in body["commits"]] == ["wip: second", "init"]
    assert [c["isHead"] for c in body["commits"]] == [True, False]


@pytest.mark.asyncio
async def test_undescribed_working_copy_is_not_listed_as_a_commit(colocated):
    _, body = await _get(colocated, "log")

    assert [c["message"] for c in body["commits"]] == ["init"]
    assert body["commits"][0]["isHead"] is True


@pytest.mark.asyncio
async def test_secondary_workspace_without_git_is_answered_by_jj(tmp_path, colocated, env):
    ws = _workspace_without_git(colocated, env, tmp_path / "ws")
    (ws / "new.txt").write_text("one\ntwo\n", encoding="utf-8")
    (ws / "a.txt").write_text("changed\n", encoding="utf-8")

    status, body = await _get(ws, "status")

    assert status == 200
    assert body["repo"] is True
    assert body["branch"] == "trunk"
    assert body["repoRoot"] == str(ws.resolve())
    rows = {f["path"]: f for f in body["files"]}
    assert rows["new.txt"]["status"] == "A"
    assert rows["new.txt"]["additions"] == 2
    assert rows["a.txt"]["status"] == "M"
    log_status, log = await _get(ws, "log")
    assert log_status == 200
    assert [c["message"] for c in log["commits"]] == ["init"]


@pytest.mark.asyncio
async def test_a_subdirectory_of_a_workspace_prefixes_paths_like_git(tmp_path, colocated, env):
    ws = _workspace_without_git(colocated, env, tmp_path / "ws")
    (ws / "sub").mkdir()
    (ws / "sub" / "b.txt").write_text("b\n", encoding="utf-8")

    _, body = await _get(ws / "sub", "status")

    assert [f["path"] for f in body["files"]] == ["sub/b.txt"]


@pytest.mark.asyncio
async def test_without_a_runnable_jj_the_panel_keeps_its_git_answer(colocated, monkeypatch):
    async def no_jj(*_a, **_kw):
        return -1, "", "no trusted 'jj' executable"

    monkeypatch.setattr(runtime, "_run_cmd", no_jj)

    status, body = await _get(colocated, "status")
    log_status, log = await _get(colocated, "log")

    assert status == 200
    assert body["branch"] == "HEAD"  # git's detached label, as before jj was involved
    assert log_status == 200
    assert [c["message"] for c in log["commits"]] == ["init"]
