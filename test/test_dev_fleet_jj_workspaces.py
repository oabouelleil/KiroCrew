"""Dev Fleet drives jj workspaces through ``jj_workspaces``.

Two tiers. The dispatch tests need no jj: they prove the git-shaped entry points in
``repository`` hand a registered workspace to the jj half. The behaviour tests run a
real jj (skipped when it is absent) against a colocated repo with a local bare
origin, with ``runtime._run_cmd`` swapped for a plain subprocess so the sandbox and
the https/ssh-only transport pin are out of the picture — those belong to
``_run_cmd``, not to the revsets and templates under test.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kiro_crew.apps.builtins.dev_fleet import jj_workspaces, repository, runtime
from kiro_crew.subprocess_utf8 import UTF8_TEXT

_JJ = shutil.which("jj")
_GIT = shutil.which("git")
needs_jj = pytest.mark.skipif(not (_JJ and _GIT), reason="jj and git are required")

_CMD_TIMEOUT_S = 30


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(jj_workspaces, "_WORKSPACES", {})


# --- dispatch (no jj needed) ------------------------------------------------


@pytest.mark.asyncio
async def test_repository_entry_points_dispatch_registered_workspaces(monkeypatch):
    path = "/fleet/ws"
    monkeypatch.setitem(jj_workspaces._WORKSPACES, path, "ws")
    monkeypatch.setattr(repository, "_upstream_remote", _async_value("origin"))
    monkeypatch.setattr(jj_workspaces, "dirty", _async_value(True))
    monkeypatch.setattr(jj_workspaces, "tip_oid", _async_value("abc123"))
    monkeypatch.setattr(jj_workspaces, "ahead", _async_value(2))

    async def no_git(*_a, **_kw):
        raise AssertionError("a jj workspace has no .git; git must not be asked")

    monkeypatch.setattr(repository, "_git", no_git)
    monkeypatch.setattr(runtime, "_run_cmd", no_git)

    assert await repository._real_dirty(path) is True
    assert await repository._dirty_split(path) == (True, [])
    assert await repository._head_oid(path) == "abc123"
    assert await repository._own_commits_count(path) == 2
    assert await repository._git_ahead(path) == 2


@pytest.mark.asyncio
async def test_unregistered_path_still_goes_to_git(monkeypatch):
    seen: list[tuple] = []

    async def fake_git(git_dir, *args, **_kw):
        seen.append((git_dir, args))
        return "deadbeef"

    monkeypatch.setattr(repository, "_git", fake_git)
    assert await repository._head_oid("/fleet/plain") == "deadbeef"
    assert seen == [("/fleet/plain", ("rev-parse", "HEAD"))]


def test_worktree_name_is_the_jj_workspace_name_not_the_directory(monkeypatch):
    monkeypatch.setitem(jj_workspaces._WORKSPACES, "/fleet/dir", "ws-name")
    assert repository.worktree_name("/fleet/dir") == "ws-name"
    assert repository.worktree_name("/fleet/plain") == "plain"
    assert repository.worktree_name("/fleet/dir", is_main=True) == repository.BASE_BRANCH


def _async_value(value):
    async def _f(*_a, **_kw):
        return value

    return _f


# --- behaviour (real jj) ----------------------------------------------------


def _run(cwd: Path, *argv: str, env: dict) -> str:
    done = subprocess.run(
        list(argv), cwd=cwd, env=env, capture_output=True, timeout=_CMD_TIMEOUT_S, **UTF8_TEXT
    )
    assert done.returncode == 0, f"{argv}: {done.stderr}"
    return done.stdout.strip()


@pytest.fixture
def jj_env(tmp_path, monkeypatch):
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
        "GIT_ALLOW_PROTOCOL": "file",
    }

    async def plain_run_cmd(cmd, *, cwd=None, timeout=30, pre_spawn=None, **_kw):
        if pre_spawn is not None and (refusal := await pre_spawn()) is not None:
            return -1, "", refusal
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
    return env


@pytest.fixture
def fleet(tmp_path, jj_env):
    """A colocated main checkout on ``trunk`` with an origin, plus helpers."""
    origin = tmp_path / "origin.git"
    _run(tmp_path, "git", "init", "-q", "--bare", "-b", "trunk", str(origin), env=jj_env)
    main = tmp_path / "main"
    _run(tmp_path, "git", "clone", "-q", str(origin), str(main), env=jj_env)
    (main / "a.txt").write_text("a\n", encoding="utf-8")
    _run(main, "git", "add", "a.txt", env=jj_env)
    _run(main, "git", "commit", "-qm", "init", env=jj_env)
    _run(main, "git", "push", "-q", "origin", "trunk", env=jj_env)
    _run(main, "jj", "git", "init", "--colocate", env=jj_env)

    def workspace(dirname: str, name: str | None = None) -> Path:
        path = tmp_path / dirname
        named = ["--name", name] if name else []
        _run(main, "jj", "workspace", "add", *named, str(path), env=jj_env)
        return path

    def push_upstream(filename: str, text: str) -> None:
        other = tmp_path / "other"
        if not other.exists():
            _run(tmp_path, "git", "clone", "-q", str(origin), str(other), env=jj_env)
        (other / filename).write_text(text, encoding="utf-8")
        _run(other, "git", "add", filename, env=jj_env)
        _run(other, "git", "commit", "-qm", f"upstream {filename}", env=jj_env)
        _run(other, "git", "push", "-q", "origin", "trunk", env=jj_env)

    def feature(path: Path, bookmark: str, filename: str = "f.txt") -> None:
        (path / filename).write_text("f\n", encoding="utf-8")
        _run(path, "jj", "describe", "-m", "feat: thing\n\nbody text", env=jj_env)
        _run(path, "jj", "bookmark", "create", bookmark, "-r", "@", env=jj_env)
        _run(path, "jj", "new", env=jj_env)

    return type(
        "Fleet",
        (),
        {
            "main": main,
            "env": jj_env,
            "workspace": staticmethod(workspace),
            "push_upstream": staticmethod(push_upstream),
            "feature": staticmethod(feature),
        },
    )


@needs_jj
@pytest.mark.asyncio
async def test_workspace_rows_report_branch_ahead_commits_and_files(fleet):
    ws = fleet.workspace("ws-feat")
    fleet.feature(ws, "feat/x")
    fleet.workspace("ws-idle")

    found = {w["workspace"]: w for w in await jj_workspaces.list_workspaces(str(fleet.main))}

    assert set(found) == {"ws-feat", "ws-idle"}  # the main checkout is not a secondary row
    assert found["ws-feat"]["branch"] == "feat/x"
    assert found["ws-idle"]["branch"] == "trunk"  # an idle workspace sits on the base, as in git
    path = found["ws-feat"]["path"]
    assert jj_workspaces.is_workspace(path)
    info = await jj_workspaces.info(path, "origin", "trunk")
    assert info["branch"] == "feat/x"
    assert info["head_oid"] == found["ws-feat"]["head"]
    assert info["dirty"] is False
    assert await jj_workspaces.ahead(path, "origin", "trunk") == 1
    commits = await jj_workspaces.recent_commits(path, "origin", "trunk", 5)
    assert [(c["subject"], c["body"]) for c in commits] == [("feat: thing", "body text")]
    assert (await jj_workspaces.changed_files(path, "origin", "trunk")).split() == ["f.txt"]


@needs_jj
@pytest.mark.asyncio
async def test_a_renamed_workspace_is_named_by_jj_not_by_its_directory(fleet):
    ws = fleet.workspace("some-dir", name="chosen")

    [row] = await jj_workspaces.list_workspaces(str(fleet.main))

    assert row["workspace"] == "chosen"
    assert repository.worktree_name(str(ws)) == "chosen"
    from kiro_crew.pod import runtime as pod_runtime

    assert pod_runtime._jj_workspaces(fleet.main)["chosen"] == ws


@needs_jj
@pytest.mark.asyncio
async def test_edits_beyond_the_branch_are_dirty_and_a_bookmarked_working_copy_is_not(fleet):
    ws = fleet.workspace("ws-feat")
    fleet.feature(ws, "feat/x")
    await jj_workspaces.list_workspaces(str(fleet.main))
    assert await jj_workspaces.dirty(str(ws)) is False

    (ws / "scratch.txt").write_text("edit\n", encoding="utf-8")
    assert await jj_workspaces.dirty(str(ws)) is True
    assert await jj_workspaces.ahead(str(ws), "origin", "trunk") == 1  # edits are not commits

    _run(ws, "jj", "bookmark", "set", "feat/x", "-r", "@", env=fleet.env)
    assert await jj_workspaces.dirty(str(ws)) is False  # now on the branch


@needs_jj
@pytest.mark.asyncio
async def test_rebase_moves_the_branch_and_a_conflict_is_undone(fleet):
    ws = fleet.workspace("ws-feat")
    fleet.feature(ws, "feat/x")
    await jj_workspaces.list_workspaces(str(fleet.main))
    path = str(ws)

    fleet.push_upstream("other.txt", "o\n")
    result = await jj_workspaces.rebase(path, "origin", "trunk")
    assert result["ok"] is True and result["behind"] == 0

    before = await jj_workspaces.tip_oid(path)
    fleet.push_upstream("f.txt", "clashes\n")
    result = await jj_workspaces.rebase(path, "origin", "trunk")
    assert result["ok"] is False and result["conflict"] is True
    assert await jj_workspaces.tip_oid(path) == before
    assert (ws / "f.txt").read_text(encoding="utf-8") == "f\n"  # files rewound, not stale
    assert await jj_workspaces.dirty(path) is False


@needs_jj
@pytest.mark.asyncio
async def test_remove_refuses_dirty_then_forgets_and_deletes(fleet):
    ws = fleet.workspace("ws-feat")
    fleet.feature(ws, "feat/x")
    await jj_workspaces.list_workspaces(str(fleet.main))
    path, repo = str(ws), str(fleet.main)

    async def lease_ok() -> str | None:
        return None

    (ws / "scratch.txt").write_text("edit\n", encoding="utf-8")
    rc, _out, err = await jj_workspaces.remove(repo, path, allow_dirty=False, pre_spawn=lease_ok)
    assert (rc, err) == (1, "worktree has uncommitted changes") and ws.is_dir()

    (ws / "scratch.txt").unlink()
    rc, _out, _err = await jj_workspaces.remove(repo, path, allow_dirty=False, pre_spawn=lease_ok)
    assert rc == 0 and not ws.exists()
    assert await jj_workspaces.list_workspaces(repo) == []


@needs_jj
@pytest.mark.asyncio
async def test_remove_stops_at_a_refused_lease_before_anything_changes(fleet):
    ws = fleet.workspace("ws-feat")
    await jj_workspaces.list_workspaces(str(fleet.main))

    async def lease_lost() -> str | None:
        return "lease lost"

    rc, _out, err = await jj_workspaces.remove(
        str(fleet.main), str(ws), allow_dirty=False, pre_spawn=lease_lost
    )
    assert (rc, err) == (-1, "lease lost") and ws.is_dir()
    assert [w["workspace"] for w in await jj_workspaces.list_workspaces(str(fleet.main))] == [
        "ws-feat"
    ]


@needs_jj
@pytest.mark.asyncio
async def test_forgetting_a_bookmark_requires_the_pinned_commit(fleet):
    ws = fleet.workspace("ws-feat")
    fleet.feature(ws, "feat/x")
    repo = str(fleet.main)
    pinned = await jj_workspaces.bookmark_oid(repo, "feat/x")
    assert pinned

    assert await jj_workspaces.forget_bookmark(repo, "feat/x", "0" * 40) is None
    assert await jj_workspaces.bookmark_oid(repo, "feat/x") == pinned
    assert await jj_workspaces.forget_bookmark(repo, "feat/x", pinned) is not None
    assert await jj_workspaces.bookmark_oid(repo, "feat/x") is None


@pytest.mark.asyncio
async def test_a_repo_without_jj_lists_nothing_and_never_spawns_it(tmp_path, monkeypatch):
    spawned: list[list[str]] = []

    async def record(cmd, **_kw):
        spawned.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(runtime, "_run_cmd", record)
    assert await jj_workspaces.list_workspaces(str(tmp_path)) == []
    assert spawned == []


@needs_jj
def test_pod_checkout_resolution_finds_a_jj_workspace_by_name(fleet):
    from kiro_crew.pod import runtime as pod_runtime

    ws = fleet.workspace("ws-feat")

    found = pod_runtime._jj_workspaces(fleet.main)

    assert found["ws-feat"] == found[str(ws)] == ws
