"""Jujutsu (``jj``) workspaces as Dev Fleet rows.

A secondary jj workspace has a ``.jj`` directory and no ``.git``, so none of the
per-worktree git calls in :mod:`repository` can answer for it. This module is the
jj half of those calls: discovery, the row facts (branch, head, dirty, ahead,
behind), the commit detail, rebase, and removal. :mod:`repository` and
:mod:`worktree_ops` dispatch here on :func:`is_workspace`; everything that is a
question about the shared object store (ancestry, ``gh`` lookups, base-branch
resolution) stays on git against the main checkout, which must be colocated
(``jj git init --colocate``) — a jj commit IS a git commit in that store.

jj vocabulary mapped onto the fleet's git vocabulary:

* **branch** — the newest local bookmark at or behind ``@``.
* **head** — that bookmark's commit, else ``@``. jj snapshots the working copy
  into ``@``, so the head already contains the edits; they are "dirty" only when
  ``@`` carries changes that no bookmark covers.
* **untracked files** — do not exist; jj tracks every non-ignored file.

Every call goes through ``runtime._run_cmd`` (sandbox, trusted PATH, git
neutralizers). ``jj`` is not a system binary on most hosts, so the operator names
it with ``KIROCREW_DEVFLEET_BIN_JJ`` exactly as for ``gh``. jj keeps repo and
workspace config outside the working tree (``jj config path --repo``), so unlike
``.git/config`` an agent-writable checkout cannot name a program for jj to run.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from kiro_crew.apps.builtins.dev_fleet import runtime
from kiro_crew.executors import subprocess_executor

#: ``{workspace root -> workspace name}`` from the latest discovery. Row-level
#: callers hold only a path, and a stat to decide "is this jj?" would put
#: filesystem work on the loop, so discovery records the answer instead.
_WORKSPACES: dict[str, str] = {}

#: Last discovery failure reported, so a host with a ``.jj`` directory but no
#: resolvable ``jj`` logs the reason once instead of on every fleet poll.
_LAST_WARNING: str | None = None

#: Newest local bookmark(s) at or behind ``@``: the workspace's branch.
_BOOKMARKED = "heads(::@ & bookmarks())"
#: What the row reports as HEAD: the branch's commit, else the working copy.
_TIP = f"coalesce({_BOOKMARKED}, @)"
#: ``@`` has changes no bookmark covers. Needs a snapshot to be current.
_UNCOVERED_EDITS = f"(@ ~ empty()) ~ {_BOOKMARKED}"


def name_of(path: str | None) -> str | None:
    """The jj workspace name of *path*, which need not match its directory name."""
    return None if path is None else _WORKSPACES.get(os.path.normpath(path))


def is_workspace(path: str | None) -> bool:
    """Whether *path* is a jj workspace root found by the latest discovery."""
    return path is not None and os.path.normpath(path) in _WORKSPACES


def _quote(text: str) -> str:
    """A revset string literal."""
    return json.dumps(text, ensure_ascii=False)


def _base_ref(remote: str, base: str) -> str:
    """The remote-tracking base bookmark, as git's ``<remote>/<base>``."""
    return f"{_quote(base)}@{_quote(remote)}"


def _own(remote: str, base: str) -> str:
    """The branch's own commits: what its tip has that the base lacks. Uncommitted
    edits in ``@`` are deliberately not among them, as in git."""
    return f"({_base_ref(remote, base)}..{_TIP}) ~ empty()"


async def _jj(
    repo: str,
    *args: str,
    snapshot: bool = False,
    timeout: int = 10,
    mode: str = "standard",
    pre_spawn: Callable[[], Awaitable[str | None]] | None = None,
) -> tuple[int, str, str]:
    """Run jj against the workspace at *repo*; ``(returncode, stdout, stderr)``.

    Reads skip the working-copy snapshot (``--ignore-working-copy``) so a poll
    writes nothing. Pass ``snapshot=True`` where the answer is about the files, and
    for any command that must update them: without it ``op restore`` rewinds the
    repo but leaves the files behind, and the workspace reads as stale.
    """
    cmd = ["jj", "--no-pager", "--color", "never", "-R", repo]
    if not snapshot:
        cmd.append("--ignore-working-copy")
    # cwd is the workspace as well as -R: jj prints file paths relative to it.
    return await runtime._run_cmd(
        [*cmd, *args], cwd=repo, timeout=timeout, mode=mode, pre_spawn=pre_spawn
    )


async def _out(repo: str, *args: str, **kw) -> str | None:
    rc, stdout, _ = await _jj(repo, *args, **kw)
    return stdout.strip() if rc == 0 else None


async def _log(repo: str, revset: str, template: str, **kw) -> str | None:
    return await _out(repo, "log", "--no-graph", "-r", revset, "-T", template, **kw)


async def _count(repo: str, revset: str) -> int | None:
    out = await _log(repo, revset, '"."')
    return None if out is None else len(out)


async def _tip(path: str) -> dict | None:
    """``{oid, ts, bookmarks}`` of the workspace's tip commit."""
    out = await _log(
        path,
        _TIP,
        'commit_id ++ "\\t" ++ committer.timestamp().format("%s") ++ "\\t" '
        '++ local_bookmarks.map(|b| b.name()).join("\\t")',
    )
    if not out:
        return None
    oid, ts, *bookmarks = out.splitlines()[0].split("\t")
    return {
        "oid": oid,
        "ts": int(ts) if ts.isdigit() else None,
        "bookmarks": [b for b in bookmarks if b],
    }


async def list_workspaces(repo: str) -> list[dict]:
    """Secondary jj workspaces of the colocated main checkout *repo*, as worktree
    records shaped like ``git worktree list`` entries (plus ``workspace``).

    Refreshes the registry :func:`is_workspace` reads. Never raises: a host
    where jj cannot run keeps its git worktrees and logs why the rest are absent.
    """
    global _WORKSPACES, _LAST_WARNING
    loop = asyncio.get_running_loop()
    if not await loop.run_in_executor(subprocess_executor(), (Path(repo) / ".jj").is_dir):
        _WORKSPACES = {}
        return []
    # A plain `git fetch` or `git pull` in the main checkout moves git refs, and
    # a secondary workspace does not notice until something imports them.
    await _jj(repo, "git", "import", timeout=20)
    rc, stdout, stderr = await _jj(repo, "workspace", "list", "-T", 'name ++ "\\n"')
    if rc != 0:
        warning = runtime._redact((stderr or stdout).strip())[: runtime._GIT_ERR_MAX]
        if warning != _LAST_WARNING:
            _LAST_WARNING = warning
            runtime.logger.warning("dev-fleet: jj workspaces not listed in %s: %s", repo, warning)
        _WORKSPACES = {}
        return []
    _LAST_WARNING = None
    names = [n for n in stdout.splitlines() if n and not n.startswith("-")]
    roots = await asyncio.gather(*(_out(repo, "workspace", "root", "--name", n) for n in names))
    found = [
        (os.path.normpath(root), name)
        for name, root in zip(names, roots)
        if root and os.path.normpath(root) != os.path.normpath(repo)
    ]
    present = await loop.run_in_executor(
        subprocess_executor(), lambda: [(r, n) for r, n in found if os.path.isdir(r)]
    )
    tips = await asyncio.gather(*(_tip(root) for root, _ in present))
    _WORKSPACES = dict(present)
    return [
        {
            "path": root,
            "head": (tip or {}).get("oid"),
            "branch": ((tip or {}).get("bookmarks") or [None])[0],
            "workspace": name,
        }
        for (root, name), tip in zip(present, tips)
    ]


async def tip_oid(path: str) -> str | None:
    """Full commit id of the workspace's head (the analogue of ``rev-parse HEAD``)."""
    tip = await _tip(path)
    return tip["oid"] if tip else None


async def dirty(path: str) -> bool | None:
    """Whether ``@`` carries edits no bookmark covers; ``None`` when jj cannot say."""
    out = await _log(path, _UNCOVERED_EDITS, '"x"', snapshot=True)
    return None if out is None else bool(out)


async def info(path: str, remote: str, base: str) -> dict:
    """The row facts ``repository._git_info`` reports for a git worktree."""
    ref = _base_ref(remote, base)
    is_dirty = await dirty(path)
    tip = await _tip(path)
    behind = await _count(path, f"{ref} ~ ::@")
    return {
        "branch": tip["bookmarks"][0] if tip and tip["bookmarks"] else None,
        "head": tip["oid"][:7] if tip else None,
        "head_oid": tip["oid"] if tip else None,
        "dirty": bool(is_dirty),
        "ahead": 0,
        "behind": behind or 0,
        "last_updated_at": tip["ts"] if tip else None,
    }


async def ahead(path: str, remote: str, base: str) -> int | None:
    """Non-empty commits on the workspace that the base lacks."""
    return await _count(path, _own(remote, base))


async def recent_commits(path: str, remote: str, base: str, limit: int) -> list[dict]:
    """Newest-first ``{hash, when, subject, body}`` of the branch's own commits."""
    out = await _log(
        path,
        _own(remote, base),
        'commit_id.short(7) ++ "\\t" ++ committer.timestamp().ago() ++ "\\t" '
        '++ json(description) ++ "\\n"',
        timeout=12,
    )
    commits: list[dict] = []
    for line in (out or "").splitlines()[:limit]:
        short, when, raw = line.split("\t", 2)
        subject, _, body = json.loads(raw).partition("\n")
        commits.append({"hash": short, "when": when, "subject": subject, "body": body.strip()})
    return commits


async def changed_files(path: str, remote: str, base: str) -> str | None:
    """Files the branch changes since it forked from the base (``base...HEAD``)."""
    return await _out(
        path,
        "diff",
        "--from",
        f"fork_point({_base_ref(remote, base)} | {_TIP})",
        "--to",
        _TIP,
        "--name-only",
        timeout=15,
    )


async def bookmark_oid(repo: str, name: str) -> str | None:
    """The one commit local bookmark *name* names; ``None`` when absent or conflicted."""
    out = await _log(repo, f"bookmarks(exact:{_quote(name)})", 'commit_id ++ "\\n"')
    lines = (out or "").splitlines()
    return lines[0] if len(lines) == 1 else None


async def forget_bookmark(repo: str, name: str, expected_oid: str) -> str | None:
    """Drop local bookmark *name* if it still names *expected_oid*.

    jj has no compare-and-swap, so this is check-then-act under the caller's
    mutation lock; the commit stays in the operation log either way. ``forget``
    rather than ``delete`` so the next push does not schedule a remote deletion.
    """
    if await bookmark_oid(repo, name) != expected_oid:
        return None
    rc, stdout, _ = await _jj(repo, "bookmark", "forget", f"exact:{name}", timeout=10)
    return stdout.strip() if rc == 0 else None


async def rebase(path: str, remote: str, base: str) -> dict:
    """Fetch the base and rebase the workspace's branch onto it; undo on conflict.

    jj records a conflict in the rebased commit instead of stopping, so the
    equivalent of ``git rebase --abort`` is restoring the operation taken just
    before the rebase.
    """
    rc, stdout, stderr = await _jj(
        path, "git", "fetch", "--remote", remote, "--branch", f"exact:{base}", timeout=90
    )
    if rc != 0:
        return {"ok": False, "error": f"jj git fetch {remote} {base} failed"}
    before = await _out(path, "op", "log", "--no-graph", "-n", "1", "-T", "id")
    ref = _base_ref(remote, base)
    rc, stdout, stderr = await _jj(
        path, "rebase", "-b", "@", "-d", ref, snapshot=True, timeout=180, mode="strict"
    )
    # Redact the FULL text, then take the tail: a tail cut of already-redacted
    # text can at worst split a redaction marker, never a secret.
    tail = runtime._redact((stdout + stderr).strip())[-200:]
    if rc != 0:
        return {"ok": False, "error": f"jj rebase failed. {tail}"}
    conflicted = await _log(path, f"conflicts() & ({ref}..@)", '"x"')
    if conflicted is None or conflicted:
        restored = (
            before is not None and (await _jj(path, "op", "restore", before, snapshot=True))[0] == 0
        )
        if not restored:
            return {
                "ok": False,
                "conflict": bool(conflicted),
                "error": (
                    "rebase conflict AND the operation could not be undone — "
                    f"run `jj op log` and restore it manually. {tail}"
                ),
            }
        return {"ok": False, "conflict": True, "error": f"rebase conflict (aborted). {tail}"}
    g = await info(path, remote, base)
    return {"ok": True, "rebased": True, "head": g["head"], "behind": g["behind"]}


async def remove(
    repo: str,
    path: str,
    *,
    allow_dirty: bool,
    pre_spawn: Callable[[], Awaitable[str | None]],
) -> tuple[int, str, str]:
    """Forget the workspace at *path* and delete its directory.

    Mirrors the ``(rc, stdout, stderr)`` of ``git worktree remove``. ``forget`` does
    not look at the files, so the dirty check git does for free is repeated here
    immediately before it. ``pre_spawn`` is the caller's lease gate, run just
    before the forget, which is the mutation that matters.
    """
    name = _WORKSPACES.get(os.path.normpath(path))
    if name is None:
        return 1, "", "not a known jj workspace"
    if not allow_dirty and await dirty(path) is not False:
        return 1, "", "worktree has uncommitted changes"
    rc, stdout, stderr = await _jj(
        repo, "workspace", "forget", name, timeout=30, pre_spawn=pre_spawn
    )
    if rc != 0:
        return rc, stdout, stderr
    # Only a directory that is still a jj workspace root is deleted: the path came
    # from discovery, but it names a directory this function is about to remove.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(subprocess_executor(), _delete_workspace_dir, path)
    except OSError as exc:
        return 1, "", f"workspace forgotten but its directory was not deleted: {exc}"
    return 0, "", ""


def _delete_workspace_dir(path: str) -> None:
    """Blocking — executor only."""
    root = Path(path)
    if root.is_symlink() or not (root / ".jj").is_dir():
        raise OSError(f"{path} is not a jj workspace root")
    shutil.rmtree(root)
