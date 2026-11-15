"""``session_set_backend``: one session moves another onto another harness.

The verb is ``authorize_target`` (operation ``set_backend``, the same reach as
``session_set_model``) around ``switch_slot_harness``, with the gate run again at the
switch's commit. Pinned here: an allowed caller switches an idle worker; a caller
without reach, a busy target, the preview being off and a harness that cannot start are
refused with nothing changed; a target that leaves the caller's reach during the
switch is refused at the commit; each outcome is audited; and the route and the MCP
tool carry it.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from chat_test_helpers import _make_state

from kiro_crew import members
from kiro_crew.agent_sdk.backends import backend_from_stored
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.dashboard import chat_handlers
from kiro_crew.dashboard import session_control as sc
from kiro_crew.dashboard.chat_utils import slot_history_key
from kiro_crew.dashboard.handlers import session_control as handlers_sc
from kiro_crew.mcp_dashboard import TABLE
from kiro_crew.mcp_tools.dashboard_client import InMemoryDashboardClient
from kiro_crew.mcp_tools.table import Caller, ToolContext
from kiro_crew.providers.acp import AcpProvider


@pytest.fixture(autouse=True)
def _enabled(monkeypatch, _floor_monkeypatch):
    from kiro_crew import session_harness_preview

    monkeypatch.setattr(sc, "session_control_enabled", lambda: True)
    _floor_monkeypatch.setattr(session_harness_preview, "session_harness_enabled", lambda: True)


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    """Kiro, Claude and Codex can start here; Pi is allowed but missing."""
    from kiro_crew import acp_backends
    from kiro_crew.agent_sdk import backend_install
    from kiro_crew.agent_sdk.backend_install import INSTALLED, MISSING, BackendInstallState

    installs = {"": INSTALLED, "claude": INSTALLED, "codex": INSTALLED, "pi": MISSING}
    monkeypatch.setattr(acp_backends, "selectable_backend_values", lambda: sorted(installs))
    monkeypatch.setattr(
        backend_install,
        "probe_backend",
        lambda b: BackendInstallState(
            b,
            b or "kiro",
            installs[b],
            missing_components=("pi-acp",) if installs[b] == MISSING else (),
            install_command="",
        ),
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A conductor ``chat-1`` and its idle worker ``chat-2`` running Claude Code."""
    state = _make_state(tmp_path)
    state.sessions.mirror_accepts_inbound = MagicMock(return_value=False)
    caller = state.get_or_create_slot("chat-1")
    worker = state.get_or_create_slot("chat-2")
    worker._created_by = caller.key
    worker.backend = "claude"
    members.note_session_backend_pick(worker.key, "claude")
    live = {"backend": "claude", "busy": False}

    def provider(_key):
        if live["backend"] is None:
            return None
        p = MagicMock(spec=AcpProvider)
        p.capabilities = capabilities_for(live["backend"])
        p.has_active_turn.return_value = live["busy"]
        return p

    state.sessions.get_provider = MagicMock(side_effect=provider)
    resets: list[str] = []

    async def reset(_state, _slot, key, *, switch_kind):
        resets.append(key)
        live["backend"] = None
        return True

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", reset)
    monkeypatch.setattr(chat_handlers, "_subagents_attached_response", AsyncMock(return_value=None))

    async def configured(s, config=None):
        pick = backend_from_stored(s.backend)
        return pick if isinstance(pick, str) else ""

    monkeypatch.setattr(chat_handlers, "_configured_backend_for_slot", configured)
    audits: list[dict] = []
    monkeypatch.setattr(sc, "_audit", lambda **kw: audits.append(kw))
    return SimpleNamespace(
        state=state, caller=caller, worker=worker, live=live, resets=resets, audits=audits
    )


def _set_backend(world, backend, *, target="chat-2", caller_fenced=None) -> dict:
    return asyncio.run(
        sc.set_backend_target(
            world.state,
            caller_session_key=slot_history_key(world.caller),
            target=target,
            backend=backend,
            caller_fenced=caller_fenced,
        )
    )


def _refused(world, backend, **kw) -> sc.SessionControlError:
    with pytest.raises(sc.SessionControlError) as exc:
        _set_backend(world, backend, **kw)
    return exc.value


def _unchanged(world) -> None:
    assert world.worker.backend == "claude"
    assert members.session_backend_pick(world.worker.key) == "claude"
    assert world.resets == []


def test_an_allowed_caller_switches_an_idle_worker(world):
    out = _set_backend(world, "codex")

    assert out == {"ok": True, "target": "chat-2", "backend": "codex", "harness": out["harness"]}
    assert world.worker.backend == "codex"
    assert world.resets == [slot_history_key(world.worker)]
    assert world.audits[-1]["operation"] == "set_backend"
    assert world.audits[-1]["outcome"] == "allowed"
    assert world.audits[-1]["detail"] == {"backend": "codex"}


def test_a_caller_without_reach_is_refused(world):
    """A fenced caller reaches only what it created. Mutation guard: dropping the
    gate lets a crew member move the person's own chat."""
    world.worker._created_by = ""

    error = _refused(world, "codex", caller_fenced=True)

    assert error.status == 403
    _unchanged(world)


def test_the_caller_itself_is_refused(world):
    error = _refused(world, "codex", target="chat-1")

    assert error.code == "self_target"
    assert world.resets == []


def test_a_busy_target_is_refused_and_audited(world):
    world.live["busy"] = True

    error = _refused(world, "codex")

    assert (error.status, error.code) == (409, "turn_in_flight")
    _unchanged(world)
    assert world.audits[-1]["outcome"] == "denied"
    assert world.audits[-1]["detail"] == {"code": "turn_in_flight", "backend": "codex"}


def test_the_preview_off_is_refused(world, _floor_monkeypatch):
    from kiro_crew import session_harness_preview

    _floor_monkeypatch.setattr(session_harness_preview, "session_harness_enabled", lambda: False)

    error = _refused(world, "codex")

    assert (error.status, error.code) == (403, "session_harness_preview_off")
    _unchanged(world)
    # Checked first: an unknown target or a bad name reads the same, and nothing is
    # audited, so the tool says nothing about which sessions exist.
    assert _refused(world, "", target="no-such-session").code == "session_harness_preview_off"
    assert world.audits == []


@pytest.mark.parametrize(
    "name, code", [("nope", "unknown_backend"), ("pi", "backend_not_installed")]
)
def test_a_harness_that_cannot_start_names_the_ones_that_can(world, name, code):
    error = _refused(world, name)

    assert (error.status, error.code) == (400, code)
    assert "codex" in error.message and "kiro" in error.message
    _unchanged(world)


@pytest.mark.parametrize("raw", ["", "  ", None, 3])
def test_a_missing_name_is_refused_before_the_gate(world, raw):
    error = _refused(world, raw, target="no-such-session")

    assert (error.status, error.code) == (400, "invalid_backend")


def test_a_target_that_leaves_reach_during_the_switch_is_refused_at_the_commit(world, monkeypatch):
    """The gate runs again right before the commit, and its own refusal comes back.
    Mutation guard: authorizing only up front lets the switch reach a session that
    became mirrored meanwhile."""

    async def mirror_meanwhile(*_a, **_kw):
        world.state.sessions.set_mirror_link(slot_history_key(world.worker), "C0FFEE", "1.2")
        return None

    monkeypatch.setattr(chat_handlers, "_subagents_attached_response", mirror_meanwhile)

    error = _refused(world, "codex")

    assert error.code == "mirrored_target"
    _unchanged(world)
    assert all(row["outcome"] != "allowed" for row in world.audits)


def test_a_switch_that_raises_is_reported_and_audited(world, monkeypatch):
    async def raises(*_a, **_kw):
        raise RuntimeError("unblock failed")

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", raises)

    error = _refused(world, "codex")

    assert (error.status, error.code) == (500, "set_backend_failed")
    assert world.worker.backend == "claude"
    assert world.audits[-1]["detail"]["code"] == "set_backend_failed"


# ── Route ────────────────────────────────────────────────────────────────────


def _request(world, *, internal: bool, body: dict):
    request = MagicMock()
    request.app = {"state": world.state}
    request.path = "/api/session-control/set-backend"
    request.method = "POST"
    request.headers = {"X-Session-Key": slot_history_key(world.caller)}
    request.query = {}
    request.get = lambda key, default=None: (
        True if (key in ("internal_auth", "peer_verified") and internal) else default
    )

    async def _json():
        return body

    request.json = _json
    return request


def test_route_without_the_secret_is_forbidden(world):
    req = _request(world, internal=False, body={"target": "chat-2", "backend": "codex"})
    resp = asyncio.run(handlers_sc.api_session_control_set_backend(req))
    assert resp.status == 403
    _unchanged(world)


def test_route_switches_and_renders_a_refusal(world):
    req = _request(world, internal=True, body={"target": "chat-2", "backend": "nope"})
    resp = asyncio.run(handlers_sc.api_session_control_set_backend(req))
    assert (resp.status, json.loads(resp.body)["code"]) == (400, "unknown_backend")

    req = _request(world, internal=True, body={"target": "chat-2", "backend": "codex"})
    resp = asyncio.run(handlers_sc.api_session_control_set_backend(req))
    assert resp.status == 200, resp.body
    assert json.loads(resp.body)["backend"] == "codex"


# ── MCP tool ─────────────────────────────────────────────────────────────────

_ROUTE = "POST /api/session-control/set-backend"


def _tool(reply: dict) -> tuple[str, list]:
    dash = InMemoryDashboardClient({_ROUTE: reply})
    args = {"target": "chat-2", "backend": "codex"}
    out = TABLE.call(
        "session_set_backend", args, ToolContext(dash, Caller.strict("dashboard:chat-v"))
    )
    return out, dash.requests


def test_tool_carries_the_verified_key_and_reports_the_harness():
    out, (post,) = _tool({"ok": True, "target": "chat-2", "backend": "codex", "harness": "Codex"})

    assert post.path == "/api/session-control/set-backend"
    assert post.session_key == "dashboard:chat-v"
    assert post.body == {"target": "chat-2", "backend": "codex"}
    assert "`chat-2` now runs on Codex" in out


def test_tool_reports_a_refusal_with_its_code():
    out, _ = _tool(
        {
            "error": "backend 'nope' is not available; choose one of ['codex']",
            "code": "unknown_backend",
        }
    )

    assert out.startswith("Error:")
    assert "choose one of ['codex']" in out
    assert "`unknown_backend`" in out
