"""Switching an existing chat's harness (``POST /api/chat/slots/{slot}/backend``).

Pinned here: an idle chat moves onto the named harness (the pick is re-pointed, the
session is reset, the transcript gets a boundary row) and can move back; a chat
already on the harness keeps its session; a turn in flight, attached sub-agents, a
harness that cannot start, a channel- or Bot-routed session and the preview being off
are refused with nothing changed; a reset a slipped-in turn declines is rolled back;
and the new pick survives a save and restore.
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
from kiro_crew.dashboard.chat_handlers import SlotAgentSwitchCaller, switch_slot_harness
from kiro_crew.providers.acp import AcpProvider

_SLOT = "chat-1"
_KEY = f"dashboard:{_SLOT}"
_CALLER = SlotAgentSwitchCaller(app="", remote="", owner=True)


@pytest.fixture(autouse=True)
def _preview_on(_floor_monkeypatch):
    from kiro_crew import session_harness_preview

    _floor_monkeypatch.setattr(session_harness_preview, "session_harness_enabled", lambda: True)


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    """Kiro, KAS, Claude and Codex can start here; Pi is allowed but missing."""
    from kiro_crew import acp_backends
    from kiro_crew.agent_sdk import backend_install
    from kiro_crew.agent_sdk.backend_install import INSTALLED, MISSING, BackendInstallState

    installs = {"": INSTALLED, "kas": INSTALLED, "claude": INSTALLED, "codex": INSTALLED}
    installs["pi"] = MISSING
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
    """A started chat whose live session runs Claude Code, with the reset recorded."""
    state = _make_state(tmp_path)
    state.sessions.mirror_accepts_inbound = MagicMock(return_value=False)
    slot = state.get_or_create_slot(_SLOT)
    slot.append("user", "hello", "msg msg-u")
    slot.backend = "claude"
    members.note_session_backend_pick(slot.key, "claude")
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
        assert switch_kind == "backend"
        resets.append(key)
        # The session the reset tears down is gone; the next turn cold-starts.
        live["backend"] = None
        return True

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", reset)
    monkeypatch.setattr(chat_handlers, "_subagents_attached_response", AsyncMock(return_value=None))

    async def configured(s, config=None):
        pick = backend_from_stored(s.backend)
        return pick if isinstance(pick, str) else "kas"

    monkeypatch.setattr(chat_handlers, "_configured_backend_for_slot", configured)
    return SimpleNamespace(state=state, slot=slot, live=live, resets=resets)


def _switch(world, backend):
    response = asyncio.run(switch_slot_harness(world.state, world.slot, _SLOT, backend, _CALLER))
    return response.status, json.loads(response.text)


def _notices(slot):
    return [m["content"] for m in slot.messages if (m.get("meta") or {}).get("kind")]


def test_an_idle_chat_moves_to_the_named_harness(world):
    status, body = _switch(world, "codex")

    assert status == 200, body
    assert body["backend"] == "codex"
    assert world.slot.backend == "codex"
    assert members.session_backend_pick(_KEY) == "codex"
    assert world.resets == [_KEY]
    assert world.slot._dirty is True
    assert any(n.startswith("Harness switched to") for n in _notices(world.slot))
    # The next turn's allocation asks the selection gate for the new harness.
    from kiro_crew.dashboard.chat_runner import _slot_backend_allocation_kwargs

    assert _slot_backend_allocation_kwargs(world.slot) == {"backend_override": "codex"}


def test_the_chat_can_move_back(world):
    assert _switch(world, "codex")[0] == 200
    world.live["backend"] = "codex"  # the next turn started on Codex

    status, body = _switch(world, "claude")

    assert status == 200, body
    assert world.slot.backend == "claude"
    assert members.session_backend_pick(_KEY) == "claude"
    assert world.resets == [_KEY, _KEY]


def test_kiro_is_stored_by_name(world):
    status, _body = _switch(world, "kiro")

    assert status == 200
    assert world.slot.backend == "kiro"
    assert members.session_backend_pick(_KEY) == ""


def test_the_harness_it_already_runs_keeps_its_session(world):
    status, body = _switch(world, "claude")

    assert status == 200, body
    assert world.resets == []
    assert not any(n.startswith("Harness switched to") for n in _notices(world.slot))


def test_a_turn_in_flight_is_refused_with_nothing_changed(world):
    world.live["busy"] = True

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "turn_in_flight")
    assert world.slot.backend == "claude"
    assert members.session_backend_pick(_KEY) == "claude"
    assert world.resets == []


def test_attached_sub_agents_are_refused(world, monkeypatch):
    from aiohttp import web

    refusal = web.json_response({"error": "x", "code": "slot_subagents_running"}, status=409)
    monkeypatch.setattr(
        chat_handlers, "_subagents_attached_response", AsyncMock(return_value=refusal)
    )

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "slot_subagents_running")
    assert world.slot.backend == "claude"
    assert world.resets == []


@pytest.mark.parametrize(
    "name, code",
    [("pi", "backend_not_installed"), ("nope", "unknown_backend")],
)
def test_a_harness_that_cannot_start_is_refused(world, name, code):
    status, body = _switch(world, name)

    assert (status, body["code"]) == (400, code)
    # The refusal lists what can start, so the picker can offer it.
    assert "codex" in body["backends"]
    assert world.slot.backend == "claude"
    assert world.resets == []


@pytest.mark.parametrize("raw", ["", None, 3])
def test_a_missing_name_is_refused(world, raw):
    status, body = _switch(world, raw)

    assert (status, body["code"]) == (400, "invalid_backend")


def test_the_preview_off_refuses(world, _floor_monkeypatch):
    from kiro_crew import session_harness_preview

    _floor_monkeypatch.setattr(session_harness_preview, "session_harness_enabled", lambda: False)

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (403, "session_harness_preview_off")
    assert world.slot.backend == "claude"


@pytest.mark.parametrize(
    "patch",
    ["is_member_session_key", "is_crewmate_worker_session"],
)
def test_a_bot_routed_session_is_refused(world, monkeypatch, patch):
    monkeypatch.setattr(members, patch, lambda _key: True)

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "harness_switch_unsupported")
    assert world.resets == []


def test_a_channel_driven_session_is_refused(world):
    world.state.sessions.mirror_accepts_inbound = MagicMock(return_value=True)

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "harness_switch_unsupported")


def test_a_declined_reset_under_a_new_turn_rolls_back(world, monkeypatch):
    async def declined(_state, _slot, _key, *, switch_kind):
        world.live["busy"] = True  # a send slipped in on the old session
        return False

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", declined)

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "turn_in_flight")
    assert world.slot.backend == "claude"
    assert members.session_backend_pick(_KEY) == "claude"


def test_a_rollback_without_an_earlier_pick_restores_the_running_harness(world, monkeypatch):
    world.slot.backend = ""
    members.note_session_backend_pick(world.slot.key, "")

    async def declined(_state, _slot, _key, *, switch_kind):
        world.live["busy"] = True
        return False

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", declined)

    assert _switch(world, "codex")[0] == 409
    # An explicit pick of what ran, which a merge save writes over the new one.
    assert world.slot.backend == "claude"


def test_the_new_pick_survives_a_save_and_restore(world, tmp_path, monkeypatch):
    from kiro_crew.dashboard.chat_persistence import _save_slot_to_history
    from kiro_crew.dashboard.slot_persistence import metadata_codec as codec

    monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    assert _switch(world, "codex")[0] == 200
    _save_slot_to_history(world.state, world.slot, force=True)
    meta = world.state.conversation_log._read_metadata(_KEY)
    assert meta.get("backend") == "codex"

    members.note_session_backend_pick(world.slot.key, "")
    fresh = _make_state(tmp_path / "other").get_or_create_slot(_SLOT)
    codec._read_backend(SimpleNamespace(meta=meta, slot=fresh))
    assert fresh.backend == "codex"
    assert members.session_backend_pick(_KEY) == "codex"


@pytest.mark.parametrize("link", ["cron:nightly", "workflow:run-1"])
def test_a_linked_tab_is_refused(world, link):
    """A cron or workflow run drives that session on its own harness, not the pick."""
    world.slot.linked_session_key = link

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "harness_switch_unsupported")
    assert world.slot.backend == "claude"
    assert world.resets == []


def test_a_harness_that_cannot_be_read_is_refused(world, monkeypatch):
    world.live["backend"] = None

    async def unreadable(_slot, config=None):
        from kiro_crew.crewmate_harness import CrewmateHarnessUnavailable

        raise CrewmateHarnessUnavailable("bot", "pi")

    monkeypatch.setattr(chat_handlers, "_configured_backend_for_slot", unreadable)

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "harness_switch_unsupported")
    assert world.slot.backend == "claude"


def test_a_reset_that_raises_before_the_pop_rolls_back(world, monkeypatch):
    async def raises(_state, _slot, _key, *, switch_kind):
        raise RuntimeError("unblock failed")

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", raises)

    with pytest.raises(RuntimeError):
        _switch(world, "codex")

    assert world.slot.backend == "claude"
    assert members.session_backend_pick(_KEY) == "claude"


def test_an_idle_session_that_declines_is_retried(world, monkeypatch):
    calls: list[str] = []

    async def declines_once(_state, _slot, key, *, switch_kind):
        calls.append(key)
        if len(calls) == 1:
            return False
        world.live["backend"] = None
        return True

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", declines_once)

    status, _body = _switch(world, "codex")

    assert status == 200
    assert calls == [_KEY, _KEY]
    assert world.slot.backend == "codex"


def test_a_teardown_that_raises_after_the_pop_keeps_the_switch_with_a_warning(world, monkeypatch):
    async def degraded(_state, _slot, _key, *, switch_kind):
        world.live["backend"] = None
        return None

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", degraded)

    status, body = _switch(world, "codex")

    assert status == 200
    assert body["warning"]
    assert world.slot.backend == "codex"


def test_a_rebind_during_the_reset_rolls_back(world, monkeypatch):
    async def rebinds(_state, slot, _key, *, switch_kind):
        slot.linked_session_key = "cron:late"
        return True

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", rebinds)

    status, body = _switch(world, "codex")

    assert (status, body["code"]) == (409, "session_rebound")
    assert world.slot.backend == "claude"


@pytest.mark.parametrize("target", ["codex", "claude"])
def test_a_caller_whose_gate_closed_is_refused_at_the_commit(world, target):
    """``still_permitted`` is read right before each commit, the switch and the
    already-there pick alike. Mutation guard: skipping it commits for a caller
    that may not touch the chat."""
    world.slot.backend = ""
    caller = _CALLER._replace(still_permitted=lambda: False)

    response = asyncio.run(switch_slot_harness(world.state, world.slot, _SLOT, target, caller))

    assert (response.status, json.loads(response.text)["code"]) == (403, "caller_not_permitted")
    assert world.slot.backend == ""
    assert world.resets == []


def test_a_caller_whose_gate_closes_during_a_declined_reset_is_not_retried(world, monkeypatch):
    """The retry is a second teardown after an await, so the gate is read again
    first. Mutation guard: retrying without it resets a session the caller lost."""
    permitted = {"ok": True}
    calls: list[str] = []

    async def declines(_state, _slot, key, *, switch_kind):
        calls.append(key)
        permitted["ok"] = False  # e.g. the target was mirrored meanwhile
        return False

    monkeypatch.setattr(chat_handlers, "_reset_slot_session_or_warn", declines)
    caller = _CALLER._replace(still_permitted=lambda: permitted["ok"])

    response = asyncio.run(switch_slot_harness(world.state, world.slot, _SLOT, "codex", caller))

    assert (response.status, json.loads(response.text)["code"]) == (403, "caller_not_permitted")
    assert calls == [_KEY]
    assert world.slot.backend == "claude"
    assert members.session_backend_pick(_KEY) == "claude"


def test_the_http_route_switches_and_refuses_another_apps_slot(world):
    """Through the route: the body's ``backend`` reaches the core, and an app token
    on a slot it does not own gets the uniform 404 with nothing changed."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from kiro_crew.dashboard.chat import api_chat_slot_backend

    def app_for(app_claim: str) -> web.Application:
        @web.middleware
        async def marker(request, handler):
            request["app"] = app_claim
            return await handler(request)

        app = web.Application(middlewares=[marker])
        app["state"] = world.state
        app.router.add_post("/api/chat/slots/{slot}/backend", api_chat_slot_backend)
        return app

    async def post(app_claim: str, body: dict) -> tuple[int, dict]:
        async with TestClient(TestServer(app_for(app_claim))) as client:
            resp = await client.post(f"/api/chat/slots/{_SLOT}/backend", json=body)
            return resp.status, await resp.json()

    status, _body = asyncio.run(post("other-app", {"backend": "codex"}))
    assert status == 404
    assert world.slot.backend == "claude"

    status, body = asyncio.run(post("", {"backend": "codex"}))
    assert status == 200, body
    assert world.slot.backend == "codex"
