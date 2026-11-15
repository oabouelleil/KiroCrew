"""Moving a chat that stopped on a usage limit onto another harness (``limit_switch``).

Pinned here:

* one press (``switch_and_continue``) switches the harness and runs the continue
  Resume runs, once; the boundary row says the limit caused it and carries the
  reason; an armed "Continue automatically" wait is cancelled, and the chat no
  longer ends on the limit row, which is what releases the gateway's held wakes;
* refused with nothing changed: the harness the chat already runs (it hit the
  limit), a chat that does not end on a usage limit, the preview off;
* the standing choice is ``agent.acp_failover`` naming ``usage_limit``: a queue
  cycle that ends on a limit row moves the chat to the next chain member that can
  start and continues it; a block that names no triggers, or another trigger, moves
  nothing; the walk never lands on the harness it left and skips one that cannot
  start;
* no loop: when the harness it moved to also hits a limit before another message,
  the chat stays; a message sent since lets it move on down the chain;
* the HTTP route needs the owner and carries the body's ``backend`` to the core.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state
from dashboard_owner_helpers import as_owner

from kiro_crew import members
from kiro_crew.agent_sdk.backends import backend_from_stored
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.dashboard import chat_handlers, limit_continue, limit_switch
from kiro_crew.dashboard.chat_handlers import SlotAgentSwitchCaller
from kiro_crew.dashboard.system_notices import SESSION_SWITCH_KIND, SESSION_SWITCH_REASON_KEY
from kiro_crew.history_schema import HUMAN_TURN_META_KEY
from kiro_crew.providers.acp import AcpProvider

_SLOT = "chat-1"
_CALLER = SlotAgentSwitchCaller(app="", remote="", owner=True)
_LIMIT_ROW = {"role": "error", "content": "You've hit your limit", "meta": {"kind": "usage_limit"}}


def _limit_row() -> dict:
    return {**_LIMIT_ROW, "meta": dict(_LIMIT_ROW["meta"])}


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
        lambda b: BackendInstallState(b, b or "kiro", installs[b], install_command=""),
    )


@pytest.fixture(autouse=True)
def _no_waits_left_over():
    yield
    for key in list(limit_continue._ARMED):
        limit_continue.cancel(key)


@pytest.fixture
def failover(monkeypatch):
    """Set the ``agent.acp_failover`` block every config read returns."""

    def _set(block: dict) -> None:
        cfg = KiroCrewConfig()
        cfg.agent = dataclasses.replace(cfg.agent, acp_failover=block)
        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda _cls: cfg))

    _set({})
    return _set


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A chat on Claude Code whose last turn stopped on a usage limit."""
    state = _make_state(tmp_path)
    state.sessions.mirror_accepts_inbound = MagicMock(return_value=False)
    slot = state.get_or_create_slot(_SLOT)
    slot.append("user", "hello", "msg msg-u")
    slot.append("error", _LIMIT_ROW["content"], "msg msg-err", meta=dict(_LIMIT_ROW["meta"]))
    slot.backend = "claude"
    members.note_session_backend_pick(slot.key, "claude")
    live: dict = {"backend": "claude"}

    def provider(_key):
        if live["backend"] is None:
            return None
        p = MagicMock(spec=AcpProvider)
        p.capabilities = capabilities_for(live["backend"])
        p.has_active_turn.return_value = False
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
        return pick if isinstance(pick, str) else "kas"

    monkeypatch.setattr(chat_handlers, "_configured_backend_for_slot", configured)
    resume = AsyncMock(return_value="resumed")
    monkeypatch.setattr(chat_handlers, "continue_slot_turn", resume)
    return SimpleNamespace(state=state, slot=slot, live=live, resets=resets, resume=resume)


def _press(world, backend):
    response = asyncio.run(
        limit_switch.switch_and_continue(
            world.state, world.slot, backend, _CALLER, user_origin=True
        )
    )
    return response.status, json.loads(response.text)


def _typed(slot, text: str) -> None:
    """A message the person types: the send path marks the row as theirs."""
    slot.append("user", text, "msg msg-u", meta={HUMAN_TURN_META_KEY: True})


def _turn_ends(world) -> None:
    """A queue cycle ends: run the hook and every task it started."""

    async def _run() -> None:
        limit_switch.after_cycle_end(world.state, world.slot)
        while world.state._background_tasks:
            await asyncio.gather(*list(world.state._background_tasks))

    asyncio.run(_run())


def _switch_rows(slot) -> list[dict]:
    return [m for m in slot.messages if (m.get("meta") or {}).get("kind") == SESSION_SWITCH_KIND]


class TestOnePress:
    def test_it_switches_and_continues_once(self, world):
        status, body = _press(world, "codex")

        assert status == 200, body
        assert body["backend"] == "codex"
        assert body["continued"] == "resumed"
        assert world.slot.backend == "codex"
        assert world.resets == [f"dashboard:{_SLOT}"]
        world.resume.assert_awaited_once()
        assert world.resume.await_args.kwargs["via"] == limit_switch.LIMIT_SWITCH_VIA
        assert world.resume.await_args.kwargs["directive_user_origin"] is True
        [row] = _switch_rows(world.slot)
        assert row["content"].startswith("Usage limit reached, so the harness switched to")
        assert row["meta"][SESSION_SWITCH_REASON_KEY] == "usage_limit"

    def test_it_cancels_an_armed_wait_and_the_chat_no_longer_ends_on_the_limit(self, world):
        world.slot.messages[-1]["meta"]["reset_at"] = "2099-01-01T00:00:00+00:00"

        async def _arm_then_press():
            limit_continue.arm(
                world.state, world.slot, limit_continue.limit_reset_of(world.slot.messages)
            )
            return await limit_switch.switch_and_continue(
                world.state, world.slot, "codex", _CALLER, user_origin=True
            )

        assert asyncio.run(_arm_then_press()).status == 200
        assert world.slot.key not in limit_continue._ARMED
        # The chat has moved on from the limit, so nothing reads it as stopped there.
        assert limit_continue.limit_reset_of(world.slot.messages) is None
        assert not limit_continue.ends_on_usage_limit(world.slot.messages)

    def test_a_reply_that_lands_before_the_lock_is_not_continued(self, world, monkeypatch):
        # A turn that ended while the switch awaited its checks moved the chat past
        # the limit: the switch reads the transcript again under the lock.
        from kiro_crew.dashboard.handlers import messaging

        real = messaging._requested_backend

        async def requested_then_reply(raw, **kwargs):
            answer = await real(raw, **kwargs)
            world.slot.append("assistant", "done after all", "msg msg-a")
            return answer

        monkeypatch.setattr(messaging, "_requested_backend", requested_then_reply)

        status, body = _press(world, "codex")

        assert (status, body["code"]) == (409, "not_on_usage_limit")
        assert world.slot.backend == "claude"
        world.resume.assert_not_awaited()

    def test_the_harness_that_hit_the_limit_is_refused(self, world):
        status, body = _press(world, "claude")

        assert status == 409
        assert body["code"] == "limit_switch_same_harness"
        assert world.resets == []
        world.resume.assert_not_awaited()

    def test_a_chat_not_on_a_usage_limit_is_refused(self, world):
        _typed(world.slot, "never mind")

        status, body = _press(world, "codex")

        assert (status, body["code"]) == (409, "not_on_usage_limit")
        assert world.slot.backend == "claude"
        world.resume.assert_not_awaited()

    def test_the_preview_off_refuses(self, world, _floor_monkeypatch):
        from kiro_crew import session_harness_preview

        _floor_monkeypatch.setattr(
            session_harness_preview, "session_harness_enabled", lambda: False
        )

        status, _body = _press(world, "codex")

        assert status == 403
        assert world.slot.backend == "claude"
        world.resume.assert_not_awaited()

    def test_a_continue_that_cannot_queue_keeps_the_switch_and_says_so(self, world):
        world.resume.side_effect = chat_handlers.SlotContinueRefusal(
            "slot is running", "slot_running"
        )

        status, body = _press(world, "codex")

        assert status == 200
        assert world.slot.backend == "codex"
        assert body["continued"] == ""
        assert body["continue_refused"] == "slot_running"
        assert "did not continue" in body["warning"]


class TestStandingChoice:
    def test_a_named_usage_limit_trigger_moves_the_chat_down_the_chain(self, world, failover):
        failover({"enabled": True, "chain": ["claude", "codex"], "triggers": ["usage_limit"]})

        _turn_ends(world)

        assert world.slot.backend == "codex"
        world.resume.assert_awaited_once()
        # No person asked for this continue.
        assert world.resume.await_args.kwargs["directive_user_origin"] is False
        assert len(_switch_rows(world.slot)) == 1

    def test_an_app_chat_is_left_on_its_harness(self, world, failover):
        failover({"enabled": True, "chain": ["claude", "codex"], "triggers": ["usage_limit"]})
        world.slot._app = "some-app"

        _turn_ends(world)

        assert world.slot.backend == "claude"
        world.resume.assert_not_awaited()

    @pytest.mark.parametrize(
        "block",
        [
            {},
            {"enabled": False, "chain": ["claude", "codex"], "triggers": ["usage_limit"]},
            # Omitted triggers are the two spawn ones: a block written before the
            # chat consumer existed must not start moving chats.
            {"enabled": True, "chain": ["claude", "codex"]},
            {"enabled": True, "chain": ["claude", "codex"], "triggers": ["backend_not_installed"]},
            # The chat's harness is the chain's last member: nowhere to go.
            {"enabled": True, "chain": ["codex", "claude"], "triggers": ["usage_limit"]},
            # The chat's harness is not in the chain.
            {"enabled": True, "chain": ["codex", "kas"], "triggers": ["usage_limit"]},
        ],
        ids=[
            "absent",
            "disabled",
            "default-triggers",
            "other-trigger",
            "last-member",
            "not-in-chain",
        ],
    )
    def test_nothing_moves_without_the_trigger_and_a_member_after_it(self, world, failover, block):
        failover(block)

        _turn_ends(world)

        assert world.slot.backend == "claude"
        assert world.resets == []
        world.resume.assert_not_awaited()

    def test_a_member_that_cannot_start_is_skipped(self, world, failover):
        failover({"enabled": True, "chain": ["claude", "pi", "codex"], "triggers": ["usage_limit"]})

        _turn_ends(world)

        assert world.slot.backend == "codex"

    def test_a_chat_that_does_not_end_on_a_limit_is_left_alone(self, world, failover):
        failover({"enabled": True, "chain": ["claude", "codex"], "triggers": ["usage_limit"]})
        world.slot.append("assistant", "all good", "msg msg-a")

        _turn_ends(world)

        assert world.slot.backend == "claude"

    def test_the_preview_off_moves_nothing(self, world, failover, _floor_monkeypatch):
        from kiro_crew import session_harness_preview

        failover({"enabled": True, "chain": ["claude", "codex"], "triggers": ["usage_limit"]})
        _floor_monkeypatch.setattr(
            session_harness_preview, "session_harness_enabled", lambda: False
        )

        _turn_ends(world)

        assert world.slot.backend == "claude"
        world.resume.assert_not_awaited()


class TestNoLoop:
    def test_a_target_that_also_hits_a_limit_stays(self, world, failover):
        failover(
            {"enabled": True, "chain": ["claude", "codex", "kas"], "triggers": ["usage_limit"]}
        )
        _turn_ends(world)
        assert world.slot.backend == "codex"
        world.live["backend"] = "codex"
        # The continued turn on Codex stops on Codex's own limit.
        world.slot.messages.append(_limit_row())

        _turn_ends(world)

        assert world.slot.backend == "codex"
        assert len(_switch_rows(world.slot)) == 1
        world.resume.assert_awaited_once()

    def test_a_message_the_person_sends_since_lets_it_move_on(self, world, failover):
        failover(
            {"enabled": True, "chain": ["claude", "codex", "kas"], "triggers": ["usage_limit"]}
        )
        _turn_ends(world)
        world.live["backend"] = "codex"
        _typed(world.slot, "keep going")
        world.slot.messages.append(_limit_row())

        _turn_ends(world)

        assert world.slot.backend == "kas"
        assert len(_switch_rows(world.slot)) == 2

    def test_a_message_from_another_session_does_not(self, world, failover):
        # A conductor driving the chat sends unmarked user rows: they cannot walk
        # it down the chain with no person involved.
        failover(
            {"enabled": True, "chain": ["claude", "codex", "kas"], "triggers": ["usage_limit"]}
        )
        _turn_ends(world)
        world.live["backend"] = "codex"
        world.slot.append("user", "[from another session] keep going", "msg msg-u")
        world.slot.messages.append(_limit_row())

        _turn_ends(world)

        assert world.slot.backend == "codex"
        assert len(_switch_rows(world.slot)) == 1


class TestMovedSinceLastMessage:
    _SWITCH = {
        "role": "assistant",
        "content": "Usage limit reached, so the harness switched to Codex.",
        "meta": {"kind": SESSION_SWITCH_KIND, SESSION_SWITCH_REASON_KEY: "usage_limit"},
    }
    _TYPED = {"role": "user", "content": "hi", "meta": {HUMAN_TURN_META_KEY: True}}

    def test_a_limit_switch_after_the_newest_message(self):
        rows = [self._TYPED, self._SWITCH, {"role": "inject", "content": "go"}]
        assert limit_switch.moved_since_last_message(rows)

    def test_an_unmarked_user_row_is_not_the_person(self):
        rows = [self._SWITCH, {"role": "user", "content": "from a peer"}]
        assert limit_switch.moved_since_last_message(rows)

    @pytest.mark.parametrize(
        "rows",
        [
            [],
            [_TYPED],
            [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
            # A plain harness pick is not a limit switch.
            [{"role": "assistant", "content": "x", "meta": {"kind": SESSION_SWITCH_KIND}}],
        ],
    )
    def test_anything_else_is_not(self, rows):
        assert not limit_switch.moved_since_last_message(rows)

    def test_a_message_after_the_switch_starts_over(self):
        rows = [self._SWITCH, self._TYPED]
        assert not limit_switch.moved_since_last_message(rows)


@pytest.mark.parametrize(
    "drain, restored, called", [(True, False, True), (False, False, False), (True, True, False)]
)
def test_the_queue_cycle_end_hands_the_chat_to_the_switch(
    tmp_path, _floor_monkeypatch, drain, restored, called
):
    from kiro_crew.dashboard import chat_runner

    _floor_monkeypatch.setattr(chat_runner, "title_then_refresh", AsyncMock())
    _floor_monkeypatch.setattr(chat_runner, "generate_session_summary", AsyncMock())
    seen: list[str] = []
    _floor_monkeypatch.setattr(limit_switch, "after_cycle_end", lambda _st, s: seen.append(s.key))
    # A restored card about to run gets the floor first.
    _floor_monkeypatch.setattr(
        chat_runner, "_chosen_restored_head", lambda _s: object() if restored else None
    )
    _floor_monkeypatch.setattr(chat_runner, "_start_chosen_restored_after_release", AsyncMock())
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-cycle")
    slot._titled = True

    asyncio.run(chat_runner._finish_queue_cycle(state, slot, drain=drain))

    assert seen == (["chat-cycle"] if called else [])


def test_the_http_route_needs_the_owner_and_switches(world):
    def app_for(owner: bool) -> web.Application:
        app = web.Application()
        app["state"] = world.state
        app.router.add_post(
            "/api/chat/slots/{slot}/limit-switch", limit_switch.api_chat_slot_limit_switch
        )
        return as_owner(app) if owner else app

    async def post(owner: bool) -> tuple[int, dict]:
        async with TestClient(TestServer(app_for(owner))) as client:
            resp = await client.post(
                f"/api/chat/slots/{_SLOT}/limit-switch", json={"backend": "codex"}
            )
            return resp.status, await resp.json()

    status, _body = asyncio.run(post(False))
    assert status in (401, 403)
    assert world.slot.backend == "claude"

    status, body = asyncio.run(post(True))
    assert status == 200, body
    assert world.slot.backend == "codex"
    assert body["continued"] == "resumed"
