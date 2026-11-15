"""A chat waiting for its usage limit to reset holds the wakes the gateway sends it.

An auto-nudge cycle, a monitor wake or a work-ledger wake that started a turn while
the chat waited would be refused by the same limit, and its new limit row would end
the person's "Continue automatically" wait (``limit_continue``). Pinned here:

* with a wait armed, a loop's cycle is held: no turn, no row, no cycle spent, and
  the wait stays armed; several cycles in one wait are all held;
* a loop armed after the wait is held the same way, from its first cycle;
* once the continue has queued its turn, or the chat moved on, the loop fires once;
* a message the person types still ends the wait, and Cancel still drops it;
* with no wait armed, a chat that ends on a usage-limit row whose reset is ahead
  holds the wake too, and releases it once the reset passes; a limit row naming no
  reset holds nothing;
* only the gateway's automation wakes are held: a cron or app message is not.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_autonudge_dashboard_fire import _fake_spawn, _loop, _orchestrator
from test_autonudge_dashboard_fire import _slot as nudge_slot

from kiro_crew.dashboard import limit_continue, session_deliver
from kiro_crew.slack import gateway as gw
from kiro_crew.testing.clock import ManualClock

START = 1_800_000_000
KEY = "chat-1-1785"


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _limit_row(reset_epoch: float | None) -> dict:
    meta: dict = {"kind": "usage_limit"}
    if reset_epoch is not None:
        meta["reset_at"] = _iso(reset_epoch)
    return {"role": "error", "content": "You've hit your 5-hour limit", "meta": meta}


@pytest.fixture(autouse=True)
def clock(_floor_monkeypatch):
    c = ManualClock(START)
    c.install(_floor_monkeypatch, limit_continue)
    yield c
    for key in list(limit_continue._ARMED):
        limit_continue.cancel(key)
    limit_continue._STARTED.clear()


def _limited_slot(reset_epoch: float | None = START + 3600) -> MagicMock:
    slot = nudge_slot(KEY)
    slot.messages = [{"role": "user", "content": "build it"}, _limit_row(reset_epoch)]
    slot._turns_started = 1
    return slot


async def _fire(slot: MagicMock, loop=None) -> tuple[object, list]:
    """One nudge cycle into *slot*: the fire's answer and the turns it spawned."""
    orch = _orchestrator()
    orch.dashboard_state.get_slot = MagicMock(return_value=slot)
    orch._dashboard_mode_admits = AsyncMock(return_value=True)
    spawn = _fake_spawn()
    with (
        patch.object(gw, "spawn_guarded_turn", spawn),
        patch("kiro_crew.dashboard.chat._run_chat", new=AsyncMock()),
    ):
        result = await orch._fire_dashboard_nudge(loop or _loop(KEY))
    return result, spawn.calls


def _arm(slot: MagicMock) -> limit_continue._Armed:
    state = SimpleNamespace(_slots={slot.key: slot}, push_slots_update=lambda: None)
    return limit_continue.arm(state, slot, START + 3600)


class TestAnArmedWait:
    @pytest.mark.asyncio
    async def test_a_nudge_turn_does_not_end_the_wait(self):
        """The reported bug: a cycle's turn hits the limit again and its row ends the wait."""
        slot = _limited_slot()
        _arm(slot)
        orch = _orchestrator()
        orch.dashboard_state.get_slot = MagicMock(return_value=slot)
        orch._dashboard_mode_admits = AsyncMock(return_value=True)

        def _turn_hits_the_limit_again(state, s, coro, **_k):
            coro.close()
            s._turns_started += 1
            s.messages.append({"role": "nudge", "content": "[auto-nudge cycle 4]"})
            s.messages.append(_limit_row(START + 3600))
            return MagicMock(name="turn-task")

        with (
            patch.object(gw, "spawn_guarded_turn", _turn_hits_the_limit_again),
            patch("kiro_crew.dashboard.chat._run_chat", new=AsyncMock()),
        ):
            await orch._fire_dashboard_nudge(_loop(KEY))
        assert limit_continue.armed_resume_iso(slot) is not None, "the nudge ended the wait"
        assert len(slot.messages) == 2

    @pytest.mark.asyncio
    async def test_a_nudge_cycle_is_held_and_the_wait_survives(self):
        slot = _limited_slot()
        _arm(slot)
        result, spawned = await _fire(slot)
        assert result is False  # BUSY on a message loop: no cycle spent
        assert spawned == []
        slot.append.assert_not_called()  # no nudge row, so the chat still ends on the limit
        assert limit_continue.armed_resume_iso(slot) is not None

    @pytest.mark.asyncio
    async def test_several_cycles_in_one_wait_are_all_held(self, clock):
        slot = _limited_slot()
        _arm(slot)
        for _ in range(4):
            clock.advance(600)
            result, spawned = await _fire(slot)
            assert (result, spawned) == (False, [])
        assert limit_continue.armed_resume_iso(slot) is not None

    @pytest.mark.asyncio
    async def test_a_loop_armed_after_the_wait_is_held_from_its_first_cycle(self):
        slot = _limited_slot()
        _arm(slot)
        fresh = _loop(KEY)
        fresh.id, fresh.cycle_count = "loop-new", 0
        result, spawned = await _fire(slot, fresh)
        assert (result, spawned) == (False, [])
        assert limit_continue.armed_resume_iso(slot) is not None

    @pytest.mark.asyncio
    async def test_the_margin_after_the_reset_is_still_held(self, clock):
        slot = _limited_slot()
        _arm(slot)
        clock.advance(3600 + limit_continue.RESUME_MARGIN_SECS - 1)
        assert (await _fire(slot))[1] == []

    @pytest.mark.asyncio
    async def test_once_the_continue_is_queued_the_loop_is_not_held(self, clock):
        slot = _limited_slot()
        armed = _arm(slot)
        clock.advance(3600 + limit_continue.RESUME_MARGIN_SECS)
        assert limit_continue.holds_automation(slot) is True
        armed.queued = True
        assert limit_continue.holds_automation(slot) is False

    @pytest.mark.asyncio
    async def test_after_the_continue_the_held_loop_fires_once(self, clock):
        slot = _limited_slot()
        _arm(slot)
        assert (await _fire(slot))[1] == []
        # The continue ran: its turn answered, so the chat no longer ends on the limit.
        clock.advance(3600 + limit_continue.RESUME_MARGIN_SECS)
        limit_continue.cancel(KEY)
        slot._turns_started += 1
        slot.messages.append({"role": "assistant", "content": "carried on"})
        result, spawned = await _fire(slot)
        assert result is True
        assert spawned == [slot]

    @pytest.mark.asyncio
    async def test_a_typed_message_still_ends_the_wait(self):
        slot = _limited_slot()
        _arm(slot)
        slot.messages.append({"role": "user", "content": "I'll take it from here"})
        assert limit_continue.armed_resume_iso(slot) is None
        assert KEY not in limit_continue._ARMED
        assert limit_continue.holds_automation(slot) is False


class TestNoWaitArmed:
    @pytest.mark.asyncio
    async def test_a_reset_still_ahead_holds_the_wake(self):
        slot = _limited_slot()
        assert (await _fire(slot))[1] == []

    @pytest.mark.asyncio
    async def test_a_cancelled_wait_still_holds_until_the_reset(self, clock):
        slot = _limited_slot()
        _arm(slot)
        assert limit_continue.cancel(KEY) is True
        assert (await _fire(slot))[1] == []
        clock.advance(3601)
        assert (await _fire(slot))[1] == [slot]

    @pytest.mark.asyncio
    async def test_a_reset_that_passed_releases_the_wake(self, clock):
        slot = _limited_slot()
        clock.advance(3601)
        assert (await _fire(slot))[1] == [slot]

    @pytest.mark.asyncio
    async def test_a_limit_row_with_no_reset_holds_nothing(self):
        slot = _limited_slot(reset_epoch=None)
        assert (await _fire(slot))[1] == [slot]

    @pytest.mark.parametrize(
        "rows",
        [
            [],
            [{"role": "assistant", "content": "done"}],
            [{"role": "error", "content": "boom", "meta": {"kind": "auth_required"}}],
        ],
        ids=["empty", "answered", "another-error"],
    )
    def test_a_chat_not_ending_on_a_limit_holds_nothing(self, rows):
        slot = nudge_slot(KEY)
        slot.messages = rows
        assert limit_continue.holds_automation(slot) is False

    def test_a_slot_with_no_message_list_holds_nothing(self):
        assert limit_continue.holds_automation(nudge_slot(KEY)) is False


class TestOnlyAutomationIsHeld:
    @pytest.mark.asyncio
    async def test_deliver_holds_an_automation_wake_without_running_its_starter(self):
        slot = _limited_slot()
        start = AsyncMock()
        outcome = await session_deliver.deliver(
            MagicMock(), slot, "x", session_deliver.automation(1, "loop"), start_turn=start
        )
        assert outcome == session_deliver.HELD
        start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_cron_message_is_not_held(self, _floor_monkeypatch):
        slot = _limited_slot()
        idle = AsyncMock(return_value=session_deliver.DELIVERED_TURN)
        _floor_monkeypatch.setattr(session_deliver, "_deliver_idle", idle)
        outcome = await session_deliver.deliver(
            MagicMock(), slot, "report", session_deliver.cron("nightly")
        )
        assert outcome == session_deliver.DELIVERED_TURN
        idle.assert_awaited_once()
