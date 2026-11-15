"""A usage-limit wait outlives a gateway restart.

``limit_continue`` records the armed waits in the data home and :func:`restore`
arms them again at boot. Pinned here:

* arming records the chat and its reset; cancelling, a queued continue and a closed
  chat take it off; a stopping gateway leaves it on;
* nothing is written before :func:`restore` has read the previous record;
* at boot a recorded chat that came back still ending on the same limit row is armed
  again (a reset already past continues once, soon), and anything else is dropped;
* an incognito chat is never recorded, and a damaged or oversized record arms nothing.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kiro_crew.dashboard import limit_continue
from kiro_crew.testing.clock import ManualClock
from kiro_crew.testing.wait import await_until

START = 1_800_000_000


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _limit_row(reset_epoch: float) -> dict:
    return {
        "role": "error",
        "content": "limit",
        "meta": {"kind": "usage_limit", "reset_at": _iso(reset_epoch)},
    }


class _Chat:
    def __init__(self, key: str, rows: list[dict], memory_mode: str = "persistent") -> None:
        self.key = key
        self.messages = rows
        self._turns_started = 0
        self.executor = ""
        self.memory_mode = memory_mode


def _state(*slots: _Chat) -> SimpleNamespace:
    return SimpleNamespace(_slots={s.key: s for s in slots}, push_slots_update=lambda: None)


@pytest.fixture(autouse=True)
def clock(_floor_monkeypatch):
    c = ManualClock(START)
    c.install(_floor_monkeypatch, limit_continue)
    _floor_monkeypatch.setattr(limit_continue, "_BOOK_TAKEN", False)
    yield c
    for key in list(limit_continue._ARMED):
        limit_continue.cancel(key, forget=False)
    limit_continue._STARTED.clear()


@pytest.fixture
def resume(_floor_monkeypatch) -> AsyncMock:
    go = AsyncMock(return_value="resumed")
    _floor_monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.continue_slot_turn", go)
    return go


def _book() -> dict:
    path = limit_continue._book_path()
    return json.loads(path.read_text())["waits"] if path.exists() else {}


def _write_book(waits: dict) -> None:
    path = limit_continue._book_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"waits": waits}))


async def _drain() -> None:
    while limit_continue._FIRING:
        await await_until(
            asyncio.gather(*list(limit_continue._FIRING)), what="a continue under way to finish"
        )


def _boot(*slots: _Chat) -> SimpleNamespace:
    """A fresh process: nothing armed, the record not yet read."""
    for key in list(limit_continue._ARMED):
        limit_continue.cancel(key, forget=False)
    limit_continue._BOOK_TAKEN = False
    state = _state(*slots)
    limit_continue.restore(state)
    return state


class TestTheRecord:
    @pytest.mark.asyncio
    async def test_nothing_is_written_before_the_previous_record_is_read(self):
        _write_book({"old": START + 60.0})
        slot = _Chat("s1", [_limit_row(START + 3600)])
        limit_continue.arm(_state(slot), slot, START + 3600)
        assert _book() == {"old": START + 60.0}

    @pytest.mark.asyncio
    async def test_arming_records_and_cancelling_forgets(self):
        _boot()
        slot = _Chat("s1", [_limit_row(START + 3600)])
        limit_continue.arm(_state(slot), slot, START + 3600)
        assert _book() == {"s1": START + 3600}
        assert limit_continue.cancel("s1") is True
        assert _book() == {}

    @pytest.mark.asyncio
    async def test_a_stopping_gateway_keeps_the_record(self):
        _boot()
        slot = _Chat("s1", [_limit_row(START + 3600)])
        limit_continue.arm(_state(slot), slot, START + 3600)
        await limit_continue.cancel_all()
        assert "s1" not in limit_continue._ARMED
        assert _book() == {"s1": START + 3600}

    @pytest.mark.asyncio
    async def test_an_incognito_chat_is_not_recorded(self):
        _boot()
        slot = _Chat("s1", [_limit_row(START + 3600)], memory_mode="incognito")
        limit_continue.arm(_state(slot), slot, START + 3600)
        assert _book() == {}

    @pytest.mark.asyncio
    async def test_a_continue_that_ran_is_off_the_record(self, clock, resume):
        def _queue(*_a, before_dispatch, **_k):
            before_dispatch()
            assert _book() == {}, "the record still held the wait while its turn ran"
            return "resumed"

        resume.side_effect = _queue
        state = _boot()
        slot = _Chat("s1", [_limit_row(START + 3600)])
        state._slots["s1"] = slot
        armed = limit_continue.arm(state, slot, START + 3600)
        clock.advance(3600 + limit_continue.RESUME_MARGIN_SECS)
        limit_continue._wake(state, armed)
        await _drain()
        resume.assert_awaited_once()
        assert _book() == {}

    @pytest.mark.asyncio
    async def test_a_closed_chat_is_off_the_record(self, clock):
        state = _boot()
        slot = _Chat("s1", [_limit_row(START + 3600)])
        state._slots["s1"] = slot
        armed = limit_continue.arm(state, slot, START + 3600)
        del state._slots["s1"]
        clock.advance(3660)
        limit_continue._wake(state, armed)
        assert _book() == {}


class TestRestore:
    @pytest.mark.asyncio
    async def test_a_chat_still_at_the_limit_is_armed_again(self):
        _write_book({"s1": START + 3600.0})
        slot = _Chat("s1", [_limit_row(START + 3600)])
        state = _boot(slot)
        armed = limit_continue._ARMED["s1"]
        assert armed.reset_at == START + 3600
        assert limit_continue.armed_resume_iso(slot) == _iso(
            START + 3600 + limit_continue.RESUME_MARGIN_SECS
        )
        assert state._slots["s1"] is slot

    @pytest.mark.asyncio
    async def test_a_reset_that_passed_while_down_continues_once_soon(self, clock, resume):
        _write_book({"s1": START - 600.0})
        slot = _Chat("s1", [_limit_row(START - 600)])
        state = _boot(slot)
        armed = limit_continue._ARMED["s1"]
        limit_continue._wake(state, armed)
        limit_continue._wake(state, armed)
        await _drain()
        resume.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_it_does_not_fire_before_the_reset(self, clock, resume):
        _write_book({"s1": START + 3600.0})
        slot = _Chat("s1", [_limit_row(START + 3600)])
        state = _boot(slot)
        clock.advance(3600)
        limit_continue._wake(state, limit_continue._ARMED["s1"])
        await _drain()
        resume.assert_not_awaited()

    @pytest.mark.parametrize(
        "rows",
        [
            [_limit_row(START + 3600), {"role": "user", "content": "I did it myself"}],
            [_limit_row(START + 7200)],
            [{"role": "assistant", "content": "done"}],
        ],
    )
    @pytest.mark.asyncio
    async def test_a_chat_that_moved_on_is_dropped(self, rows):
        _write_book({"s1": START + 3600.0})
        _boot(_Chat("s1", rows))
        assert limit_continue._ARMED == {}
        assert _book() == {}

    @pytest.mark.asyncio
    async def test_a_chat_that_did_not_come_back_is_dropped(self):
        _write_book({"gone": START + 3600.0})
        _boot()
        assert limit_continue._ARMED == {}
        assert _book() == {}

    @pytest.mark.parametrize(
        "body",
        [
            "not json",
            json.dumps(["s1"]),
            json.dumps({"waits": {"s1": "soon"}}),
            json.dumps({"waits": {"s1": True}}),
            json.dumps({"waits": {"s1": START + 3600.0, "pad": "x" * 70_000}}),
        ],
        ids=["not-json", "a-list", "a-string-time", "a-bool-time", "oversized"],
    )
    @pytest.mark.asyncio
    async def test_a_damaged_record_arms_nothing(self, body):
        path = limit_continue._book_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        _boot(_Chat("s1", [_limit_row(START + 3600)]))
        assert limit_continue._ARMED == {}
