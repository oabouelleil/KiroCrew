"""Continue a chat once, after the usage limit it stopped on resets.

A turn that stops on a spent session, 5-hour or weekly allowance leaves an
``error`` row of kind ``usage_limit``. When the provider named when the limit
opens again, the row's meta carries ``reset_at`` (an ISO instant) and the error
card offers "Continue automatically after <time>".

``POST /api/chat/slots/{slot}/limit-continue`` arms one wait for the chat and
``DELETE`` on the same path cancels it. A second POST replaces the first. When the
wait ends the gateway runs the same Continue the card's Resume button runs
(:func:`continue_slot_turn`), and only if the chat is still the one armed, has
started no turn since, still ends on that same row and was not cancelled, checked
again under the slot's lock just before the continue is queued. Cancel keeps that
authority until then; once the turn is queued, ``DELETE`` answers ``started``, and
keeps answering it for a few minutes (or until a turn after the continued one starts)
from a small record apart from the wait, so a Cancel that reaches the gateway after the
wait ended still learns the continue went ahead. A message typed meanwhile, a
closed chat, a turn already running, a conductor mode "Pause all" holding the
chat, a relay archive or a gateway shutting down makes it do nothing. Nothing is
armed unless a person presses the button.

The wait is a timer in the running gateway, and a small record of the armed waits
(:data:`BOOK_NAME` in the data home) lets it outlive the process: a restart, a crash
or an applied update stops the timer, and :func:`restore` arms it again at boot,
after the open-tab restore, for each recorded chat that came back and still ends on
the same limit row. A reset that passed while the gateway was down continues soon
after boot. Nothing writes the record before :func:`restore` has read the previous
one, a stopping gateway leaves it as it is, and an incognito or temporary chat,
which a restart does not bring back, is never recorded. The record grants nothing a
chat's own state does not already allow: an entry acts only on a chat that still
ends on that exact limit row, and does what one press of Resume does, once. The wait
holds the chat only weakly, so a closed chat is not kept alive by it. While a wait is
armed the chat's slot summary carries ``limit_continue_at`` (the ISO instant it will
resume), which is how a reloaded card learns it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aiohttp import web

from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.paths import data_home
from kiro_crew.dashboard.chat_utils import USAGE_LIMIT_KIND
from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request
from kiro_crew.dashboard.slot_ownership import slot_not_found

if TYPE_CHECKING:
    from kiro_crew.dashboard.state import DashboardState

logger = logging.getLogger(__name__)

#: Seconds past the named reset the chat waits before it continues, so a turn that
#: starts the instant the clause names is not refused again by a limit that has not
#: quite opened.
RESUME_MARGIN_SECS = 60

#: The longest the wait's timer runs before it reads the wall clock again. The loop's
#: timer does not count time the host spent suspended, so one long timer can end hours
#: after the reset; slices this long end within one slice of it.
WALL_RECHECK_SECS = 60

#: Two readings of the reset closer than this are the same instant. A browser keeps
#: only milliseconds, and a relative clause ("resets in 2 hours") gives the row
#: microseconds, so an exact comparison would refuse the row's own time.
_SAME_INSTANT_SECS = 0.001

#: How long a continue that queued its turn is remembered as started, so a Cancel that
#: reaches the gateway after the wait ended is still told so.
STARTED_TTL_SECS = 300

#: The most chats whose started continue is remembered at once; the oldest goes first.
STARTED_CAP = 64

#: The record of armed waits, under the data home, so a restart can arm them again.
BOOK_NAME = "limit_continue_waits.json"

#: One key and one instant per waiting chat is far below this. A larger file is not
#: one this module wrote, so it is read as empty rather than in full.
_BOOK_MAX_BYTES = 64 * 1024


@dataclass
class _Armed:
    """One chat's pending continue: its timer, the row it was armed on, and when it ends."""

    key: str
    slot: weakref.ref[Any]
    reset_at: float
    #: The chat's turn count when armed: any turn started since ends the wait.
    turns: int
    resume_at: float
    loop: asyncio.AbstractEventLoop
    handle: asyncio.TimerHandle | None = None
    #: The timer ran out and the continue is under way; a second wake does nothing.
    firing: bool = False
    #: Cancel was pressed (or a later wait replaced this one): the continue must not queue.
    cancelled: bool = False
    #: The continue passed its last check and queued the turn: Cancel cannot stop it now.
    queued: bool = False

    def holds(self, slot: Any) -> bool:
        """Whether *slot* is the chat armed, with no turn since and the same limit row last."""
        return (
            slot is not None
            and self.slot() is slot
            and getattr(slot, "_turns_started", 0) == self.turns
            and limit_reset_of(slot.messages) == self.reset_at
        )


@dataclass
class _Started:
    """A continue that queued its turn: the chat, the last turn count it may show, and its end."""

    slot: weakref.ref[Any]
    turns: int
    until: float


_ARMED: dict[str, _Armed] = {}
#: Continues that queued their turn, by chat, kept apart from the wait that ended.
_STARTED: dict[str, _Started] = {}
# Fire tasks held until done, so a collected task cannot drop a continue.
_FIRING: set[asyncio.Task[None]] = set()
#: Set once :func:`restore` has read the previous process's record; until then no
#: write may replace it.
_BOOK_TAKEN = False


def _book_path() -> Any:
    return data_home() / BOOK_NAME


def _read_book() -> dict[str, float]:
    """The recorded waits, chat key to reset instant; ``{}`` when absent or damaged."""
    path = _book_path()
    try:
        if path.is_symlink():
            return {}
        raw = path.read_bytes()
    except OSError:
        return {}
    if len(raw) > _BOOK_MAX_BYTES:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    waits = data.get("waits") if isinstance(data, dict) else None
    if not isinstance(waits, dict):
        return {}
    return {
        key: float(at)
        for key, at in waits.items()
        if isinstance(key, str) and isinstance(at, (int, float)) and not isinstance(at, bool)
    }


def _save() -> None:
    """Record the armed waits a restart should bring back. Never raises.

    A wait whose turn is queued is done, and a chat a restart does not bring back
    (incognito, temporary) has nothing to come back to, so neither is recorded.
    """
    if not _BOOK_TAKEN:
        return
    waits: dict[str, float] = {}
    for key, armed in _ARMED.items():
        slot = armed.slot()
        if armed.queued or slot is None:
            continue
        if getattr(slot, "memory_mode", "persistent") != "persistent":
            continue
        waits[key] = armed.reset_at
    try:
        path = _book_path()
        if waits:
            atomic_write(path, json.dumps({"waits": waits}, sort_keys=True))
        else:
            path.unlink(missing_ok=True)
    except OSError:
        logger.warning("limit-continue: the waits could not be recorded", exc_info=True)


def restore(state: DashboardState) -> int:
    """Arm again the waits the previous process recorded; how many were armed.

    Run once at boot, on the loop, after the open-tab restore. A recorded chat that
    did not come back, or that ends on anything but a limit row naming that same instant, is
    dropped. A reset already past continues a margin after it, which is soon.
    """
    global _BOOK_TAKEN
    recorded = _read_book()
    _BOOK_TAKEN = True
    count = 0
    for key, reset_at in recorded.items():
        slot = state._slots.get(key)
        if slot is None:
            continue
        row_at = limit_reset_of(slot.messages)
        if row_at is None or abs(row_at - reset_at) >= _SAME_INSTANT_SECS:
            continue
        try:
            arm(state, slot, row_at)
        except Exception:  # one chat's wait must never stop the boot
            logger.warning("limit-continue: a wait could not be armed again", exc_info=True)
            continue
        count += 1
    _save()
    if count:
        logger.info("limit-continue: armed %d wait(s) again after the restart", count)
        state.push_slots_update()
    return count


def _instant(raw: object) -> float | None:
    """Epoch seconds of an ISO instant with a zone; ``None`` for anything else."""
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return moment.timestamp() if moment.tzinfo is not None else None


def limit_reset_of(messages: list[dict]) -> float | None:
    """When the limit the chat ends on resets (epoch seconds), else ``None``.

    Walks back to the newest ``error`` row. It counts only when it is a usage-limit
    row naming a readable ``reset_at``; a user or assistant row with content first
    means the chat moved on.
    """
    for m in reversed(messages):
        role = m.get("role")
        raw = m.get("meta")
        meta: dict[str, Any] = raw if isinstance(raw, dict) else {}
        if role == "error":
            return _instant(meta.get("reset_at")) if meta.get("kind") == USAGE_LIMIT_KIND else None
        if role in ("user", "assistant") and m.get("content"):
            return None
    return None


def cancel(slot_key: str, *, forget: bool = True) -> bool:
    """Drop the wait armed for *slot_key*; True when there was one and its turn was not queued.

    A wait whose continue is already under way stays cancellable until the turn is
    queued: the continue checks ``cancelled`` under the slot's lock just before it
    queues. Once queued, the wait is left to finish and this answers False. With
    *forget* false (a gateway stopping) the record keeps it, so the next boot arms
    it again.
    """
    armed = _ARMED.get(slot_key)
    if armed is None or armed.queued:
        return False
    del _ARMED[slot_key]
    armed.cancelled = True
    if armed.handle is not None:
        armed.handle.cancel()
    if forget:
        _save()
    return True


def _remember_started(slot: Any, *, turns: int) -> None:
    """Keep that *slot*'s continue queued its turn; *turns* is the last count it may show."""
    now = time.time()
    for key in [k for k, rec in _STARTED.items() if rec.until <= now or rec.slot() is None]:
        del _STARTED[key]
    _STARTED.pop(slot.key, None)
    while len(_STARTED) >= STARTED_CAP:
        del _STARTED[next(iter(_STARTED))]
    _STARTED[slot.key] = _Started(weakref.ref(slot), turns, now + STARTED_TTL_SECS)


def started(slot_key: str, slot: Any = None) -> bool:
    """Whether the wait for *slot_key* queued its turn, so Cancel cannot stop it.

    True while the wait's continue is under way, and, when *slot* is given, for
    :data:`STARTED_TTL_SECS` after it queued its turn, until a turn after the continued
    one starts or the chat is replaced.
    """
    armed = _ARMED.get(slot_key)
    if armed is not None:
        return armed.queued
    record = _STARTED.get(slot_key)
    if record is None:
        return False
    if (
        slot is None
        or record.slot() is not slot
        or record.until <= time.time()
        or getattr(slot, "_turns_started", 0) > record.turns
    ):
        if slot is not None:
            del _STARTED[slot_key]
        return False
    return True


async def cancel_all(_app: web.Application | None = None) -> None:
    """Stop every wait and any continue under way: the gateway is shutting down.

    The record is left as it is, so the next boot arms the same waits again.
    """
    for key in list(_ARMED):
        cancel(key, forget=False)
    firing = list(_FIRING)
    for task in firing:
        task.cancel()
    if firing:
        await asyncio.gather(*firing, return_exceptions=True)
    _STARTED.clear()


def armed_resume_iso(slot: Any) -> str | None:
    """The ISO instant *slot* will resume, while a wait is armed on the row it still ends on."""
    armed = _ARMED.get(slot.key)
    if armed is None or armed.queued:
        return None
    if not armed.holds(slot):
        # A turn or message in the chat since, or a replaced chat, ends the wait.
        cancel(slot.key)
        return None
    return datetime.fromtimestamp(armed.resume_at, tz=timezone.utc).isoformat()


def _may_continue(state: DashboardState, armed: _Armed, slot: Any) -> bool:
    """Whether the armed chat may take its continue now; every automatic wake asks the same."""
    from kiro_crew import crew_pause
    from kiro_crew.dashboard.goal_reminder import gateway_stopping
    from kiro_crew.dashboard.relay_archive import is_relay_archive

    return (
        not armed.cancelled
        and state._slots.get(armed.key) is slot
        and armed.holds(slot)
        and not is_relay_archive(slot)
        and not crew_pause.holds_slot(slot)
        and not gateway_stopping()
    )


async def _fire(state: DashboardState, armed: _Armed) -> None:
    try:
        await _continue(state, armed)
    finally:
        # The wait ends with its continue, queued or not; a queued one stays on record
        # so a Cancel that arrives later still hears that it started.
        if _ARMED.get(armed.key) is armed:
            del _ARMED[armed.key]
            # A stopping gateway keeps a wait it did not queue on record for the next boot.
            from kiro_crew.dashboard.goal_reminder import gateway_stopping

            if armed.queued or not gateway_stopping():
                _save()
        slot = armed.slot()
        if armed.queued and slot is not None:
            _remember_started(slot, turns=armed.turns + 1)
    state.push_slots_update()


async def _continue(state: DashboardState, armed: _Armed) -> None:
    slot = armed.slot()
    if slot is None or not _may_continue(state, armed, slot):
        return
    from kiro_crew.dashboard.chat_handlers import SlotContinueRefusal, continue_slot_turn

    def _still_due() -> None:
        # Under the slot's lock, with no await before the continue is queued: a
        # Cancel, a message, a pause or a replacement that landed while the lock and
        # the probes were awaited.
        if not _may_continue(state, armed, slot):
            raise SlotContinueRefusal("the chat moved on", "limit_continue_moved_on")
        armed.queued = True
        # Off the record before the turn runs, so a crash mid-turn cannot repeat it.
        _save()

    try:
        await continue_slot_turn(
            state,
            slot,
            directive_user_origin=True,
            via="usage_limit_reset",
            allow_restored_queue=True,
            before_dispatch=_still_due,
            replaced=lambda: state._slots.get(armed.key) is not slot,
        )
    except SlotContinueRefusal as refusal:
        logger.info("limit-reset continue of %s refused: %s", armed.key, refusal.code)
    except Exception:
        logger.warning("limit-reset continue of %s failed", armed.key, exc_info=True)


def _wake(state: DashboardState, armed: _Armed) -> None:
    """The timer ran out: continue now, or wait out what the wall clock still owes."""
    if _ARMED.get(armed.key) is not armed or armed.firing:
        return
    slot = armed.slot()
    if slot is None or state._slots.get(armed.key) is not slot:
        # The chat closed or was replaced: nothing is left to continue.
        del _ARMED[armed.key]
        _save()
        return
    if armed.resume_at > time.time():
        # Not due on the wall clock yet: wait another slice, since a suspended host
        # is time the loop's timer does not count.
        _schedule(state, armed)
        return
    # The wait stays armed, so Cancel keeps its authority until the turn is queued.
    armed.firing = True
    task = armed.loop.create_task(_fire(state, armed))
    _FIRING.add(task)
    task.add_done_callback(_FIRING.discard)


def arm(state: DashboardState, slot: Any, reset_at: float) -> _Armed:
    """Hold one continue for *slot* until *reset_at* plus the margin, replacing any earlier wait."""
    cancel(slot.key)
    _STARTED.pop(slot.key, None)
    loop = asyncio.get_running_loop()
    armed = _Armed(
        key=slot.key,
        slot=weakref.ref(slot),
        reset_at=reset_at,
        turns=getattr(slot, "_turns_started", 0),
        resume_at=reset_at + RESUME_MARGIN_SECS,
        loop=loop,
    )
    _schedule(state, armed)
    _ARMED[slot.key] = armed
    _save()
    return armed


def _schedule(state: DashboardState, armed: _Armed) -> None:
    """Wake at the resume time or after one slice, whichever comes first."""
    owed = min(max(armed.resume_at - time.time(), 0.0), WALL_RECHECK_SECS)
    armed.handle = armed.loop.call_later(owed, _wake, state, armed)


def _declined(warning: str) -> web.Response:
    return web.json_response({"ok": False, "warning": warning})


async def _sent_reset_at(request: web.Request) -> tuple[float | None, str]:
    """The instant the caller says it read off the row, and a warning for what was dropped.

    No body, an empty object or no ``reset_at`` is none, with no warning. A body that
    is not a JSON object, or a ``reset_at`` that is not an ISO instant with a zone, is
    dropped with a warning, and the chat's own row decides.
    """
    if not await request.read():
        return None, ""
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return None, "the request body is not a JSON object, so it was ignored"
    raw = body.get("reset_at")
    if raw is None:
        return None, ""
    sent = _instant(raw)
    if sent is None:
        return None, "reset_at is not an ISO instant with a time zone, so it was ignored"
    return sent, ""


async def api_chat_slot_limit_continue(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{slot}/limit-continue — continue this chat when its limit resets.

    A request that cannot be honoured answers 200 with ``ok`` false and a
    ``warning``, and arms nothing: the card keeps its Resume button. A malformed
    body is dropped: the chat's own row decides, and the answer carries a
    ``warning`` naming what was dropped.
    """
    state: DashboardState = request.app["state"]
    # The owner check comes first, so a caller who is not the owner learns nothing
    # about which chats exist: the same refusal for a missing slot and a live one.
    denied = await require_owner_dashboard_request(request, "chat.limit_continue")
    if denied is not None:
        return denied
    slot = state._slots.get(request.match_info["slot"])
    if slot is None:
        return slot_not_found()
    reset_at = limit_reset_of(slot.messages)
    if reset_at is None:
        return _declined("this chat does not end on a usage limit that names a reset time")
    sent, dropped = await _sent_reset_at(request)
    if sent is not None and abs(sent - reset_at) >= _SAME_INSTANT_SECS:
        return _declined("the reset time sent is not the one this chat's limit names")
    if reset_at <= time.time():
        return _declined("that reset time has already passed")
    armed = arm(state, slot, reset_at)
    state.push_slots_update()
    answer: dict[str, Any] = {
        "ok": True,
        "slot": slot.key,
        "continue_at": datetime.fromtimestamp(armed.resume_at, tz=timezone.utc).isoformat(),
    }
    if dropped:
        logger.warning("limit-continue for %s: %s", slot.key, dropped)
        answer["warning"] = dropped
    return web.json_response(answer)


async def api_chat_slot_limit_continue_cancel(request: web.Request) -> web.Response:
    """DELETE /api/chat/slots/{slot}/limit-continue — cancel the wait for this chat.

    ``cancelled`` is True when a wait was dropped before its turn was queued, including
    one whose continue was already under way. When the turn was already queued, even if
    the wait has since ended, nothing is cancelled and the answer carries ``started: true``.
    """
    state: DashboardState = request.app["state"]
    # The owner check comes first, so a caller who is not the owner learns nothing
    # about which chats exist: the same refusal for a missing slot and a live one.
    denied = await require_owner_dashboard_request(request, "chat.limit_continue")
    if denied is not None:
        return denied
    slot = state._slots.get(request.match_info["slot"])
    if slot is None:
        return slot_not_found()
    if started(slot.key, slot):
        return web.json_response({"ok": True, "cancelled": False, "started": True})
    cancelled = cancel(slot.key)
    if cancelled:
        state.push_slots_update()
    return web.json_response({"ok": True, "cancelled": cancelled})
