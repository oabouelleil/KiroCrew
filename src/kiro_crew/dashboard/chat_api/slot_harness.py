"""The harness switch: ``POST /api/chat/slots/{slot}/backend`` and :func:`switch_slot_harness`.

Moves an existing chat onto another agent harness. The chat keeps its key, its
transcript and its queue; the pick ``session_create(backend=...)`` stores at birth
(``slot.backend``) is re-pointed and the live session is reset, so the next turn
allocates on the new harness. That allocation finds the saved session id belongs
to another harness (``detect_provider_switch``), drops it and replays the visible
transcript, so the new harness starts with the conversation so far.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from kiro_crew.dashboard.chat_handlers import (
        _TEARDOWN_INCOMPLETE_WARNING,
        AcpProvider,
        DashboardState,
        LLMProvider,
        SlotAgentSwitchCaller,
        _app_cancel_denied,
        _append_session_switch_notice,
        _ChatSlot,
        _configured_backend_for_slot,
        _reset_slot_session_or_warn,
        _slot_replaced_while_queued,
        _slot_switch_session_lock,
        _subagents_attached_response,
        _switch_target_busy,
        capabilities_of,
        deny_app_slot_access,
        effective_session_key,
        logger,
        read_bounded_json,
        relay_archive_refusal,
        slot_not_found,
    )


async def api_chat_slot_backend(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{slot}/backend -- move the chat onto another harness.

    Body ``{"backend": "<name>"}``, the name spelled as ``spawn_run(backend=...)``
    and governance spell it (``kiro``, ``kas``, ``claude``, ``codex``, ...).
    """
    state: DashboardState = request.app["state"]
    name = request.match_info["slot"]
    slot = state._slots.get(name)
    if not slot:
        return slot_not_found()
    denied = deny_app_slot_access(request.get("app", ""), slot, name, "slot_backend")
    if denied is not None:
        return denied
    body, body_err = await read_bounded_json(request)
    if body_err is not None:
        return body_err
    assert body is not None  # read_bounded_json returns (dict, None) on success
    return await switch_slot_harness(
        state, slot, name, body.get("backend"), SlotAgentSwitchCaller.for_request(request)
    )


def _harness_refusal(message: str, code: str, status: int) -> web.Response:
    return web.json_response({"error": message, "code": code}, status=status)


def _caller_withdrawn(caller: SlotAgentSwitchCaller) -> web.Response | None:
    """The refusal when the caller's own gate closed during the awaits, else ``None``.

    Run synchronously right before each commit, as the agent switch runs it.
    """
    if caller.still_permitted is not None and not caller.still_permitted():
        return _harness_refusal("the caller is no longer permitted", "caller_not_permitted", 403)
    return None


def _current_harness(provider: object) -> str | None:
    """The runtime backend id *provider* runs, or ``None`` when it names none."""
    if not isinstance(provider, AcpProvider):
        return None
    backend = capabilities_of(provider).backend
    return backend if isinstance(backend, str) else None


async def slot_harness_now(state: DashboardState, slot: _ChatSlot) -> str | None:
    """The runtime backend id *slot* runs: its live session's, else the one its next start uses.

    ``None`` when neither can be read.
    """
    from kiro_crew.crewmate_harness import CrewmateHarnessUnavailable

    on_now = _current_harness(state.sessions.get_provider(effective_session_key(slot)))
    if on_now is not None:
        return on_now
    try:
        return await _configured_backend_for_slot(slot)
    except CrewmateHarnessUnavailable:
        return None


async def switch_slot_harness(
    state: DashboardState,
    slot: _ChatSlot,
    name: str,
    raw_backend: object,
    caller: SlotAgentSwitchCaller,
    *,
    usage_limit: bool = False,
) -> web.Response:
    """Move *slot* onto the harness *raw_backend* names. The one entry point for every caller.

    Refused, with nothing changed, while the ``session_harness`` preview is off, for a
    harness that cannot start here (the spawn endpoint's check, without its failover
    walk), for a session whose harness a channel or a Bot decides, while a turn runs
    and while sub-agents are attached. A caller whose ``still_permitted`` turns false
    during the awaits is refused at the commit. A chat already on the harness answers 200 and
    keeps its session. Otherwise the pick is committed, the session is reset, and the
    transcript gets a boundary row; a reset a slipped-in turn declines is rolled back.

    *usage_limit* is the usage-limit switch (``dashboard.limit_switch``):
    a chat already on the harness is refused with ``limit_switch_same_harness``, since
    staying there only meets the same limit, and the boundary row says why it moved
    and carries the reason in its meta.
    """
    from kiro_crew import session_harness_preview as harness_preview
    from kiro_crew.agent_sdk.backends import backend_to_stored
    from kiro_crew.crewmate_harness import harness_label
    from kiro_crew.dashboard.chat_utils import USAGE_LIMIT_KIND
    from kiro_crew.dashboard.handlers import messaging
    from kiro_crew.dashboard.session_control import channel_drives_session
    from kiro_crew.dashboard.system_notices import SESSION_SWITCH_REASON_KEY
    from kiro_crew.members import (
        is_crewmate_worker_session,
        is_member_session_key,
        note_session_backend_pick,
    )

    if not await asyncio.to_thread(harness_preview.session_harness_enabled):
        return _harness_refusal(
            harness_preview.PREVIEW_OFF_MESSAGE, harness_preview.PREVIEW_OFF_CODE, 403
        )
    if not isinstance(raw_backend, str) or not raw_backend:
        return _harness_refusal("backend must be a harness name", "invalid_backend", 400)
    target, _replaced, refusal = await messaging._requested_backend(raw_backend, failover=False)
    if refusal is not None:
        return refusal
    assert target is not None  # a named backend that is not refused is the target
    refusal = relay_archive_refusal(slot)
    if refusal is not None:
        return refusal
    stored = backend_to_stored(target)
    label = harness_label(stored)
    # Lock order as every sibling switch: slot._lock, then the session lock (see
    # _slot_switch_session_lock). The session key is read inside the slot lock.
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(slot._lock)
        if _slot_replaced_while_queued(state, slot, name, caller.scope, "chat.slot_backend"):
            return slot_not_found()
        session_key = effective_session_key(slot)
        await stack.enter_async_context(_slot_switch_session_lock(session_key))
        if _slot_replaced_while_queued(state, slot, name, caller.scope, "chat.slot_backend"):
            return slot_not_found()
        denied = _app_cancel_denied(caller.scope, slot, "chat.slot_backend", session_key)
        if denied is not None:
            return denied
        # A linked tab (a channel, cron, workflow or resumed session), a member DM and a
        # Bot's own worker get their harness from their own routing: the runs that drive
        # such a session would not read a per-chat pick, so the chat would flip harness
        # on every turn that alternates between them.
        if (
            getattr(slot, "linked_session_key", "")
            or channel_drives_session(state, slot, session_key)
            or is_member_session_key(session_key)
            or is_crewmate_worker_session(session_key)
        ):
            return _harness_refusal(
                "this chat's harness is set by its channel or its Bot, not per chat",
                "harness_switch_unsupported",
                409,
            )
        on_now = await slot_harness_now(state, slot)
        if on_now is None:
            # Nothing to roll back to: a refused switch could not restore what ran.
            return _harness_refusal(
                "the harness this chat runs on cannot be read", "harness_switch_unsupported", 409
            )
        if on_now == target:
            if usage_limit:
                return _harness_refusal(
                    "this chat already runs on that harness, which hit the limit",
                    "limit_switch_same_harness",
                    409,
                )
            denied = _caller_withdrawn(caller)
            if denied is not None:
                return denied
            # Already there: record the pick so the chat stays on it, tear nothing down.
            if slot.backend != stored:
                slot.backend = stored
                note_session_backend_pick(slot.key, stored)
                slot._dirty = True
                state.push_slots_update()
            return web.json_response({"ok": True, "backend": stored, "harness": label})
        children_409 = await _subagents_attached_response(state, slot, session_key, "slot_backend")
        if children_409 is not None:
            return children_409
        if _switch_target_busy(
            state,
            slot,
            session_key,
            state.sessions.get_provider(session_key),
            own_turn=caller.own_turn,
        ):
            return _harness_refusal("a turn is in flight", "turn_in_flight", 409)
        denied = _caller_withdrawn(caller)
        if denied is not None:
            return denied
        prior = slot.backend
        # Rolled back to an explicit pick of the harness the chat was on: a merge save
        # writes only a truthy pick, so restoring "" could leave the new one on disk.
        restore = prior or backend_to_stored(on_now)
        # Committed before the reset: a send landing during the reset await
        # cold-starts from the slot's current pick, so it must already be the new one.
        slot.backend = stored
        note_session_backend_pick(slot.key, stored)

        def _rollback() -> None:
            if slot.backend == stored:
                slot.backend = restore
                note_session_backend_pick(slot.key, restore)
            # The periodic flush may have written the new pick during the await.
            slot._dirty = True

        async def _reset() -> bool | None:
            try:
                return await _reset_slot_session_or_warn(
                    state, slot, session_key, switch_kind="backend"
                )
            except BaseException:
                # Raised before the session was popped: the old session still runs
                # the old harness, so the slot must say so too.
                _rollback()
                raise

        logger.info("Slot %s harness switched to %r, resetting session", name, stored)
        teardown_incomplete = False
        reset_ok = await _reset()
        if reset_ok is None:
            teardown_incomplete = True
        elif not reset_ok:
            busy = state.sessions.get_provider(session_key)
            if _current_harness(busy) == target:
                # A send that landed after the commit already cold-started on the target.
                pass
            elif isinstance(busy, LLMProvider):
                if busy.has_active_turn():
                    _rollback()
                    return _harness_refusal("a turn is in flight", "turn_in_flight", 409)
                # An idle session declined: retry once, as the workspace switch does.
                # The first reset awaited, so the caller's gate is read again first.
                denied = _caller_withdrawn(caller)
                if denied is not None:
                    _rollback()
                    return denied
                reset_ok = await _reset()
                if reset_ok is None:
                    teardown_incomplete = True
                elif not reset_ok:
                    _rollback()
                    return _harness_refusal("a turn is in flight", "turn_in_flight", 409)
        if effective_session_key(slot) != session_key:
            _rollback()
            return _harness_refusal(
                "slot session was rebound during the switch", "session_rebound", 409
            )
        slot._dirty = True
        if usage_limit:
            await _append_session_switch_notice(
                slot,
                "backend_usage_limit",
                label,
                meta={SESSION_SWITCH_REASON_KEY: USAGE_LIMIT_KIND},
            )
        else:
            await _append_session_switch_notice(slot, "backend", label)
    state.push_slots_update()
    resp: dict = {"ok": True, "backend": stored, "harness": label}
    if teardown_incomplete:
        resp["warning"] = _TEARDOWN_INCOMPLETE_WARNING
    return web.json_response(resp)
