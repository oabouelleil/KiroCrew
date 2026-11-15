"""Move a chat that stopped on a usage limit onto another harness, and continue it there.

A turn that stops on a spent allowance leaves an ``error`` row of kind
``usage_limit`` (see :mod:`kiro_crew.dashboard.limit_continue`). Instead of waiting
for the reset, the chat can carry on on another agent harness:

* ``POST /api/chat/slots/{slot}/limit-switch`` with ``{"backend": "<name>"}`` is the
  error card's "Switch to <harness> and continue";
* with ``agent.acp_failover`` enabled and naming the ``usage_limit`` trigger, the
  gateway does the same by itself when a chat's queue cycle ends on such a row
  (:func:`after_cycle_end`), moving it to the first chain member after the harness
  it ran on that can start now.

Both run :func:`switch_and_continue`: the harness switch
(``chat_handlers.switch_slot_harness``, which applies every gate, the
``session_harness`` preview included), then the continue Resume runs
(``chat_handlers.continue_slot_turn``). After the boundary row the switch appends,
the chat does not end on the limit row, so a "Continue automatically" wait armed
on that row could never fire; it is cancelled too.

Never onto the harness that hit the limit, and never round and round: the gateway
moves a chat by itself at most once per message a person sends it, and never an
app's chat. When the harness it moved to also stops on a limit before the person's
next message, the chat stays there and shows that limit's card. The chat does not
move back when the first limit resets.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

from kiro_crew.dashboard.chat_utils import USAGE_LIMIT_KIND
from kiro_crew.dashboard.limit_continue import cancel as cancel_limit_continue
from kiro_crew.dashboard.limit_continue import ends_on_usage_limit
from kiro_crew.dashboard.system_notices import SESSION_SWITCH_KIND, SESSION_SWITCH_REASON_KEY
from kiro_crew.history_schema import HUMAN_TURN_META_KEY

if TYPE_CHECKING:
    from kiro_crew.dashboard.chat_handlers import SlotAgentSwitchCaller
    from kiro_crew.dashboard.state import DashboardState

logger = logging.getLogger(__name__)

#: The ``via`` the continue after a usage-limit switch records on its audit line.
LIMIT_SWITCH_VIA = "usage_limit_switch"


def moved_since_last_message(messages: list[dict]) -> bool:
    """Whether a usage-limit switch came after the newest message a person sent the chat.

    Walks back to the newest ``user`` row a person authored (``HUMAN_TURN_META_KEY``).
    A Resume, a recovery, a loop's wake and a message another session or an app
    sends are not one, so the turn a switch continued, and any automated turn
    after it, still count as the same stretch.
    """
    for m in reversed(messages):
        role = m.get("role")
        meta = m.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        if role == "user" and m.get("content") and meta.get(HUMAN_TURN_META_KEY):
            return False
        if (
            role == "assistant"
            and meta.get("kind") == SESSION_SWITCH_KIND
            and meta.get(SESSION_SWITCH_REASON_KEY) == USAGE_LIMIT_KIND
        ):
            return True
    return False


async def switch_and_continue(
    state: DashboardState,
    slot: Any,
    backend: object,
    caller: SlotAgentSwitchCaller,
    *,
    user_origin: bool,
) -> web.Response:
    """Move *slot*, which ends on a usage limit, onto *backend* and continue it there.

    Refused with whatever ``switch_slot_harness(usage_limit=True)`` refuses, which
    includes ``not_on_usage_limit`` for a chat that does not end on a usage-limit row
    and ``limit_switch_same_harness`` for the harness it already runs. A switch that
    commits answers 200 with the switch's own body plus ``continued`` (``"resumed"``
    or ``"continued"``, or ``""`` with ``continue_refused`` and a ``warning`` when the
    continue could not queue). *user_origin* is the continue's
    ``directive_user_origin``: True only when a person asked for it.
    """
    from kiro_crew.dashboard.chat_handlers import (
        SlotContinueRefusal,
        agent_switch_outcome,
        continue_slot_turn,
        switch_slot_harness,
    )

    switched = await switch_slot_harness(state, slot, slot.key, backend, caller, usage_limit=True)
    ok, answer = agent_switch_outcome(switched)
    if not ok:
        return switched
    if cancel_limit_continue(slot.key):
        state.push_slots_update()
    try:
        answer["continued"] = await continue_slot_turn(
            state,
            slot,
            directive_user_origin=user_origin,
            via=LIMIT_SWITCH_VIA,
            allow_restored_queue=True,
            replaced=lambda: state._slots.get(slot.key) is not slot,
        )
    except SlotContinueRefusal as refusal:
        # A message typed or a turn started meanwhile: the switch stands and the
        # chat carries on from that instead.
        answer["continued"] = ""
        answer["continue_refused"] = refusal.code
        answer["warning"] = f"the harness switched, but the chat did not continue: {refusal}"
    return web.json_response(answer)


async def api_chat_slot_limit_switch(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{slot}/limit-switch -- move off a usage limit and continue.

    Body ``{"backend": "<name>"}``, spelled as the harness switch spells it.
    """
    from kiro_crew.dashboard.chat_handlers import SlotAgentSwitchCaller, read_bounded_json
    from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request
    from kiro_crew.dashboard.slot_ownership import slot_not_found

    state: DashboardState = request.app["state"]
    # The owner check comes first, so a caller who is not the owner learns nothing
    # about which chats exist: the same refusal for a missing slot and a live one.
    denied = await require_owner_dashboard_request(request, "chat.limit_switch")
    if denied is not None:
        return denied
    slot = state._slots.get(request.match_info["slot"])
    if slot is None:
        return slot_not_found()
    body, body_err = await read_bounded_json(request)
    if body_err is not None:
        return body_err
    assert body is not None  # read_bounded_json returns (dict, None) on success
    # Provenance follows the caller, as the Continue route's does: an app's request
    # never gets the authenticated person's authority.
    return await switch_and_continue(
        state,
        slot,
        body.get("backend"),
        SlotAgentSwitchCaller.for_request(request),
        user_origin=not request.get("app"),
    )


async def _auto_switch(state: DashboardState, slot: Any) -> None:
    """Move *slot* off its usage limit to the next ``agent.acp_failover`` member, if one can."""
    from kiro_crew import crew_pause
    from kiro_crew.acp_failover import failover_policy
    from kiro_crew.config import KiroCrewConfig
    from kiro_crew.dashboard.chat_handlers import (
        SlotAgentSwitchCaller,
        agent_switch_outcome,
        slot_harness_now,
    )
    from kiro_crew.dashboard.goal_reminder import gateway_stopping
    from kiro_crew.dashboard.handlers import messaging
    from kiro_crew.dashboard.relay_archive import is_relay_archive

    if (
        state._slots.get(slot.key) is not slot
        # An app's chat runs on the harness the app chose.
        or slot._app
        or not ends_on_usage_limit(slot.messages)
        or is_relay_archive(slot)
        or crew_pause.holds_slot(slot)
        or gateway_stopping()
    ):
        return
    policy = failover_policy((await asyncio.to_thread(KiroCrewConfig.load)).agent.acp_failover)
    if not policy.enabled or USAGE_LIMIT_KIND not in policy.triggers:
        return
    if moved_since_last_message(slot.messages):
        logger.info("usage-limit switch of %s skipped: it already moved once", slot.key)
        return
    current = await slot_harness_now(state, slot)
    if current is None:
        return
    available, _states = await messaging._startable_backends()
    target, _tried = await messaging._failover_backend(
        messaging._backend_name(current), USAGE_LIMIT_KIND, available
    )
    if target is None:
        return
    # The gateway acting on the owner's own config, as the Crew Mode migration does.
    # No person asked for this continue, so it carries no person's authority.
    caller = SlotAgentSwitchCaller(app="", remote=LIMIT_SWITCH_VIA, owner=True)
    resp = await switch_and_continue(
        state, slot, messaging._backend_name(target), caller, user_origin=False
    )
    ok, body = agent_switch_outcome(resp)
    if ok:
        logger.info("usage-limit switch of %s to %s: %s", slot.key, target, body.get("continued"))
    else:
        logger.info("usage-limit switch of %s refused: %s", slot.key, body.get("code"))


def after_cycle_end(state: DashboardState, slot: Any) -> None:
    """A chat's queue cycle ended with nothing left to run: switch it when it ends on a limit.

    Synchronous and cheap when the chat does not end on a usage-limit row; otherwise
    the switch runs in a tracked task of its own, since it takes the slot's lock.
    """
    if not ends_on_usage_limit(slot.messages):
        return

    async def _run() -> None:
        try:
            await _auto_switch(state, slot)
        except Exception:
            logger.warning("usage-limit switch of %s failed", slot.key, exc_info=True)

    task = asyncio.ensure_future(_run())
    state._background_tasks.add(task)
    task.add_done_callback(state._background_tasks.discard)
