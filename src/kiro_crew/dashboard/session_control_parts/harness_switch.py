"""A target's harness: ``session_set_backend``.

The session-control door to ``chat_api.slot_harness.switch_slot_harness``, the one
entry point every harness switch goes through, so this verb gets every refusal the
composer's Harness row gets and adds only the session-control gate around it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from kiro_crew.dashboard.session_control_parts.caller_access import (
    SessionControlError,
    _audit,
    _caller_is_ownership_fenced,
    caller_slot_key,
    prewarm_enabled_check,
)
from kiro_crew.dashboard.session_control_parts.target_authorization import authorize_target
from kiro_crew.sel import sel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kiro_crew.dashboard.state import DashboardState, _ChatSlot

# The session-control family's logger name, which log filters key on.
logger = logging.getLogger("kiro_crew.dashboard.session_control")


async def set_backend_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    backend: object,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Move idle *target* onto the harness *backend* names, now.

    Reaches the same sessions ``session_set_model`` does (``authorize_target``,
    never the caller itself). Unlike that verb the change is not deferred to the
    target's next turn: ``switch_slot_harness`` commits it and resets the session
    under the target's own locks, and refuses a busy target rather than waiting.
    The gate runs again, synchronously, right before that commit
    (``SlotAgentSwitchCaller.still_permitted``), so a target that became
    unreachable during the switch's awaits is refused with nothing changed.

    A refusal from the switch keeps its own code (``turn_in_flight``,
    ``slot_subagents_running``, ``unknown_backend``, ...) and its message; a
    target that cannot start names the harnesses that can.

    ``caller_fenced`` has the meaning ``stop_target`` documents.
    """
    # Deferred for the same import cycle `stop_target` documents.
    from kiro_crew.dashboard.chat_handlers import SlotAgentSwitchCaller, switch_slot_harness

    # Validated before any gate, as `set_model_target` does.
    if not isinstance(backend, str) or not backend.strip():
        raise SessionControlError(
            "backend must be a harness name", code="invalid_backend", status=400
        )
    name = backend.strip()
    # Same prewarm ordering as `stop_target`.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the switch
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()
    caller_key = caller_slot_key(state, caller_session_key)
    if caller_fenced is None:
        caller_fenced = bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="set_backend",
        precomputed_ownership_fenced=caller_fenced,
    )
    slot_key = slot.key

    def still_permitted() -> bool:
        return _still_reachable(state, slot, caller_session_key, caller_fenced)

    caller = SlotAgentSwitchCaller(
        app="", remote="session_control", owner=False, still_permitted=still_permitted
    )
    try:
        response = await switch_slot_harness(state, slot, slot_key, name, caller)
    except Exception as exc:
        # The switch rolls the pick back on a raise before the session is reset.
        logger.exception("session_set_backend of %s failed", slot_key)
        _audit(
            caller_session_key=caller_session_key,
            operation="set_backend",
            slot_key=slot_key,
            outcome="denied",
            detail={"code": "set_backend_failed", "error": type(exc).__name__},
        )
        raise SessionControlError(
            "the harness switch failed; harness not changed",
            code="set_backend_failed",
            status=500,
        ) from exc
    body = json.loads(response.text or "{}")
    if response.status != 200:
        code = str(body.get("code") or "set_backend_failed")
        _audit(
            caller_session_key=caller_session_key,
            operation="set_backend",
            slot_key=slot_key,
            outcome="denied",
            detail={"code": code, "backend": name},
        )
        raise SessionControlError(
            str(body.get("error") or "harness not changed"), code=code, status=response.status
        )
    result: dict[str, Any] = {
        "ok": True,
        "target": slot_key,
        "backend": body.get("backend", name),
        "harness": body.get("harness", name),
    }
    if body.get("warning"):
        result["warning"] = body["warning"]
    _audit(
        caller_session_key=caller_session_key,
        operation="set_backend",
        slot_key=slot_key,
        outcome="allowed",
        detail={"backend": result["backend"]},
    )
    return result


def _still_reachable(
    state: "DashboardState", slot: "_ChatSlot", caller_session_key: str, caller_fenced: bool
) -> bool:
    """Whether the caller may still act on *slot*, judged synchronously at the commit."""
    if state._slots.get(slot.key) is not slot:
        return False
    try:
        live = authorize_target(
            state,
            caller_session_key=caller_session_key,
            target=slot.key,
            operation="set_backend",
            skip_enabled_check=True,
            precomputed_ownership_fenced=caller_fenced,
        )
    except SessionControlError:
        return False
    return live is slot
