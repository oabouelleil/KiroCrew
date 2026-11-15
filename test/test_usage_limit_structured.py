"""A usage limit claude-agent-acp reports as data, not only as a sentence.

claude-agent-acp fails a turn the CLI stopped on a spent allowance with
``data = {"errorKind": "rate_limit", "rateLimit": <SDKRateLimitInfo>}``, whose
``resetsAt`` (epoch seconds) says when it opens again and ``rateLimitType`` which
window it is. Pinned here:

* a rejected ``rateLimit`` is a usage limit even when its sentence is one the
  wording pattern does not know, and its ``resetsAt`` wins over the sentence;
* a ``resetsAt`` that is not a number, not ahead, or past the cap falls back to the
  sentence; a status that is not ``rejected`` is not a limit by itself;
* the Opus and Sonnet limits' own sentences are limits, a "rate limit" is not;
* the row a limit leaves carries the window and the harness, and a limit is never
  retried as transient.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from kiro_crew import limit_reset
from kiro_crew.acp.client import AcpError
from kiro_crew.acp.transport_errors import _raise_acp_error, classify_provider_error
from kiro_crew.acp_backends import ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX
from kiro_crew.limit_reset import MAX_RESET_AHEAD_SECS, UsageLimit, read_usage_limit
from kiro_crew.testing.clock import ManualClock


@pytest.fixture
def clock(monkeypatch) -> ManualClock:
    c = ManualClock()
    c.install(monkeypatch, limit_reset, time=False, datetime=True)
    return c


def _frame(message: str, rate_limit: dict | None, kind: str = "rate_limit") -> dict:
    data: dict = {"errorKind": kind}
    if rate_limit is not None:
        data["rateLimit"] = rate_limit
    return {"code": -32603, "message": message, "data": data}


def _rejected(resets_at: object, window: object = "five_hour") -> dict:
    return {"status": "rejected", "resetsAt": resets_at, "rateLimitType": window}


def _raised(error: dict, backend: str = ACP_BACKEND_CLAUDE) -> AcpError:
    with pytest.raises(AcpError) as excinfo:
        _raise_acp_error(error, backend=backend)
    return excinfo.value


class TestReadUsageLimit:
    def test_the_structured_reset_wins_over_the_sentence(self, clock):
        at = clock.time() + 4000
        frame = _frame("You've hit your 5-hour limit \u00b7 resets in 2 hours", _rejected(at))
        found = read_usage_limit(frame, frame["message"], worded=True)
        assert found == UsageLimit(float(at), "five_hour")

    @pytest.mark.parametrize(
        "resets_at",
        ["soon", True, None, -1, "1800000000"],
    )
    def test_an_unusable_structured_reset_falls_back_to_the_sentence(self, clock, resets_at):
        frame = _frame(
            "You've hit your weekly limit \u00b7 resets in 2 hours", _rejected(resets_at)
        )
        found = read_usage_limit(frame, frame["message"], worded=True)
        assert found is not None
        assert found.reset_at == clock.time() + 7200

    def test_a_reset_past_the_cap_falls_back_too(self, clock):
        # A millisecond epoch read as seconds lands far past the cap.
        frame = _frame("You've hit your weekly limit", _rejected(clock.time() * 1000))
        assert read_usage_limit(frame, frame["message"], worded=True) == UsageLimit(
            None, "five_hour"
        )
        frame = _frame("x", _rejected(clock.time() + MAX_RESET_AHEAD_SECS + 1))
        assert read_usage_limit(frame, "x", worded=False) == UsageLimit(None, "five_hour")

    def test_a_rejected_rate_limit_is_a_limit_without_known_wording(self, clock):
        at = clock.time() + 600
        frame = _frame("Internal error", _rejected(at, "seven_day_sonnet"))
        assert read_usage_limit(frame, "Internal error", worded=False) == UsageLimit(
            float(at), "seven_day_sonnet"
        )

    @pytest.mark.parametrize(
        "frame",
        [
            _frame("slow down", {"status": "allowed_warning", "resetsAt": 1}),
            _frame("slow down", None),
            _frame("x", _rejected(1), kind="billing_error"),
            {"code": -32603, "message": "x", "data": "rateLimit rejected"},
            "not a frame",
        ],
    )
    def test_anything_else_unworded_is_no_limit(self, frame):
        assert read_usage_limit(frame, "x", worded=False) is None

    @pytest.mark.parametrize("window", ["Five Hour!", "x" * 41, 7, "", None])
    def test_a_window_not_in_the_harness_shape_is_dropped(self, clock, window):
        frame = _frame("x", _rejected(clock.time() + 60, window))
        found = read_usage_limit(frame, "x", worded=False)
        assert found is not None and found.window == ""


class TestTheRaise:
    def test_the_raised_error_is_a_terminal_limit_with_window_and_harness(self, clock):
        at = clock.time() + 3600
        exc = _raised(
            _frame(
                "You've hit your Opus limit \u00b7 resets 3:45pm", _rejected(at, "seven_day_opus")
            )
        )
        assert exc.usage_limit is True
        assert exc.transient is False
        assert exc.usage_limit_reset_at == at
        assert exc.usage_limit_window == "seven_day_opus"
        assert exc.usage_limit_harness == ACP_BACKEND_CLAUDE

    def test_a_structured_limit_with_unknown_wording_is_not_retried(self, clock):
        exc = _raised(_frame("Request rate limited, try again", _rejected(clock.time() + 60)))
        assert exc.usage_limit is True
        assert exc.transient is False

    def test_the_row_carries_the_window_and_harness(self, clock):
        from kiro_crew.dashboard.chat_runner import _terminal_error_meta
        from kiro_crew.dashboard.chat_utils import USAGE_LIMIT_KIND

        at = clock.time() + 3600
        meta = _terminal_error_meta(_raised(_frame("Internal error", _rejected(at))))
        assert meta is not None
        assert meta["kind"] == USAGE_LIMIT_KIND
        assert datetime.fromisoformat(str(meta["reset_at"])).timestamp() == at
        assert meta["limit_window"] == "five_hour"
        assert meta["harness"] == ACP_BACKEND_CLAUDE

    def test_a_limit_with_no_reset_still_names_its_window(self):
        from kiro_crew.dashboard.chat_runner import _terminal_error_meta

        meta = _terminal_error_meta(_raised(_frame("Internal error", _rejected(None, "seven_day"))))
        assert meta is not None
        assert "reset_at" not in meta
        assert meta["limit_window"] == "seven_day"


@pytest.mark.parametrize(
    ("text", "is_limit"),
    [
        ("You've hit your Opus limit \u00b7 resets 3:45pm", True),
        ("You've hit your Sonnet limit \u00b7 resets 3:45pm", True),
        ("You've hit your 5-hour limit \u00b7 resets 3:45pm", True),
        ("You've hit your weekly limit \u00b7 resets Mon 12:00am", True),
        ("You've hit your rate limit", False),
        ("the opus limit of this hardware", False),
    ],
)
def test_the_model_family_limits_are_worded_limits(text, is_limit):
    assert (classify_provider_error(text).kind == "usage_limit") is is_limit


class TestCodexWording:
    """Codex's limit sentences (codex-rs ``UsageLimitReachedError``) are limits too.

    Codex formats the reset in the host's own zone: a clock time on the same day
    ("Try again at 3:45 PM."), else "Nov 16th, 2026 9:00 AM". The test host runs on
    UTC (the suite pins it), so a naive wall clock here is UTC.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "You\u2019ve hit your usage limit. Try again at 3:45 PM.",
            "You've hit your usage limit. Try again later.",
            "You\u2019ve hit your usage limit for GPT-6. Switch to another model now,"
            " or try again at Nov 16th, 2026 9:00 AM.",
        ],
    )
    def test_codex_sentences_are_limits(self, text):
        assert classify_provider_error(text).kind == "usage_limit"

    def test_a_same_day_clock_is_read(self):
        now = datetime(2026, 11, 15, 12, 0).astimezone()
        at = limit_reset.parse_limit_reset(
            "You've hit your usage limit. Try again at 3:45 PM.", now
        )
        assert at == datetime(2026, 11, 15, 15, 45).timestamp()

    def test_a_dated_clock_is_read(self):
        now = datetime(2026, 11, 15, 12, 0).astimezone()
        text = "You've hit your usage limit, or try again at Nov 16th, 2026 9:00 AM."
        assert limit_reset.parse_limit_reset(text, now) == datetime(2026, 11, 16, 9, 0).timestamp()

    @pytest.mark.parametrize(
        "tail",
        [
            "Try again later.",
            "try again at Foo 16th, 2026 9:00 AM.",
            "try again at Feb 30th, 2026 9:00 AM.",
            "try again at Nov 14th, 2026 9:00 AM.",  # already past
        ],
    )
    def test_anything_else_has_no_time(self, tail):
        now = datetime(2026, 11, 15, 12, 0).astimezone()
        assert limit_reset.parse_limit_reset(f"You've hit your usage limit. {tail}", now) is None

    def test_the_raised_codex_error_carries_the_time(self, clock):
        exc = _raised(
            {
                "code": -32603,
                "message": "Internal error",
                "data": {"message": "You've hit your usage limit. Try again at 11:59 PM."},
            },
            backend=ACP_BACKEND_CODEX,
        )
        assert exc.usage_limit is True
        assert exc.usage_limit_reset_at is not None
        assert exc.usage_limit_harness == ACP_BACKEND_CODEX
