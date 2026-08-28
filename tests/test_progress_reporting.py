"""Long stages must report, and reporting must not change what they compute.

A stage that prints only on completion cannot be told from a hang, and the stages these
cover run for hours. Two properties matter and both are easy to lose silently: the
reporter is OFF by default (so a library call stays quiet), and turning it on changes
nothing but the output.
"""
import io
import re
import sys
import time

import pytest

from simplipy import SimpliPyEngine
from simplipy.progress import DEFAULT_INTERVAL, Progress, _duration
from simplipy.promotion import promote

RULES = [(("+", "_0", "0"), ("_0",)), (("*", "_0", "1"), ("_0",)),
         (("*", "_0", "0"), ("0",)), (("cos", "0"), ("1",)),
         (("+", "_0", "_0"), ("*", "2", "_0")), (("neg", "neg", "_0"), ("_0",))]


@pytest.fixture(scope="module")
def engine():
    return SimpliPyEngine.load("base", install=True)


def _capture(fn):
    buf = io.StringIO()
    saved, sys.stdout = sys.stdout, buf
    try:
        result = fn()
    finally:
        sys.stdout = saved
    return result, buf.getvalue()


def test_duration_reads_at_every_scale() -> None:
    assert _duration(45) == "45s"
    assert _duration(605) == "10m 05s"
    assert _duration(3725) == "1h 02m 05s"
    assert _duration(93784) == "1d 2h 03m"
    assert _duration(-1) == "0s"          # a clock that went backwards is not an exception


def test_disabled_progress_is_inert() -> None:
    prog = Progress(False)
    (out, text) = _capture(lambda: list(prog.track(range(4), "stage")))
    assert out == [0, 1, 2, 3]
    assert text == ""


def test_the_heartbeat_is_driven_by_the_CLOCK_not_the_count() -> None:
    """The property that makes this useful: one pathological item cannot stall the
    report. A count-based tick every N items says nothing while item N+1 runs for an
    hour; a clock-based one keeps printing."""
    prog = Progress(True, stream=io.StringIO(), interval=0.05)

    def slow():
        yield 1
        time.sleep(0.12)     # one slow item, well past the interval
        yield 2

    list(prog.track(slow(), "stage", total=2))
    lines = prog.stream.getvalue().splitlines()
    beats = [line for line in lines if re.search(r"\d/2 \(", line)]
    assert beats, f"no heartbeat emitted across a slow item: {lines}"


def test_a_tracked_stage_reports_progress_rate_and_eta() -> None:
    prog = Progress(True, stream=io.StringIO(), interval=0.0)
    list(prog.track(range(4), "stage"))
    text = prog.stream.getvalue()
    assert "stage: 4 items" in text
    assert re.search(r"stage: 2/4 \(50\.0%\)\s+\S+ elapsed\s+[\d.]+/s\s+ETA ", text)
    assert re.search(r"stage: done -- 4 items in \S+ \([\d.]+/s\)", text)


def test_an_empty_stage_costs_one_line() -> None:
    prog = Progress(True, stream=io.StringIO(), interval=0.0)
    list(prog.track([], "stage"))
    assert prog.stream.getvalue().strip() == "stage: nothing to do"


def test_promotion_is_silent_by_default(engine) -> None:
    (result, text) = _capture(lambda: promote(list(RULES), engine, run_positive_controls=False))
    assert text == "", f"promote() printed without being asked: {text!r}"
    assert result[0], "and it must still have promoted something"


def test_promotion_reports_every_stage_when_verbose(engine) -> None:
    (_, text) = _capture(
        lambda: promote(list(RULES), engine, run_positive_controls=False, verbose=True))
    # Each of the five stages announces itself, so a stall is attributable to one of them.
    for stage in ("stage 1/5", "stage 2/5", "stage 3/5", "stage 4/5", "stage 5/5"):
        assert stage in text, f"{stage} never reported:\n{text}"
    # The ladder's tiers are the long pole and are named individually.
    for tier in ("tier _cf", "tier _cb", "tier ?cf", "tier ?cb", "tier ground"):
        assert tier in text, f"{tier} never reported:\n{text}"
    assert "sort promotion complete" in text


def test_reporting_does_not_change_the_result(engine) -> None:
    (quiet, _) = _capture(lambda: promote(list(RULES), engine, run_positive_controls=False))
    (loud, _) = _capture(
        lambda: promote(list(RULES), engine, run_positive_controls=False,
                        progress=Progress(True, stream=io.StringIO(), interval=0.0)))
    assert quiet[0] == loud[0]
    assert quiet[1]["stage_counts"] == loud[1]["stage_counts"]


def test_the_default_interval_is_short_enough_to_be_useful() -> None:
    """A heartbeat slower than a coffee break is not a heartbeat."""
    assert 0 < DEFAULT_INTERVAL <= 60.0
