"""Progress reporting for the long single-threaded stages of a mine.

A stage that prints only when it finishes is indistinguishable from a hang, and the
stages here run for hours: promotion re-certifies every rule, the covered-prune replays
waves over the whole set. Both scale worse than linearly in the rule count, so "it was
fast last time" is not evidence about this time.

The reporter is therefore TIME-driven, not count-driven. A heartbeat every
:data:`DEFAULT_INTERVAL` seconds means the gap between two lines is bounded by the clock
regardless of how slow an individual item is -- a per-item counter cannot promise that,
because one pathological rule can stall a count-based tick indefinitely.

Every line is flushed. A run redirected to a file that buffers its output is a run whose
progress cannot be read while it matters.
"""
import sys
import time
from typing import Any, Callable, Iterable, Iterator

#: Seconds between heartbeats inside a stage.
DEFAULT_INTERVAL = 30.0


def _duration(seconds: float) -> str:
    """``93784`` -> ``'1d 2h 03m'``; short spans keep seconds."""
    seconds = max(0.0, float(seconds))
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class Progress:
    """Emit stage banners and periodic heartbeats for long-running work.

    Disabled instances are inert: :meth:`track` yields the iterable unchanged and every
    other method returns immediately, so instrumentation costs nothing when off.
    """

    def __init__(self, enabled: bool = False, *, stream: Any = None,
                 interval: float = DEFAULT_INTERVAL, prefix: str = "") -> None:
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stdout
        self.interval = float(interval)
        self.prefix = prefix
        self._started = time.monotonic()

    def write(self, message: str) -> None:
        """One flushed line. Flushing is the point: see the module docstring."""
        if not self.enabled:
            return
        self.stream.write(f"{self.prefix}{message}\n")
        self.stream.flush()

    def track(self, iterable: Iterable, name: str, total: "int | None" = None,
              detail: "Callable[[Any], str] | None" = None) -> Iterator:
        """Wrap ``iterable``, announcing the stage and heartbeating while it runs.

        ``total`` is taken from ``len(iterable)`` when the object supports it, so a plain
        list needs no count. ``detail`` may map the most recent item to a short string
        appended to each heartbeat -- what the stage is chewing on right now, which is the
        difference between "still alive" and "still alive, and here is where".
        """
        if not self.enabled:
            yield from iterable
            return
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None

        if total == 0:
            # An empty stage is worth one line, not a banner and a completion.
            self.write(f"{name}: nothing to do")
            return

        start = time.monotonic()
        self.write(f"{name}: {total if total is not None else '?'} items")
        last = start
        done = 0
        latest: Any = None
        for item in iterable:
            latest = item
            yield item
            done += 1
            now = time.monotonic()
            if now - last >= self.interval:
                self.write(f"  {name}: {self._line(done, total, start, now)}"
                           + (f"  |  {detail(latest)}" if detail is not None else ""))
                last = now
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        self.write(f"{name}: done -- {done} items in {_duration(elapsed)} ({rate:.1f}/s)")

    @staticmethod
    def _line(done: int, total: "int | None", start: float, now: float) -> str:
        elapsed = now - start
        rate = done / elapsed if elapsed > 0 else 0.0
        if total:
            pct = 100.0 * done / total
            # ETA from the rate SO FAR. Honest for a uniform stage and optimistic for one
            # that slows down, which is why the elapsed time is printed beside it rather
            # than replaced by it.
            eta = (total - done) / rate if rate > 0 else float("inf")
            eta_text = _duration(eta) if eta != float("inf") else "unknown"
            return (f"{done}/{total} ({pct:.1f}%)  {_duration(elapsed)} elapsed  "
                    f"{rate:.1f}/s  ETA {eta_text}")
        return f"{done} items  {_duration(elapsed)} elapsed  {rate:.1f}/s"

    def stage(self, name: str, **counts: Any) -> None:
        """A one-line stage result, e.g. ``promoted=12 demoted=3``."""
        if counts:
            body = "  ".join(f"{k}={v}" for k, v in counts.items())
            self.write(f"{name}: {body}")
        else:
            self.write(name)

    def total_elapsed(self) -> str:
        return _duration(time.monotonic() - self._started)
