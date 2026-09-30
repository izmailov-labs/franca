"""Time and sleep, injected: the one seam through which franca touches the event loop.

Everything in franca that needs the passage of time -- request latency, retry
backoff, job polling, deadline arithmetic -- asks a `Clock` for it rather than
calling `time` or `asyncio` directly. That indirection buys two things at once.

Loop neutrality. franca is asyncio-declared but never imports `asyncio` outside this
module (ruff's TID251 banned-api rule enforces it across `src/`). The `Clock`
protocol is where the event loop is abstracted away: a trio user passes their own
`TrioClock` whose `sleep` awaits `trio.sleep`, and nothing else in the library has
to know. Deadlines are `clock.monotonic()` arithmetic, so no `asyncio.timeout`, task
group or lock is needed in core -- a total deadline belongs to the caller's
`asyncio.timeout` or `trio.move_on_after`.

Testability. Retry and poll loops are deterministic once the clock is a fake: tests
advance time by hand instead of actually waiting, and latency measurements become
exact.

Three methods, each for a distinct reason:

* `monotonic` -- durations: `latency_ms`, backoff deadlines, poll budgets. Immune to
  wall-clock jumps (NTP slew, DST, a suspended laptop), which is what makes it safe
  to subtract two readings.
* `wall` -- `time.time()`. Exists for exactly one reason: an HTTP `Retry-After`
  header may carry an HTTP-date instead of a delay in seconds, and turning that date
  into a delay means subtracting the current wall time. Nothing else should read it.
* `sleep` -- retry backoff and job polling. The only place franca yields to the loop
  for a duration, hence the only method that is `async`.
"""

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    """Structural interface for the three time operations franca performs.

    Any object with these three methods satisfies it; no inheritance is required.
    `AsyncioClock` is the default. A test double returns fixed values and records
    the durations it was asked to sleep; a `TrioClock` awaits `trio.sleep` instead.
    This protocol -- together with the injected `Transport` -- is what keeps franca
    loop-neutral without depending on anyio.
    """

    def monotonic(self) -> float:
        """Return a monotonic reading in seconds, for measuring durations.

        Two readings may be subtracted to obtain an elapsed time; the absolute
        value is meaningless. Never affected by wall-clock adjustments.
        """

    def wall(self) -> float:
        """Return the current wall-clock time as seconds since the Unix epoch.

        Used solely to convert an HTTP-date `Retry-After` header into a delay.
        Not suitable for measuring durations: it can jump backwards.
        """

    async def sleep(self, seconds: float) -> None:
        """Suspend the current task for at least `seconds`, yielding to the loop.

        Used for retry backoff and job polling. Implementations must await the
        loop's own sleep primitive (`asyncio.sleep`, `trio.sleep`), never block.
        """


class AsyncioClock:
    """The default `Clock`: stdlib `time` for readings, `asyncio.sleep` for waiting.

    This is the only class in `franca` that touches `asyncio`; every other module
    receives a `Clock` by injection so that the library runs unchanged under trio.
    """

    def monotonic(self) -> float:
        """Return `time.monotonic()`."""
        return time.monotonic()

    def wall(self) -> float:
        """Return `time.time()`."""
        return time.time()

    async def sleep(self, seconds: float) -> None:
        """Await `asyncio.sleep(seconds)`."""
        await asyncio.sleep(seconds)
