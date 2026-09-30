"""Test doubles for code that calls models: franca's own kit, offered as public API.

Everything a test needs in order to exercise a `Model` without a socket, a key or a
real second passing is here, in one import: a `ScriptedTransport` that replays a
`Cassette` of recorded exchanges, a `StaticKeyProvider` that hands out fake keys, and
a `FakeClock` that answers time questions from a counter instead of the machine.
Injected together they make a call deterministic end to end -- same bytes, same
latency, same backoff -- which is what turns an adapter regression into a diff rather
than a flake.

This package is public API. franca's own suite uses it, but so do plugin authors
writing adapters out of tree and users testing an application built on franca, and
their expectations are the reason for its one hard constraint: **stdlib and franca
only, and never pytest**. A plugin author may run these doubles under `unittest`,
`nose`-style runners or a plain script, so importing a test framework here would
either break them outright or force pytest into their production dependency set. The
rule is not merely documented -- `tests/core/test_testing_helpers.py` reads the
source of this module and of every franca module it transitively imports, and fails
if any of them imports pytest. Pytest-specific glue (fixtures, markers, the
`--update-golden` option) belongs in franca's own `tests/conftest.py`.

M0 ships the three doubles above. Later milestones add the recording side
(`RecordingTransport`, `recording()`, secret scrubbing) with SSE replay in M1, the
tracer doubles (`NullTracer`, `RecordingTracer`) with the middleware clients, and
`packages()`, `run_conformance()` and `offline_registry()` once the registry and the
conformance suites exist. Each will appear in `__all__` as it lands.
"""

from franca.core.keys import StaticKeyProvider
from franca.transports.scripted import Cassette, Exchange, ScriptedTransport

__all__ = [
    "Cassette",
    "Exchange",
    "FakeClock",
    "ScriptedTransport",
    "StaticKeyProvider",
]


class FakeClock:
    """A `Clock` that never waits: readings come from a counter, sleeps are recorded.

    Satisfies `franca.core.clock.Clock` structurally, so it is injected wherever
    `AsyncioClock` would go and nothing above it can tell the difference. Three
    behaviours matter, and each exists for a specific test:

    * `monotonic()` returns the current reading and *then* advances it by `step`. A
      `Model.complete` that stamps `t0 = clock.monotonic()` before the call and
      `t1 = clock.monotonic()` after it therefore observes a latency of exactly
      `step` seconds -- deterministic, and non-zero when the test wants a non-zero
      one. Leave `step` at its default of `0.0` and time stands perfectly still.
    * `sleep(seconds)` does not sleep. It appends `seconds` to `slept` and advances
      the monotonic reading by that much, so a retry loop's whole backoff schedule
      runs in microseconds and is then asserted on as a list of numbers rather than
      by measuring elapsed wall time.
    * `wall()` is a fixed value. It exists only so that an HTTP-date `Retry-After`
      can be turned into a delay; pinning it makes that arithmetic reproducible.

    `advance(seconds)` moves the monotonic reading directly, for tests that want to
    cross a deadline without going through a sleep.

    Not thread-safe and not meant to be: it is a counter behind three methods.

    Attributes:
        slept: Every duration passed to `sleep`, in call order. Assert on this
            instead of on elapsed time -- it is exact, and it is the record of what
            the code under test *intended* to wait for.
    """

    def __init__(
        self, *, monotonic: float = 0.0, wall: float = 1_000_000.0, step: float = 0.0
    ) -> None:
        """Start the counters, with nothing slept yet.

        Args:
            monotonic: The first value `monotonic()` returns.
            wall: The value `wall()` returns, for the lifetime of the clock.
            step: How far `monotonic()` advances the reading after each call. `0.0`
                freezes time; a positive value gives every measured interval a
                known, non-zero duration.
        """
        self._monotonic = monotonic
        self._wall = wall
        self._step = step
        self.slept: list[float] = []

    def monotonic(self) -> float:
        """Return the current monotonic reading, then advance it by `step`.

        Returns:
            The reading as it stood before this call.
        """
        now = self._monotonic
        self._monotonic += self._step
        return now

    def wall(self) -> float:
        """Return the fixed wall-clock reading; nothing ever moves it.

        Returns:
            The `wall` value the clock was constructed with.
        """
        return self._wall

    async def sleep(self, seconds: float) -> None:
        """Record `seconds` and advance the monotonic reading by it, without waiting.

        The coroutine completes without yielding to the event loop, so a backoff
        schedule of hours costs nothing to run.

        Args:
            seconds: The duration the caller asked to wait for.
        """
        self.slept.append(seconds)
        self._monotonic += seconds

    def advance(self, seconds: float) -> None:
        """Move the monotonic reading forward by `seconds` without recording a sleep.

        For crossing a deadline directly, when the point of the test is what happens
        after time has passed rather than who asked to wait.

        Args:
            seconds: How far to advance. Negative values are allowed but move a
                reading that is supposed to be monotonic; nothing in franca does it.
        """
        self._monotonic += seconds

    def __repr__(self) -> str:
        """Report the counters and how many sleeps were recorded, never their contents."""
        return (
            f"{type(self).__name__}(monotonic={self._monotonic!r}, wall={self._wall!r}, "
            f"step={self._step!r}, slept={len(self.slept)})"
        )
