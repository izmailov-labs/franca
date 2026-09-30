"""Tests for `franca.core.clock`: the default clock, the protocol seam, and the layering rule.

The layering test is the executable form of the convention in CLAUDE.md and the root
`pyproject.toml`: `asyncio` is imported in `core/clock.py` and nowhere else in `src/`.
It walks the *installed* package (resolved from `franca.__file__`) so that it holds
for whatever tree pytest is actually exercising.
"""

from __future__ import annotations

import ast
import time
import warnings
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import franca
from franca.core.clock import AsyncioClock

if TYPE_CHECKING:
    from franca.core.clock import Clock

# --------------------------------------------------------------------------------------
# AsyncioClock
# --------------------------------------------------------------------------------------


def test_monotonic_is_non_decreasing() -> None:
    clock = AsyncioClock()
    readings = [clock.monotonic() for _ in range(100)]
    assert all(later >= earlier for earlier, later in pairwise(readings))


def test_wall_tracks_time_time() -> None:
    clock = AsyncioClock()
    assert abs(clock.wall() - time.time()) < 5.0


async def test_sleep_zero_returns() -> None:
    clock = AsyncioClock()
    await clock.sleep(0)  # reaching the next line is the assertion: the await completes


async def test_sleep_actually_waits() -> None:
    clock = AsyncioClock()
    before = clock.monotonic()
    await clock.sleep(0.02)
    # asyncio may fire a timer up to one clock resolution early; half the requested
    # duration is a comfortable margin that still proves the call did not return at once.
    assert clock.monotonic() - before >= 0.01


def test_asyncio_clock_satisfies_protocol() -> None:
    clock: Clock = AsyncioClock()
    assert isinstance(clock.monotonic(), float)
    assert isinstance(clock.wall(), float)


# --------------------------------------------------------------------------------------
# Clock as a structural seam
# --------------------------------------------------------------------------------------


class _FixedClock:
    """A `Clock` that never moves and records what it was asked to sleep."""

    def __init__(self, *, monotonic: float, wall: float) -> None:
        """Pin both readings to the given values."""
        self._monotonic = monotonic
        self._wall = wall
        self.slept: list[float] = []

    def monotonic(self) -> float:
        """Return the pinned monotonic reading."""
        return self._monotonic

    def wall(self) -> float:
        """Return the pinned wall reading."""
        return self._wall

    async def sleep(self, seconds: float) -> None:
        """Record the request instead of waiting."""
        self.slept.append(seconds)


async def test_fake_clock_satisfies_protocol_structurally() -> None:
    fake = _FixedClock(monotonic=10.0, wall=1_000_000.0)
    clock: Clock = fake  # mypy --strict checks this assignment; no inheritance involved
    assert clock.monotonic() == 10.0
    assert clock.wall() == 1_000_000.0
    await clock.sleep(2.5)
    await clock.sleep(0.0)
    assert fake.slept == [2.5, 0.0]
    assert clock.monotonic() == 10.0  # sleeping on a fake does not advance it


# --------------------------------------------------------------------------------------
# Layering: asyncio is confined to core/clock.py
# --------------------------------------------------------------------------------------

_CLOCK_MODULE = "core/clock.py"


def _asyncio_imports(tree: ast.AST) -> list[str]:
    """Return every `import asyncio[.x]` / `from asyncio[.x] import` in `tree`, nested included."""
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend(
                f"import {alias.name}"
                for alias in node.names
                if alias.name == "asyncio" or alias.name.startswith("asyncio.")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and (node.module == "asyncio" or node.module.startswith("asyncio."))
        ):
            hits.append(f"from {node.module} import ...")
    return hits


def test_detector_sees_clock_module_import() -> None:
    """Guard against a vacuous layering test: the detector must find asyncio in clock.py."""
    root = Path(franca.__file__).parent
    tree = ast.parse((root / _CLOCK_MODULE).read_text(encoding="utf-8"))
    assert _asyncio_imports(tree) == ["import asyncio"]


def test_detector_ignores_relative_and_unrelated_imports() -> None:
    tree = ast.parse("from . import asyncio\nimport asynciox\nfrom asyncio_ext import y\n")
    assert _asyncio_imports(tree) == []


def test_detector_catches_nested_and_dotted_imports() -> None:
    src = "def f():\n    import asyncio.tasks\n    from asyncio import sleep\n"
    assert _asyncio_imports(ast.parse(src)) == ["import asyncio.tasks", "from asyncio import ..."]


def test_only_clock_module_imports_asyncio() -> None:
    root = Path(franca.__file__).parent
    offenders: dict[str, list[str]] = {}
    skipped: list[str] = []
    seen_clock = False
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == _CLOCK_MODULE:
            seen_clock = True
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            skipped.append(rel)  # a sibling mid-write; skip that file only
            continue
        hits = _asyncio_imports(tree)
        if hits:
            offenders[rel] = hits
    if skipped:
        warnings.warn(f"layering check skipped unparsable modules: {skipped}", stacklevel=2)
    assert seen_clock, f"{_CLOCK_MODULE} missing from the installed package under {root}"
    assert offenders == {}, f"asyncio is allowed only in {_CLOCK_MODULE}; found in {offenders}"
