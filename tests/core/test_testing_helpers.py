"""Tests for `franca.testing`: the `FakeClock` counter and the never-pytest constraint.

Two halves. The first pins `FakeClock`'s arithmetic -- what `monotonic()` returns and
when it moves, that `sleep` records instead of waiting, that `wall` is fixed -- and the
structural claim that it is a `Clock`, which mypy checks on a typed assignment rather
than at runtime.

The second is the layering guard. `franca.testing` is public API for plugin authors who
may run another test framework entirely, so it must never import pytest. Asserting
`"pytest" not in sys.modules` cannot show that: this suite *is* pytest, so the name is
always there. Instead the closure of franca modules that `franca.testing` transitively
imports is computed from the source with `ast`, and every file in it is checked two ways
-- a literal text scan for `import pytest` (which also catches `import pytest_asyncio`)
and an AST scan that additionally catches `from pytest import ...`. A subprocess probe
covers the runtime side, where `sys.modules` is trustworthy because pytest was never
loaded.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import franca
import franca.testing
from franca.core.keys import StaticKeyProvider
from franca.testing import Cassette, Exchange, FakeClock, ScriptedTransport

if TYPE_CHECKING:
    from franca.core.clock import Clock

_PROBE = "import sys\nimport franca.testing\nprint('pytest' in sys.modules)\n"
_ROOT = Path(franca.__file__).parent

# --------------------------------------------------------------------------------------
# FakeClock: monotonic
# --------------------------------------------------------------------------------------


def test_monotonic_starts_at_the_configured_reading() -> None:
    assert FakeClock(monotonic=12.5).monotonic() == 12.5


def test_monotonic_is_constant_when_step_is_zero() -> None:
    clock = FakeClock(monotonic=3.0)
    assert [clock.monotonic() for _ in range(5)] == [3.0] * 5


def test_monotonic_returns_then_advances_by_the_step() -> None:
    clock = FakeClock(step=0.25)
    assert [clock.monotonic() for _ in range(4)] == [0.0, 0.25, 0.5, 0.75]


def test_two_readings_measure_exactly_one_step() -> None:
    """The latency a `Model.complete` would stamp: t1 - t0 == step, deterministically."""
    clock = FakeClock(monotonic=100.0, step=0.25)
    t0 = clock.monotonic()
    t1 = clock.monotonic()
    assert t1 - t0 == 0.25


def test_two_clocks_do_not_share_state() -> None:
    first, second = FakeClock(step=1.0), FakeClock(step=1.0)
    first.monotonic()
    first.monotonic()
    assert second.monotonic() == 0.0


# --------------------------------------------------------------------------------------
# FakeClock: sleep and advance
# --------------------------------------------------------------------------------------


async def test_sleep_records_the_durations_in_call_order() -> None:
    clock = FakeClock()
    await clock.sleep(2.0)
    await clock.sleep(0.0)
    await clock.sleep(8.5)
    assert clock.slept == [2.0, 0.0, 8.5]


async def test_sleep_advances_monotonic_by_the_slept_duration() -> None:
    clock = FakeClock(monotonic=5.0)
    await clock.sleep(30.0)
    await clock.sleep(0.5)
    assert clock.monotonic() == 35.5


async def test_a_long_backoff_schedule_costs_nothing() -> None:
    """An hour of backoff, asserted on the record rather than on elapsed time."""
    clock = FakeClock()
    for delay in (1.0, 2.0, 4.0, 8.0, 3585.0):
        await clock.sleep(delay)
    assert sum(clock.slept) == 3600.0
    assert clock.monotonic() == 3600.0


async def test_sleep_does_not_apply_the_step() -> None:
    """Only a `monotonic()` read steps; a sleep moves time by exactly what was asked."""
    clock = FakeClock(step=1.0)
    await clock.sleep(10.0)
    assert clock.monotonic() == 10.0


def test_slept_starts_empty() -> None:
    assert FakeClock().slept == []


def test_advance_moves_monotonic_without_recording_a_sleep() -> None:
    clock = FakeClock(monotonic=1.0)
    clock.advance(4.0)
    clock.advance(0.5)
    assert clock.monotonic() == 5.5
    assert clock.slept == []


# --------------------------------------------------------------------------------------
# FakeClock: wall, protocol conformance, repr
# --------------------------------------------------------------------------------------


def test_wall_defaults_to_a_fixed_epoch_reading() -> None:
    assert FakeClock().wall() == 1_000_000.0


async def test_wall_never_moves() -> None:
    clock = FakeClock(wall=1_700_000_000.0, step=1.0)
    clock.monotonic()
    await clock.sleep(60.0)
    clock.advance(60.0)
    assert clock.wall() == 1_700_000_000.0


async def test_fake_clock_satisfies_the_clock_protocol() -> None:
    clock: Clock = FakeClock(step=0.5)  # mypy --strict checks this; no inheritance involved
    assert clock.monotonic() == 0.0
    assert clock.wall() == 1_000_000.0
    await clock.sleep(1.0)
    assert clock.monotonic() == 1.5


def test_repr_reports_the_counters_and_a_sleep_count() -> None:
    clock = FakeClock(monotonic=2.0, wall=9.0, step=0.5)
    clock.advance(1.0)
    assert repr(clock) == "FakeClock(monotonic=3.0, wall=9.0, step=0.5, slept=0)"


# --------------------------------------------------------------------------------------
# Re-exports
# --------------------------------------------------------------------------------------


def test_the_re_exports_are_the_objects_from_their_home_modules() -> None:
    from franca.transports import scripted

    assert StaticKeyProvider is franca.testing.StaticKeyProvider
    assert ScriptedTransport is scripted.ScriptedTransport
    assert Cassette is scripted.Cassette
    assert Exchange is scripted.Exchange


def test_all_lists_exactly_the_public_names() -> None:
    assert franca.testing.__all__ == [
        "Cassette",
        "Exchange",
        "FakeClock",
        "ScriptedTransport",
        "StaticKeyProvider",
    ]
    for name in franca.testing.__all__:
        assert hasattr(franca.testing, name)


# --------------------------------------------------------------------------------------
# Never pytest: the source-text guard
# --------------------------------------------------------------------------------------


def _module_path(root: Path, name: str) -> Path | None:
    """Return the file backing the dotted module `name`, or `None` if it is not a franca module."""
    parts = name.split(".")
    if parts[0] != "franca":
        return None
    module = root.joinpath(*parts[1:]).with_suffix(".py")
    if module.is_file():
        return module
    package = root.joinpath(*parts[1:], "__init__.py")
    return package if package.is_file() else None


def _ancestors(name: str) -> list[str]:
    """Return every package that importing `name` also executes, `name` excluded."""
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _package_of(name: str) -> str:
    """Return the package a relative import written inside module `name` resolves against."""
    path = _module_path(_ROOT, name)
    if path is not None and path.name == "__init__.py":
        return name
    return name.rpartition(".")[0] or name


def _franca_imports(tree: ast.AST, module: str) -> set[str]:
    """Return the franca modules `module`'s source imports, relative imports resolved.

    A `from franca.core import keys` names both the package and the submodule, and only
    the ones that exist on disk survive the closure walk, so listing a class here is
    harmless.
    """
    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = _package_of(module).split(".")
                kept = anchor[: len(anchor) - node.level + 1]
                base = ".".join([*kept, base]) if base else ".".join(kept)
            if base:
                hits.add(base)
                hits.update(f"{base}.{alias.name}" for alias in node.names)
    return {hit for hit in hits if hit == "franca" or hit.startswith("franca.")}


def _pytest_imports(tree: ast.AST) -> list[str]:
    """Return every `import pytest[.x]` / `from pytest[.x] import ...` in `tree`."""
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend(
                f"import {alias.name}"
                for alias in node.names
                if alias.name == "pytest" or alias.name.startswith("pytest.")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and (node.module == "pytest" or node.module.startswith("pytest."))
        ):
            hits.append(f"from {node.module} import ...")
    return hits


def _import_closure(start: str) -> dict[str, Path]:
    """Return every franca module reachable from `start` by source-level imports."""
    found: dict[str, Path] = {}
    unparsable: list[str] = []
    queue = [start]
    while queue:
        name = queue.pop()
        if name in found:
            continue
        path = _module_path(_ROOT, name)
        if path is None:
            continue
        found[name] = path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # a sibling module mid-write; its imports stay unexplored
            unparsable.append(name)
            continue
        queue.extend(_franca_imports(tree, name))
        queue.extend(_ancestors(name))
    if unparsable:
        warnings.warn(f"import closure could not parse: {unparsable}", stacklevel=2)
    return found


def test_module_path_resolves_modules_packages_and_nothing_else() -> None:
    assert _module_path(_ROOT, "franca.core.keys") == _ROOT / "core" / "keys.py"
    assert _module_path(_ROOT, "franca.testing") == _ROOT / "testing" / "__init__.py"
    assert _module_path(_ROOT, "franca") == _ROOT / "__init__.py"
    assert _module_path(_ROOT, "franca.core.keys.StaticKeyProvider") is None
    assert _module_path(_ROOT, "pytest") is None


def test_the_import_detector_finds_absolute_and_relative_franca_imports() -> None:
    src = "import franca.core.sse\nfrom franca.core import keys\nfrom . import ir\nimport json\n"
    found = _franca_imports(ast.parse(src), "franca.chat.dialects.anthropic")
    assert found == {
        "franca.core.sse",
        "franca.core",
        "franca.core.keys",
        "franca.chat.dialects",
        "franca.chat.dialects.ir",
    }


def test_the_pytest_detector_catches_every_import_form() -> None:
    src = "import pytest\nfrom pytest import raises\nimport pytest.mark\n"
    assert sorted(_pytest_imports(ast.parse(src))) == [
        "from pytest import ...",
        "import pytest",
        "import pytest.mark",
    ]


def test_the_pytest_detector_ignores_lookalikes() -> None:
    src = "import pytest_asyncio\nfrom pytest_cov import x\nfrom . import pytest\n"
    assert _pytest_imports(ast.parse(src)) == []


def test_the_closure_reaches_the_modules_testing_is_built_on() -> None:
    """Guard against a vacuous layering test: the closure must be non-trivial."""
    closure = _import_closure("franca.testing")
    assert {
        "franca",
        "franca.testing",
        "franca.core",
        "franca.core.keys",
        "franca.transports",
        "franca.transports.scripted",
        "franca.core.transport",
    } <= set(closure)
    assert "franca.transports.httpx" not in closure  # the optional extra stays optional


def test_no_module_in_the_testing_closure_imports_pytest() -> None:
    offenders: dict[str, list[str]] = {}
    for name, path in sorted(_import_closure("franca.testing").items()):
        source = path.read_text(encoding="utf-8")
        hits = ["text: import pytest"] if "import pytest" in source else []
        try:
            hits += _pytest_imports(ast.parse(source, filename=str(path)))
        except SyntaxError:  # pragma: no cover - only a sibling mid-write reaches this
            hits.append("unparsable")
        if hits:
            offenders[name] = hits
    assert offenders == {}, f"franca.testing must never import pytest; found {offenders}"


def test_importing_franca_testing_does_not_load_pytest_at_runtime() -> None:
    """The runtime half, in a subprocess where `sys.modules` is actually trustworthy."""
    probe = subprocess.run(  # noqa: S603 - fixed argv, this interpreter, no shell
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=120,
    )
    assert probe.stdout.strip() == "False"
