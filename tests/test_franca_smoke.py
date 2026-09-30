"""Scaffolding smoke tests: packaging metadata and the async test plumbing."""

import asyncio

import franca


def test_version_is_exposed() -> None:
    assert isinstance(franca.__version__, str)
    assert franca.__version__


def test_public_api_is_declared() -> None:
    assert "__version__" in franca.__all__


async def test_async_tests_actually_run() -> None:
    """Prove pytest-asyncio auto mode is wired up before any real async code lands."""
    await asyncio.sleep(0)
    assert asyncio.get_running_loop().is_running()
