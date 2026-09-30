"""One aimock server per session, for the `mock` suite.

aimock (https://aimock.copilotkit.dev) is a local mock of the LLM wires: it answers
`POST /v1/chat/completions` in OpenAI's shape and `POST /v1/messages` in Anthropic's,
non-streaming and as server-sent events, from JSON fixtures that match on the last
user message. What franca gets from it is the part of the live path the cassettes
cannot exercise -- a real `HttpxTransport`, real headers, real chunked SSE -- with
none of what makes the contract suite expensive: no key, no bill, and a reply whose
every byte was decided in `fixtures/chat.json`.

The server is found one of two ways. With `AIMOCK_BASE_URL` set, an already-running
instance is used as-is -- `make aimock` starts one, reloading on every fixture edit,
which is the shape for poking at fixtures interactively. Otherwise the fixture spawns one
through `npx` on a free port, pinned to `AIMOCK_VERSION`, waits for `/health`, and
stops it when the session ends. The spawned server is started with `AIMOCK_API_KEYS`,
so it *rejects* a request whose key is missing or wrong -- that is how the suite proves
`Connector.headers()` put the key in the header the endpoint row names, rather than
merely that it sent one somewhere.

Nothing here fails for want of a prerequisite. No `npx` on the machine, a server that
never answers `/health`, an `AIMOCK_BASE_URL` that is down: each is a skip with a
reason `-ra` prints, in the same spirit as a contract test skipping on a missing key.
The suite is deselected by default (`-m "not contract and not mock"` in `addopts`),
and `make test-mock` selects it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

AIMOCK_VERSION = os.environ.get("AIMOCK_VERSION", "1.42.0")
"""The `@copilotkit/aimock` release to spawn. The Makefile carries the same default."""

API_KEY = "franca-mock-key"
"""The one key the spawned server accepts; `make aimock` sets the same one."""

FIXTURES = Path(__file__).parent / "fixtures"
HOST = "127.0.0.1"
_STARTUP_TIMEOUT_S = 90.0
"""Generous on purpose: the first `npx` run on a machine downloads the package."""
_EXTERNAL_TIMEOUT_S = 5.0
_LOG_TAIL_LINES = 15


@dataclass(frozen=True)
class MockServer:
    """Where the suite's aimock lives and how to talk to it.

    Attributes:
        base_url: Scheme, host and port with no trailing slash, ready to replace an
            `Endpoint.base_url`.
        api_key: The key to send. The spawned server enforces it; an external one
            only does if it was started with `AIMOCK_API_KEYS`.
        spawned: Whether this session started the server, and therefore knows its
            configuration. Tests that depend on key enforcement skip when `False`.
    """

    base_url: str
    api_key: str
    spawned: bool


def _free_port() -> int:
    """Ask the kernel for a port nobody is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        port: int = sock.getsockname()[1]
        return port


def _healthy(http: httpx.Client, base_url: str) -> bool:
    """Whether `/health` answers 200; a connection error is simply `False`."""
    with contextlib.suppress(httpx.HTTPError):
        return http.get(f"{base_url}/health").status_code == 200
    return False


def _wait_for_health(
    base_url: str, *, timeout_s: float, proc: subprocess.Popen[bytes] | None = None
) -> str | None:
    """Poll `/health` until it answers or the deadline passes.

    Args:
        base_url: The server to poll.
        timeout_s: How long to keep trying.
        proc: The spawned process, if any, so an early exit is reported as such
            instead of as a timeout.

    Returns:
        `None` once the server is healthy, otherwise the reason it is not.
    """
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=2.0) as http:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return f"aimock exited with status {proc.returncode} before answering /health"
            if _healthy(http, base_url):
                return None
            time.sleep(0.2)
    return f"{base_url}/health did not answer within {timeout_s:.0f}s"


def _log_tail(path: Path) -> str:
    """The last few lines of the server log, for a skip reason worth reading."""
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-_LOG_TAIL_LINES:])


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Stop the server and everything it spawned.

    `npx` runs the package's binary as a child, so signalling `npx` alone can leave the
    node process listening on the port. The server was started in its own session, so
    on POSIX the whole process group is signalled; elsewhere the direct child is.
    """
    if proc.poll() is not None:
        return
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
    else:  # pragma: no cover - Windows, where there is no process group to signal
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - a server that ignores SIGTERM
        if hasattr(os, "killpg"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def _aimock(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MockServer]:
    """The aimock server for this session: reused from `AIMOCK_BASE_URL`, else spawned.

    Private, and the three fixtures below are what a test asks for. `MockServer` would
    otherwise have to be imported from a conftest, which pytest's `importlib` import
    mode leaves outside any package -- and a second `conftest.py` anywhere in the tree
    would then collide under that one module name in mypy. Handing over `str` and
    `bool` costs nothing and cannot break that way.

    Yields:
        The server's address, the key to send, and whether this session started it.
    """
    external = os.environ.get("AIMOCK_BASE_URL")
    if external:
        base_url = external.rstrip("/")
        reason = _wait_for_health(base_url, timeout_s=_EXTERNAL_TIMEOUT_S)
        if reason is not None:
            pytest.skip(f"AIMOCK_BASE_URL is set but unusable: {reason}")
        yield MockServer(
            base_url=base_url,
            api_key=os.environ.get("AIMOCK_API_KEY", API_KEY),
            spawned=False,
        )
        return

    npx = shutil.which("npx")
    if npx is None:
        pytest.skip("npx is not on PATH; the mock suite spawns aimock through it")

    port = _free_port()
    base_url = f"http://{HOST}:{port}"
    workdir = tmp_path_factory.mktemp("aimock")
    log_path = workdir / "aimock.log"
    command = [
        npx,
        "--yes",
        "--package",
        f"@copilotkit/aimock@{AIMOCK_VERSION}",
        "llmock",
        "--port",
        str(port),
        "--host",
        HOST,
        "--fixtures",
        str(FIXTURES),
        "--log-level",
        "warn",
    ]
    with log_path.open("wb") as log:
        # S603: every argument above is a literal or a path this file computed; nothing
        # comes from a request or a user. The working directory is a temp dir so any
        # file aimock writes lands there and not in the repository.
        proc = subprocess.Popen(  # noqa: S603
            command,
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "AIMOCK_API_KEYS": API_KEY},
            start_new_session=True,
        )
    try:
        reason = _wait_for_health(base_url, timeout_s=_STARTUP_TIMEOUT_S, proc=proc)
        if reason is not None:
            _stop(proc)
            tail = _log_tail(log_path)
            pytest.skip(f"could not start aimock {AIMOCK_VERSION}: {reason}\n{tail}".rstrip())
        yield MockServer(base_url=base_url, api_key=API_KEY, spawned=True)
    finally:
        _stop(proc)


@pytest.fixture(scope="session")
def aimock_url(_aimock: MockServer) -> str:
    """The mock's base URL, ready to replace an `Endpoint.base_url`."""
    return _aimock.base_url


@pytest.fixture(scope="session")
def aimock_key(_aimock: MockServer) -> str:
    """The API key the mock expects."""
    return _aimock.api_key


@pytest.fixture(scope="session")
def aimock_spawned(_aimock: MockServer) -> bool:
    """Whether this session started the server, and so knows how it was configured."""
    return _aimock.spawned
