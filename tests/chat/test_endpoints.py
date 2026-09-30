"""The built-in chat endpoint rows, held to the invariants a row cannot state itself.

Every row here is data, which is the point: adding a provider is adding one. The cost of
that design is that nothing type-checks a row against the rest of the table -- a copied
`id`, a provider with no profile to resolve against, a `base_url` with a trailing slash
meeting a `path` with a leading one -- so those invariants are asserted here instead.

The table is discovered from the module rather than listed, so a row added without a
thought for any of this fails these tests rather than quietly sitting outside them.
"""

from typing import Final

import pytest

from franca.chat import endpoints as endpoints_module
from franca.chat.profiles import CHAT_PROFILES
from franca.core.endpoint import Endpoint
from franca.core.enums import AuthScheme
from franca.core.ids import (
    ANTHROPIC,
    ANTHROPIC_MESSAGES,
    CHAT,
    DEEPSEEK,
    GOOGLE,
    GOOGLE_GENERATE_CONTENT,
    OPENAI,
    OPENAI_CHAT,
    XAI,
)

ROWS: Final[tuple[Endpoint, ...]] = tuple(
    value
    for name in sorted(dir(endpoints_module))
    if name.endswith("_ENDPOINT") and isinstance(value := getattr(endpoints_module, name), Endpoint)
)
IDS: Final = [row.id for row in ROWS]


def test_the_table_is_the_five_rows_this_suite_expects() -> None:
    """A guard on the discovery above: a silently empty table would pass everything."""
    assert set(IDS) == {
        "anthropic/chat/messages",
        "openai/chat/completions",
        "google/chat/generate_content",
        "xai/chat/completions",
        "deepseek/chat/completions",
    }


def test_every_id_is_unique() -> None:
    """`id` is the trace key and the cassette directory name, so a collision merges two."""
    assert len(IDS) == len(set(IDS))


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_an_id_is_provider_capability_name(row: Endpoint) -> None:
    provider, capability, name = row.id.split("/")
    assert provider == row.provider
    assert capability == row.capability
    assert name


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_every_row_serves_the_chat_capability(row: Endpoint) -> None:
    assert row.capability == CHAT


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_a_url_joins_without_a_doubled_or_missing_slash(row: Endpoint) -> None:
    """`Endpoint.url` concatenates and normalises nothing, so the row owns the boundary."""
    assert row.base_url.startswith("https://")
    assert not row.base_url.endswith("/")
    assert row.path.startswith("/")
    assert "//" not in row.url().removeprefix("https://")


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_every_provider_can_resolve_a_profile(row: Endpoint) -> None:
    """A row whose provider has no default row would raise on the first call it served."""
    profile = CHAT_PROFILES.resolve(row.provider, "some-unmeasured-model")

    assert profile.provider == row.provider


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_a_row_survives_a_json_round_trip(row: Endpoint) -> None:
    """Rows are configuration as much as code; one must serialise without losing a field."""
    assert Endpoint.model_validate_json(row.model_dump_json()) == row


def test_the_three_auth_schemes_are_all_exercised_by_the_table() -> None:
    """Why `auth` is an enum on the row: one `Connector`, three header names."""
    assert {row.auth for row in ROWS} == {
        AuthScheme.bearer,
        AuthScheme.x_api_key,
        AuthScheme.x_goog_api_key,
    }


def test_a_dialect_is_shared_across_providers_but_a_provider_is_not_shared() -> None:
    """The claim the README makes about compatibility surfaces, as an assertion.

    Three providers speak `openai_chat`, so the dialect axis is genuinely independent of
    the provider axis. No provider appears twice, because a second surface for the same
    provider would be a second row this table does not yet have.
    """
    by_dialect = {row.dialect for row in ROWS}
    assert by_dialect == {ANTHROPIC_MESSAGES, OPENAI_CHAT, GOOGLE_GENERATE_CONTENT}

    openai_wire = {row.provider for row in ROWS if row.dialect == OPENAI_CHAT}
    assert openai_wire == {OPENAI, XAI, DEEPSEEK}

    providers = [row.provider for row in ROWS]
    assert len(providers) == len(set(providers))
    assert set(providers) == {ANTHROPIC, OPENAI, GOOGLE, XAI, DEEPSEEK}


def test_only_the_google_row_leaves_its_path_to_the_adapter() -> None:
    """Gemini carries the model in the path, so its row cannot name the path it serves.

    Stated as a test because it looks like a bug: `/v1beta/models` is never the URL a
    completion is sent to. Every other row's `path` is exactly what goes on the wire.
    """
    google = next(row for row in ROWS if row.provider == GOOGLE)
    assert google.path == "/v1beta/models"

    others = [row for row in ROWS if row.provider != GOOGLE]
    assert all(row.path.endswith(("/messages", "/chat/completions")) for row in others)
