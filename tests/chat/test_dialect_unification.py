"""One package, every dialect: the claim the README makes, pinned as a test.

`README.md` shows a single `PromptPackage` rendered as three different wire bodies --
the system prompt in three places, the output budget under three names, the model id in
the body twice and in the path once. That is the whole product claim, and until now
nothing offline held the adapters to it: the only test driving more than one adapter is
`tests/mock`, which is deselected by default and needs node.

So these tests assert the two halves of "same request, every wire" that can be checked
without a network. First, that no dialect *drops* anything: every adapter renders the
user turn, the system instruction and the output budget, each under its own spelling.
Second, that each dialect's spelling is the one its provider actually documents -- the
divergence is the feature, so a rename here is a behaviour change and should fail.

The response half of the claim is pinned over real HTTP in `tests/mock`, because
asserting it offline would mean inventing provider payloads, and an invented payload
tests the fixture rather than the wire.
"""

from typing import Final, NamedTuple

import pytest

from franca.chat.adapter import DialectAdapter
from franca.chat.dialects.anthropic_messages import AnthropicMessagesAdapter
from franca.chat.dialects.google_generate_content import GoogleGenerateContentAdapter
from franca.chat.dialects.openai_chat import OpenAIChatAdapter
from franca.chat.ir import Item, PromptPackage, SystemBlock
from franca.chat.profile import ChatProfile
from franca.core.ids import ANTHROPIC, GOOGLE, OPENAI, Provider

SYSTEM_TEXT: Final = "You are terse."
USER_TEXT: Final = "Name one primary colour."
BUDGET: Final = 32

PACKAGE: Final = PromptPackage(
    system=(SystemBlock(text=SYSTEM_TEXT),),
    items=(Item(role="user", kind="text", text=USER_TEXT),),
    max_output_tokens=BUDGET,
)


class Case(NamedTuple):
    """One dialect, and where that wire puts each part of the package."""

    dialect: str
    adapter: DialectAdapter
    provider: Provider
    model: str
    budget_key: str
    model_in_path: bool


CASES: Final = (
    Case(
        "anthropic_messages",
        AnthropicMessagesAdapter(),
        ANTHROPIC,
        "claude-sonnet-5",
        "max_tokens",
        model_in_path=False,
    ),
    Case(
        "openai_chat",
        OpenAIChatAdapter(),
        OPENAI,
        "gpt-6-astra",
        "max_completion_tokens",
        model_in_path=False,
    ),
    Case(
        "google_generate_content",
        GoogleGenerateContentAdapter(),
        GOOGLE,
        "gemini-3.8-flash",
        "maxOutputTokens",
        model_in_path=True,
    ),
)
IDS: Final = [case.dialect for case in CASES]


def _render(case: Case) -> dict[str, object]:
    """Render `PACKAGE` through one dialect with an empty profile.

    `WireRequest.body` is optional because a wire may carry raw content instead; every
    chat dialect sends JSON, so a `None` here is a bug rather than a case to handle.
    """
    profile = ChatProfile(provider=case.provider, model_prefix="")
    body = case.adapter.to_request(PACKAGE, case.model, profile).body
    assert body is not None, f"{case.dialect} rendered no JSON body"
    return dict(body)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_dialect_carries_the_whole_package(case: Case) -> None:
    """Nothing is silently dropped: the prompt, the system text and the budget all ship.

    Serialised and searched as one string on purpose. *Where* each wire puts these is
    the next test's business; this one only refuses a dialect that loses one.
    """
    rendered = repr(_render(case))

    assert USER_TEXT in rendered
    assert SYSTEM_TEXT in rendered
    assert str(BUDGET) in rendered


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_output_budget_uses_each_wires_own_name(case: Case) -> None:
    """One IR field, three keys. A rename is a wire change and has to fail here."""
    body = _render(case)
    generation_config = body.get("generationConfig")
    scope = generation_config if isinstance(generation_config, dict) else body

    assert scope[case.budget_key] == BUDGET


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_model_id_rides_where_that_wire_expects_it(case: Case) -> None:
    """Google puts the model in the path and rejects it in the body; the others invert."""
    profile = ChatProfile(provider=case.provider, model_prefix="")
    wire = case.adapter.to_request(PACKAGE, case.model, profile)

    body = _render(case)
    if case.model_in_path:
        assert wire.path is not None
        assert case.model in wire.path
        assert "model" not in body
    else:
        assert body["model"] == case.model


def test_no_two_dialects_render_the_same_body() -> None:
    """The guard on all of the above: identical bodies would pass every assertion.

    If a future adapter were wired to the wrong dialect, the tests above could still
    pass while two rows rendered byte-identical requests. They must not.
    """
    bodies = [repr(_render(case)) for case in CASES]

    assert len(set(bodies)) == len(CASES)
