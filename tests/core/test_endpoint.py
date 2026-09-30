"""Executable specification for `franca.core.endpoint`.

Pure data: `Endpoint` rows, tri-state `EndpointFeatures`, and the `Requirements` /
`Selection` contracts that dialect selection (a later milestone) consumes and produces.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from franca.core.endpoint import Endpoint, EndpointFeatures, Requirements, Selection
from franca.core.enums import AuthScheme
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES, CHAT, Dialect, Provider

BASE_URL = "https://api.anthropic.com"
PATH = "/v1/messages"


def _endpoint(**overrides: Any) -> Endpoint:
    """Build the canonical Anthropic messages row, with any field overridden."""
    kwargs: dict[str, Any] = {
        "id": "anthropic/chat/messages",
        "provider": ANTHROPIC,
        "capability": CHAT,
        "dialect": ANTHROPIC_MESSAGES,
        "base_url": BASE_URL,
        "path": PATH,
        "auth": AuthScheme.x_api_key,
        "extra_headers": {"anthropic-version": "2023-06-01"},
    }
    kwargs.update(overrides)
    return Endpoint(**kwargs)


# --------------------------------------------------------------------------------------
# Endpoint.url
# --------------------------------------------------------------------------------------


def test_url_defaults_to_own_path() -> None:
    assert _endpoint().url() == BASE_URL + PATH


def test_url_honours_override_path() -> None:
    assert _endpoint().url("/v1/complete") == BASE_URL + "/v1/complete"


def test_url_treats_empty_string_as_an_override_not_none() -> None:
    assert _endpoint().url("") == BASE_URL


def test_url_is_plain_concatenation_without_normalisation() -> None:
    endpoint = _endpoint(base_url="https://example.test/", path="/v1")
    assert endpoint.url() == "https://example.test//v1"


# --------------------------------------------------------------------------------------
# EndpointFeatures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (None, None)],
)
def test_get_reads_all_three_states_of_a_declared_field(
    value: bool | None, expected: bool | None
) -> None:
    assert EndpointFeatures(strict=value).get("strict") is expected


def test_declared_features_default_to_unverified() -> None:
    features = EndpointFeatures()
    for name in ("strict", "server_state", "artifacts", "caching", "hoisting", "streaming"):
        assert features.get(name) is None


def test_production_defaults_true() -> None:
    features = EndpointFeatures()
    assert features.production is True
    assert features.get("production") is True


def test_extra_feature_set_via_kwargs_is_reachable_through_get() -> None:
    extras: dict[str, Any] = {"batch": True, "vision": False}
    features = EndpointFeatures(**extras)
    assert features.get("batch") is True
    assert features.get("vision") is False


def test_unknown_feature_name_reads_as_none() -> None:
    assert EndpointFeatures().get("no_such_feature") is None


@pytest.mark.parametrize("junk", ["yes", 1, 0, [True], {"on": True}])
def test_non_bool_extra_reads_as_none(junk: object) -> None:
    features = EndpointFeatures.model_validate({"batch": junk})
    assert features.get("batch") is None


def test_extra_set_to_none_reads_as_none() -> None:
    features = EndpointFeatures.model_validate({"batch": None})
    assert features.get("batch") is None


def test_declared_field_wins_over_extras_lookup() -> None:
    # A declared field can never be shadowed by model_extra: pydantic routes a known
    # name to the field, so get() sees the validated value.
    features = EndpointFeatures.model_validate({"strict": True})
    assert features.get("strict") is True
    assert not features.model_extra


# --------------------------------------------------------------------------------------
# Endpoint validation
# --------------------------------------------------------------------------------------


def test_sibling_construction_with_raw_slugs_validates() -> None:
    # The exact row `test_connector.py` builds, fed as raw strings.
    endpoint = Endpoint.model_validate(
        {
            "id": "anthropic/chat/messages",
            "provider": "anthropic",
            "capability": "chat",
            "dialect": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "path": "/v1/messages",
            "auth": AuthScheme.x_api_key,
            "extra_headers": {"anthropic-version": "2023-06-01"},
        }
    )
    assert endpoint.provider == "anthropic"
    assert endpoint.capability == "chat"
    assert endpoint.dialect == "anthropic_messages"
    assert endpoint.auth is AuthScheme.x_api_key


def test_dialect_anthropic_messages_validates() -> None:
    assert _endpoint(dialect=Dialect("anthropic_messages")).dialect == ANTHROPIC_MESSAGES


def test_provider_with_hyphen_is_rejected_at_provider_loc() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _endpoint(provider=Provider("my-shim"))
    assert any("provider" in error["loc"] for error in exc_info.value.errors())


@pytest.mark.parametrize("field", ["capability", "dialect"])
def test_other_slug_fields_are_validated_too(field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _endpoint(**{field: "Not.A.Slug"})
    assert any(field in error["loc"] for error in exc_info.value.errors())


def test_auth_accepts_scheme_by_value() -> None:
    endpoint = Endpoint.model_validate(
        {**_endpoint().model_dump(), "auth": "bearer"},
    )
    assert endpoint.auth is AuthScheme.bearer


def test_extra_headers_default_is_a_fresh_dict_per_instance() -> None:
    first = Endpoint.model_validate({**_endpoint().model_dump(exclude={"extra_headers"})})
    second = Endpoint.model_validate({**_endpoint().model_dump(exclude={"extra_headers"})})
    assert first.extra_headers == {}
    assert second.extra_headers == {}
    assert first.extra_headers is not second.extra_headers


def test_features_default_is_unverified_production() -> None:
    endpoint = Endpoint.model_validate({**_endpoint().model_dump(exclude={"features"})})
    assert endpoint.features == EndpointFeatures()
    assert endpoint.features.production is True
    assert endpoint.features.get("strict") is None


# --------------------------------------------------------------------------------------
# Frozen
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (_endpoint(), "path", "/v2/messages"),
        (EndpointFeatures(), "strict", True),
        (Requirements(), "allow_stubs", True),
    ],
)
def test_models_are_frozen(model: BaseModel, field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        setattr(model, field, value)


# --------------------------------------------------------------------------------------
# Requirements / Selection
# --------------------------------------------------------------------------------------


def test_requirements_defaults() -> None:
    requirements = Requirements()
    assert requirements.pin is None
    assert requirements.needs == frozenset()
    assert requirements.must_be_production is True
    assert requirements.must_be_verified is False
    assert requirements.allow_stubs is False


def test_requirements_needs_coerces_to_frozenset() -> None:
    requirements = Requirements.model_validate({"needs": ["strict", "caching"]})
    assert requirements.needs == frozenset({"strict", "caching"})
    assert isinstance(requirements.needs, frozenset)


def test_selection_round_trips_its_fields() -> None:
    endpoint = _endpoint()
    selection = Selection(
        endpoint=endpoint,
        dialect=ANTHROPIC_MESSAGES,
        adapter="anthropic_messages_chat",
        via="native",
        unverified=frozenset({"caching"}),
        rejected=("openai/chat/responses: not production",),
    )
    assert selection.endpoint == endpoint
    assert selection.dialect == ANTHROPIC_MESSAGES
    assert selection.adapter == "anthropic_messages_chat"
    assert selection.via == "native"
    assert selection.unverified == frozenset({"caching"})
    assert selection.rejected == ("openai/chat/responses: not production",)
    assert Selection.model_validate(selection.model_dump()) == selection


def test_selection_rejects_unknown_via() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Selection.model_validate(
            {
                "endpoint": _endpoint().model_dump(),
                "dialect": "anthropic_messages",
                "adapter": "x",
                "via": "guess",
                "unverified": [],
                "rejected": [],
            }
        )
    assert any("via" in error["loc"] for error in exc_info.value.errors())
