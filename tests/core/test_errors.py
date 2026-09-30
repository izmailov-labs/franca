"""Executable specification for `franca.core.errors`."""

from typing import Any, Literal, get_args

import pytest
from pydantic import BaseModel, Field, ValidationError

from franca.core.errors import (
    ConfigError,
    FailureClass,
    FrancaError,
    JobFailedError,
    JobTimeoutError,
    ModelError,
    PluginConflictError,
    ProfileError,
    SelectionError,
    format_validation,
)
from franca.core.ids import ANTHROPIC, OPENAI, Provider

RAW = {"error": {"type": "overloaded_error", "message": "Overloaded", "echo": "prompt text"}}
RAW_MARKERS = ("overloaded_error", "Overloaded", "prompt text", "echo")


def _model_error(**overrides: Any) -> ModelError:
    kwargs: dict[str, Any] = {
        "status": 529,
        "provider": ANTHROPIC,
        "retryable": True,
        "failure_class": "provider",
        "raw": RAW,
        "retry_after_s": 2.5,
    }
    kwargs.update(overrides)
    return ModelError("overloaded", **kwargs)


# --------------------------------------------------------------------------------------
# Hierarchy (diagrams/panels/10c-error-hierarchy.mmd)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("err", "bases"),
    [
        (_model_error(), (ModelError, FrancaError, Exception)),
        (
            SelectionError("none", provider=ANTHROPIC, reasons=()),
            (SelectionError, ModelError, FrancaError, Exception),
        ),
        (
            JobTimeoutError("late", provider=ANTHROPIC),
            (JobTimeoutError, ModelError, FrancaError, Exception),
        ),
        (
            JobFailedError("failed", provider=ANTHROPIC),
            (JobFailedError, ModelError, FrancaError, Exception),
        ),
        (ProfileError("no profile"), (ProfileError, FrancaError, Exception)),
        (ConfigError("bad config"), (ConfigError, FrancaError, Exception)),
        (
            PluginConflictError("dup"),
            (PluginConflictError, ConfigError, FrancaError, Exception),
        ),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else "bases",
)
def test_isinstance_chain_matches_the_diagram(
    err: FrancaError, bases: tuple[type[BaseException], ...]
) -> None:
    for base in bases:
        assert isinstance(err, base)


@pytest.mark.parametrize(
    ("cls", "not_a"),
    [
        (ModelError, (SelectionError, JobTimeoutError, JobFailedError)),
        (SelectionError, (JobTimeoutError, JobFailedError)),
        (JobTimeoutError, (SelectionError, JobFailedError)),
        (JobFailedError, (SelectionError, JobTimeoutError)),
        (ProfileError, (ModelError, ConfigError)),
        (ConfigError, (ModelError, ProfileError, PluginConflictError)),
        (PluginConflictError, (ModelError, ProfileError)),
    ],
)
def test_siblings_are_not_confused(
    cls: type[FrancaError], not_a: tuple[type[FrancaError], ...]
) -> None:
    for other in not_a:
        assert not issubclass(cls, other)


def test_franca_error_is_a_plain_exception() -> None:
    err = FrancaError("anything")
    assert isinstance(err, Exception)
    assert str(err) == "anything"


def test_one_clause_catches_the_whole_library() -> None:
    with pytest.raises(FrancaError):
        raise JobFailedError("failed", provider=OPENAI)


def test_failure_class_alias_lists_the_ten_kinds() -> None:
    assert set(get_args(FailureClass.__value__)) == {
        "transport",
        "auth",
        "rate_limit",
        "provider",
        "parse",
        "timeout",
        "selection",
        "unsupported",
        "request",
        "lint",
    }


# --------------------------------------------------------------------------------------
# ModelError attributes, str and repr
# --------------------------------------------------------------------------------------


def test_model_error_attributes_round_trip() -> None:
    err = _model_error()
    assert err.message == "overloaded"
    assert err.status == 529
    assert err.provider == ANTHROPIC
    assert err.retryable is True
    assert err.failure_class == "provider"
    assert err.raw is RAW
    assert err.retry_after_s == 2.5


def test_model_error_optional_fields_default_to_none() -> None:
    err = ModelError("no key", status=None, provider=OPENAI, retryable=False, failure_class="auth")
    assert err.status is None
    assert err.raw is None
    assert err.retry_after_s is None


@pytest.mark.parametrize(
    "failure_class",
    [
        "transport",
        "auth",
        "rate_limit",
        "provider",
        "parse",
        "timeout",
        "selection",
        "unsupported",
        "request",
        "lint",
    ],
)
def test_every_failure_class_is_accepted(failure_class: FailureClass) -> None:
    err = _model_error(failure_class=failure_class)
    assert err.failure_class == failure_class


def test_str_is_the_message_only() -> None:
    err = _model_error()
    assert str(err) == "overloaded"
    assert err.args == ("overloaded",)


def test_str_and_repr_never_include_raw() -> None:
    err = _model_error()
    rendered = f"{err!s}\n{err!r}"
    for marker in RAW_MARKERS:
        assert marker not in rendered


def test_repr_carries_the_structured_fields() -> None:
    text = repr(_model_error())
    assert text.startswith("ModelError('overloaded', ")
    assert "status=529" in text
    assert "provider='anthropic'" in text
    assert "retryable=True" in text
    assert "failure_class='provider'" in text
    assert "retry_after_s=2.5" in text
    assert "raw" not in text


def test_provider_is_stored_verbatim_including_plugin_slugs() -> None:
    plugin = Provider("acme_llm")
    err = ModelError("boom", status=500, provider=plugin, retryable=True, failure_class="provider")
    assert err.provider == plugin


# --------------------------------------------------------------------------------------
# Fixed-field subclasses
# --------------------------------------------------------------------------------------


def test_selection_error_fixes_status_class_and_retryability() -> None:
    err = SelectionError(
        "no endpoint for anthropic satisfies the requirements",
        provider=ANTHROPIC,
        reasons=("anthropic/messages: no tools", "anthropic/legacy: retired"),
    )
    assert err.status is None
    assert err.failure_class == "selection"
    assert err.retryable is False
    assert err.raw is None
    assert err.retry_after_s is None
    assert err.reasons == ("anthropic/messages: no tools", "anthropic/legacy: retired")
    assert str(err) == "no endpoint for anthropic satisfies the requirements"


def test_selection_error_normalises_reasons_to_a_tuple() -> None:
    err = SelectionError("ambiguous", provider=OPENAI, reasons=["openai", "xai"])
    assert err.reasons == ("openai", "xai")
    assert isinstance(err.reasons, tuple)


def test_selection_error_repr_shows_reasons_but_not_raw() -> None:
    text = repr(SelectionError("ambiguous", provider=OPENAI, reasons=("openai", "xai")))
    assert text.startswith("SelectionError('ambiguous', ")
    assert "failure_class='selection'" in text
    assert "reasons=('openai', 'xai')" in text
    assert "raw" not in text


@pytest.mark.parametrize(
    ("cls", "status", "failure_class"),
    [
        (JobTimeoutError, 408, "timeout"),
        (JobFailedError, 422, "provider"),
    ],
)
def test_job_errors_fix_status_class_and_retryability(
    cls: type[JobTimeoutError | JobFailedError], status: int, failure_class: str
) -> None:
    err = cls("job ended badly", provider=OPENAI, raw=RAW)
    assert err.status == status
    assert err.failure_class == failure_class
    assert err.retryable is False
    assert err.provider == OPENAI
    assert err.raw is RAW
    assert err.retry_after_s is None
    assert str(err) == "job ended badly"


@pytest.mark.parametrize("cls", [JobTimeoutError, JobFailedError])
def test_job_errors_default_raw_to_none_and_hide_it(
    cls: type[JobTimeoutError | JobFailedError],
) -> None:
    assert cls("job ended badly", provider=OPENAI).raw is None
    rendered = repr(cls("job ended badly", provider=OPENAI, raw=RAW))
    for marker in RAW_MARKERS:
        assert marker not in rendered


@pytest.mark.parametrize(
    ("factory", "override"),
    [
        (
            lambda **kw: SelectionError("x", provider=ANTHROPIC, reasons=(), **kw),
            {"status": 500},
        ),
        (
            lambda **kw: SelectionError("x", provider=ANTHROPIC, reasons=(), **kw),
            {"failure_class": "provider"},
        ),
        (
            lambda **kw: SelectionError("x", provider=ANTHROPIC, reasons=(), **kw),
            {"retryable": True},
        ),
        (lambda **kw: JobTimeoutError("x", provider=ANTHROPIC, **kw), {"status": 500}),
        (lambda **kw: JobTimeoutError("x", provider=ANTHROPIC, **kw), {"failure_class": "auth"}),
        (lambda **kw: JobTimeoutError("x", provider=ANTHROPIC, **kw), {"retryable": True}),
        (lambda **kw: JobFailedError("x", provider=ANTHROPIC, **kw), {"status": 500}),
        (lambda **kw: JobFailedError("x", provider=ANTHROPIC, **kw), {"failure_class": "auth"}),
        (lambda **kw: JobFailedError("x", provider=ANTHROPIC, **kw), {"retryable": True}),
    ],
    ids=lambda v: next(iter(v)) if isinstance(v, dict) else "ctor",
)
def test_fixed_fields_cannot_be_overridden_at_construction(
    factory: Any, override: dict[str, Any]
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        factory(**override)


# --------------------------------------------------------------------------------------
# format_validation
# --------------------------------------------------------------------------------------


class _Inner(BaseModel, frozen=True):
    """Leaf of the nested fixture: one integer."""

    count: int


class _Outer(BaseModel, frozen=True):
    """Root of the nested fixture: a nested model and a list of nested models."""

    inner: _Inner
    items: list[_Inner] = Field(default_factory=list)


class _ProviderSettings(BaseModel, frozen=True):
    """Settings-shaped fixture whose `api_key` has a pattern the secret will not match."""

    api_key: str = Field(pattern=r"^[a-z]{3}$")
    mode: Literal["stateless", "server"] = "stateless"


def _validation_error(model: type[BaseModel], data: dict[str, Any]) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(data)
    return exc_info.value


def test_format_validation_renders_dotted_path_message_and_type() -> None:
    text = format_validation(_validation_error(_Outer, {"inner": {"count": "many"}}))
    assert text.startswith("inner.count: ")
    assert text.endswith(" [int_parsing]")
    assert "; " not in text


def test_format_validation_joins_errors_and_stringifies_list_indices() -> None:
    data = {"inner": {}, "items": [{"count": 1}, {"count": "two"}]}
    text = format_validation(_validation_error(_Outer, data))
    clauses = text.split("; ")
    assert len(clauses) == 2
    assert clauses[0].startswith("inner.count: ")
    assert clauses[0].endswith(" [missing]")
    assert clauses[1].startswith("items.1.count: ")
    assert clauses[1].endswith(" [int_parsing]")


def test_format_validation_root_error_has_an_empty_path() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _Inner.model_validate(["not", "a", "mapping"])
    text = format_validation(exc_info.value)
    assert text.startswith(": ")
    assert text.endswith(" [model_type]")


def test_format_validation_never_echoes_the_rejected_input() -> None:
    leaky = "sk-live-SECRET123"
    exc = _validation_error(_ProviderSettings, {"api_key": leaky, "mode": "sk-live-SECRET123-mode"})
    # The default rendering leaks, which is exactly why format_validation exists.
    assert leaky in str(exc)
    text = format_validation(exc)
    assert leaky not in text
    assert "SECRET123" not in text
    clauses = text.split("; ")
    assert len(clauses) == 2
    assert clauses[0].startswith("api_key: ")
    assert clauses[0].endswith(" [string_pattern_mismatch]")
    assert clauses[1].startswith("mode: ")
    assert clauses[1].endswith(" [literal_error]")


def test_format_validation_omits_the_documentation_url() -> None:
    text = format_validation(_validation_error(_Inner, {"count": "many"}))
    assert "https://" not in text
    assert "errors.pydantic.dev" not in text


def test_format_validation_feeds_a_request_model_error_without_the_input() -> None:
    leaky = "sk-live-SECRET123"
    exc = _validation_error(_ProviderSettings, {"api_key": leaky})
    err = ModelError(
        format_validation(exc),
        status=None,
        provider=OPENAI,
        retryable=False,
        failure_class="request",
    )
    assert leaky not in str(err)
    assert leaky not in repr(err)
    assert err.failure_class == "request"
