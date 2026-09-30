"""Executable specification for `franca.chat.ir`.

Most of this file pins field defaults, because the IR's defaults *are* its contract:
an adapter reads `pkg.reasoning.mode is None` and says nothing on the wire, so a
default changing from `None` to `"enabled"` would silently rewrite every request.

Four behaviours here are not visible in the field list and each one has a cost when
it breaks. `Item.artifact` is hidden from `repr()` but must survive `model_dump()` --
a transcript persisted without a thinking signature is rejected by the provider on
the next turn. `ModelResponse.raw` is the reverse. `PromptPackage` is strict, so
`max_output_tokens="1"` raises while `items=[...]` still takes a plain list. And a
bare dict is not promoted to an `Item` in Python, though a JSON object still is,
which is what keeps a dumped package re-readable.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from franca.chat.ir import (
    Item,
    ModelResponse,
    OutputFormat,
    PackageMeta,
    PromptPackage,
    Reasoning,
    Refusal,
    SystemBlock,
    ToolDef,
)
from franca.core.enums import StateMode
from franca.core.ids import ANTHROPIC, ANTHROPIC_MESSAGES
from franca.core.types import CallTrace, HasTrace, Usage

SIGNATURE = "EqQBCkYIBRgCKkBs0m4kL2NvdC1zaWduYXR1cmU"


def a_trace() -> CallTrace:
    return CallTrace(
        endpoint_id="anthropic/chat/messages",
        dialect=ANTHROPIC_MESSAGES,
        selection_via="native",
    )


def a_response(items: tuple[Item, ...] = (), **extra: Any) -> ModelResponse:
    return ModelResponse(
        model="claude-opus-4-5",
        provider=ANTHROPIC,
        dialect=ANTHROPIC_MESSAGES,
        items=items,
        **extra,
    )


def _stamp(obj: HasTrace, trace: CallTrace) -> HasTrace:
    """Push a response through the structural seam middleware uses, so mypy checks it."""
    return obj.with_trace(trace)


def test_item_defaults() -> None:
    item = Item(role="user", kind="text")

    assert item.text is None
    assert item.name is None
    assert item.call_id is None
    assert item.args is None
    assert item.asset is None
    assert item.server_side is False
    assert item.artifact is None


def test_item_rejects_a_role_outside_the_vocabulary() -> None:
    with pytest.raises(ValidationError):
        Item(role="operator", kind="text")  # type: ignore[arg-type]


def test_system_block_defaults() -> None:
    block = SystemBlock(text="be brief")

    assert block.position == 0


def test_tool_def_defaults() -> None:
    tool = ToolDef(name="search", parameters={"type": "object"})

    assert tool.description == ""
    assert tool.strict is None
    assert tool.server_side is False


def test_reasoning_defaults() -> None:
    reasoning = Reasoning()

    assert reasoning.mode is None
    assert reasoning.effort is None
    assert reasoning.budget_tokens is None
    assert reasoning.artifact_present is False


def test_output_format_defaults() -> None:
    output_format = OutputFormat()

    assert output_format.kind is None
    assert output_format.json_schema is None
    assert output_format.strict is None
    assert output_format.strategy is None


def test_package_meta_defaults() -> None:
    meta = PackageMeta()

    assert meta.route is None
    assert meta.intended_output_style is None
    assert meta.is_code_route is False
    assert meta.configuration_update is False
    assert meta.store is None


def test_refusal_defaults() -> None:
    refusal = Refusal()

    assert refusal.category is None
    assert refusal.explanation is None


def test_refusal_category_is_an_open_set() -> None:
    """A category nobody has seen before must reach the caller, not raise."""
    refusal = Refusal(category="a_category_invented_next_tuesday", explanation="no")

    assert refusal.category == "a_category_invented_next_tuesday"


def test_prompt_package_defaults() -> None:
    pkg = PromptPackage()

    assert pkg.provider is None
    assert pkg.dialect is None
    assert pkg.model is None
    assert pkg.system == ()
    assert pkg.items == ()
    assert pkg.tools == ()
    assert pkg.reasoning == Reasoning()
    assert pkg.output_format == OutputFormat()
    assert pkg.max_output_tokens is None
    assert pkg.sampling == {}
    assert pkg.cache_hints == {}
    assert pkg.state_mode is StateMode.stateless
    assert pkg.continuation is None
    assert pkg.meta == PackageMeta()
    assert pkg.provider_options == {}


def test_prompt_package_mutable_defaults_are_not_shared() -> None:
    assert PromptPackage().sampling is not PromptPackage().sampling
    assert PromptPackage().cache_hints is not PromptPackage().cache_hints
    assert PromptPackage().provider_options is not PromptPackage().provider_options


def test_model_response_defaults() -> None:
    res = a_response()

    assert res.items == ()
    assert res.stop_reason is None
    assert res.usage == Usage()
    assert res.refusal is None
    assert res.served_model is None
    assert res.continuation is None
    assert res.structured is None
    assert res.raw == {}
    assert res.trace is None


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (Item(role="user", kind="text"), "text"),
        (SystemBlock(text="be brief"), "text"),
        (ToolDef(name="search", parameters={}), "name"),
        (Reasoning(), "effort"),
        (OutputFormat(), "kind"),
        (PackageMeta(), "route"),
        (Refusal(), "category"),
        (PromptPackage(), "model"),
        (a_response(), "stop_reason"),
    ],
)
def test_every_model_is_frozen(model: BaseModel, field: str) -> None:
    with pytest.raises(ValidationError):
        setattr(model, field, "changed")


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (Item, {"role": "user", "kind": "text"}),
        (SystemBlock, {"text": "be brief"}),
        (ToolDef, {"name": "search", "parameters": {}}),
        (Reasoning, {}),
        (OutputFormat, {}),
        (PackageMeta, {}),
        (Refusal, {}),
        (PromptPackage, {}),
    ],
)
def test_an_unknown_key_is_rejected_with_its_name_in_the_loc(
    model_cls: type[BaseModel], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError) as caught:
        model_cls(**kwargs, thinkign_budget=1)

    error = caught.value.errors()[0]
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == ("thinkign_budget",)


def test_strict_rejects_a_string_for_an_int_field() -> None:
    with pytest.raises(ValidationError) as caught:
        PromptPackage(max_output_tokens="1")  # type: ignore[arg-type]

    error = caught.value.errors()[0]
    assert error["type"] == "int_type"
    assert error["loc"] == ("max_output_tokens",)


def test_a_sequence_field_accepts_a_list_and_stores_a_tuple() -> None:
    """Runtime only: the annotation is a tuple, so a mypy-strict caller still passes one."""
    block = SystemBlock(text="be brief")
    item = Item(role="user", kind="text", text="hi")
    tool = ToolDef(name="search", parameters={})

    pkg = PromptPackage(system=[block], items=[item], tools=[tool])  # type: ignore[arg-type]

    assert pkg.system == (block,)
    assert pkg.items == (item,)
    assert pkg.tools == (tool,)


def test_a_dict_is_not_coerced_into_an_item() -> None:
    with pytest.raises(ValidationError) as caught:
        PromptPackage(items=[{"role": "user", "kind": "text", "text": "hi"}])  # type: ignore[arg-type]

    error = caught.value.errors()[0]
    assert error["type"] == "is_instance_of"
    assert error["loc"] == ("items", 0)


def test_a_package_round_trips_through_json() -> None:
    """The price of rejecting dicts: a dumped package is re-read as JSON, not as a dict."""
    pkg = PromptPackage(
        model="claude-opus-4-5",
        system=(SystemBlock(text="be brief", position=1),),
        items=(Item(role="user", kind="text", text="hi", artifact={"signature": SIGNATURE}),),
        tools=(ToolDef(name="search", parameters={"type": "object"}),),
        max_output_tokens=64,
        sampling={"temperature": 0.2},
    )

    again = PromptPackage.model_validate_json(pkg.model_dump_json())

    assert again == pkg


def test_artifact_is_hidden_from_repr_but_present_in_the_dump() -> None:
    item = Item(role="assistant", kind="reasoning", text="hmm", artifact={"signature": SIGNATURE})

    assert "artifact" not in repr(item)
    assert SIGNATURE not in repr(item)
    assert item.model_dump()["artifact"] == {"signature": SIGNATURE}


def test_an_item_round_trips_its_artifact_exactly() -> None:
    """A transcript that loses a thinking signature is rejected on the next turn."""
    item = Item(
        role="assistant",
        kind="reasoning",
        text="hmm",
        artifact={"type": "thinking", "signature": SIGNATURE, "nested": {"keep": [1, 2]}},
    )

    again = Item.model_validate(item.model_dump())

    assert again.artifact == item.artifact
    assert again == item


def test_model_response_raw_is_excluded_from_the_dump_and_the_repr() -> None:
    res = a_response(raw={"id": "msg_01", "content": [{"type": "text", "text": "hi"}]})

    assert res.raw["id"] == "msg_01"
    assert "raw" not in res.model_dump()
    assert "msg_01" not in repr(res)


def test_text_is_the_assistant_and_model_text_items_only() -> None:
    res = a_response(
        items=(
            Item(role="user", kind="text", text="what is up"),
            Item(role="assistant", kind="reasoning", text="thinking hard"),
            Item(role="assistant", kind="text", text="Hello, "),
            Item(role="assistant", kind="text"),
            Item(role="model", kind="text", text="world"),
            Item(role="assistant", kind="tool_call", name="search", args={"q": "x"}),
            Item(role="tool", kind="tool_result", call_id="c1", text="ignored"),
        )
    )

    assert res.text == "Hello, world"


def test_text_is_empty_without_any_answer() -> None:
    assert a_response().text == ""


def test_tool_calls_are_the_tool_call_items_in_order() -> None:
    first = Item(role="assistant", kind="tool_call", name="search", call_id="c1", args={})
    second = Item(role="assistant", kind="tool_call", name="read", call_id="c2", args={})
    res = a_response(
        items=(
            Item(role="assistant", kind="text", text="on it"),
            first,
            Item(role="assistant", kind="reasoning", text="hmm"),
            second,
        )
    )

    assert res.tool_calls == (first, second)


def test_tool_calls_is_empty_when_nothing_was_called() -> None:
    assert a_response(items=(Item(role="assistant", kind="text", text="hi"),)).tool_calls == ()


def test_system_text_orders_by_position_then_declaration() -> None:
    pkg = PromptPackage(
        system=(
            SystemBlock(text="third", position=10),
            SystemBlock(text="second-a"),
            SystemBlock(text="second-b"),
            SystemBlock(text="first", position=-5),
        )
    )

    assert pkg.system_text() == "first\n\nsecond-a\n\nsecond-b\n\nthird"


def test_system_text_skips_empty_blocks_and_emits_no_separator_for_them() -> None:
    pkg = PromptPackage(system=(SystemBlock(text="a"), SystemBlock(text=""), SystemBlock(text="b")))

    assert pkg.system_text() == "a\n\nb"


def test_system_text_is_empty_without_blocks() -> None:
    assert PromptPackage().system_text() == ""


def test_instruction_text_is_the_system_text_plus_instruction_shaped_items() -> None:
    pkg = PromptPackage(
        system=(SystemBlock(text="be brief"),),
        items=(
            Item(role="developer", kind="text", text="answer in JSON"),
            Item(role="user", kind="text", text="what is up"),
            Item(role="system", kind="text", text="stay on topic"),
            Item(role="system", kind="reasoning", text="not an instruction"),
            Item(role="assistant", kind="text", text="hi"),
        ),
    )

    assert pkg.instruction_text() == "be brief\n\nanswer in JSON\n\nstay on topic"


def test_instruction_text_equals_system_text_for_the_usual_package() -> None:
    pkg = PromptPackage(
        system=(SystemBlock(text="be brief"),),
        items=(Item(role="user", kind="text", text="what is up"),),
    )

    assert pkg.instruction_text() == pkg.system_text() == "be brief"


def test_instruction_text_is_empty_without_any_instruction() -> None:
    assert PromptPackage().instruction_text() == ""


def test_last_item_is_none_on_an_empty_package() -> None:
    assert PromptPackage().last_item() is None


def test_last_item_is_the_final_item() -> None:
    last = Item(role="user", kind="text", text="and now this")
    pkg = PromptPackage(items=(Item(role="user", kind="text", text="first"), last))

    assert pkg.last_item() == last


def test_model_response_satisfies_has_trace() -> None:
    res = a_response(items=(Item(role="assistant", kind="text", text="hi"),))

    assert isinstance(res, HasTrace)


def test_with_trace_returns_a_copy_carrying_the_trace() -> None:
    res = a_response(items=(Item(role="assistant", kind="text", text="hi"),))

    stamped = _stamp(res, a_trace())

    assert isinstance(stamped, ModelResponse)
    assert stamped is not res
    assert stamped.trace == a_trace()
    assert res.trace is None
    assert stamped.text == res.text
