"""The chat leaf: `Model` bound to the chat IR, plus the client alias above it.

`ChatModel` is `franca.core.model.Model` with its three type parameters fixed to the
chat capability, and one override: `prepare`, the hook the base leaf calls before it
hands a request to an adapter. Everything else -- building the `WireRequest`, sending
it, translating a `ValidationError`, stamping the `CallTrace` -- is capability-agnostic
and stays in core, which is the whole reason the hook exists.

The delta type is `Never`, not `Delta`, and that is temporary. Chat's streaming delta
does not exist until M1, and `Never` is the honest spelling of "this client yields
nothing": it makes `async for` over `ChatModel.stream` a type error rather than a
runtime surprise, and it keeps `wrap` from splicing a chat leaf into a pipeline that
expects deltas. When M1 lands, `Never` becomes `Delta` here and in `ChatClient`, which
widens both -- existing code that never touched the stream path keeps compiling.

`prepare` is likewise the M0 subset. It fills in the identifiers the caller left open
and refuses a retired model before any I/O happens. The full version, which re-derives
the package's endpoint requirements (§5.3), checks them against the selection rather
than re-selecting, and folds newly unverified features into `WireRequest.unverified`,
is M3 work and lands with `requirements_from` and the tri-state profile gates.
"""

from typing import Any, ClassVar, Never

from franca.chat.ir import ModelResponse, PromptPackage
from franca.chat.profile import ChatProfile
from franca.core.client import Client
from franca.core.errors import ModelError
from franca.core.ids import CHAT, Capability
from franca.core.model import Model


class ChatModel(Model[PromptPackage, ModelResponse, Never]):
    """One chat model, bound to one endpoint, one adapter and one profile.

    Constructed exactly like its base -- `model`, `connector`, `adapter`, `profile`,
    `clock` and an optional `selection` -- and satisfies `ChatClient` structurally, so
    `wrap` composes middleware around it without either side naming the other.
    """

    capability: ClassVar[Capability] = CHAT

    def prepare(self, req: PromptPackage) -> PromptPackage:
        """Refuse a retired model, then fill in the identifiers the caller left open.

        The order matters: a retired model is refused first, so no work is done and
        nothing is sent for a model that is known to be gone. That check reads
        `ChatProfile.retired` as the tri-state it is -- only an explicit `True` refuses,
        because `None` means nobody has checked and `False` means the model is alive.
        A profile that is not a `ChatProfile` has no such field and cannot refuse.

        Filling is deliberately non-destructive. `provider`, `dialect` and `model` on a
        package are *hints* to the registry, and by the time a leaf sees the package
        the decision they were hinting at has already been made, so the leaf records
        what actually happened -- but only where the caller said nothing. A caller who
        pinned a value keeps it, even where it disagrees with this leaf: the package is
        the caller's record of intent, and a silent rewrite would erase the evidence
        that a pin was ignored. Nothing downstream reads these three fields, so a
        disagreement is visible rather than harmful.

        The copy is skipped entirely when there is nothing to fill, so a fully
        specified package is passed through as the same object.

        Args:
            req: The package as the caller passed it.

        Returns:
            The package the adapter should render: `req` itself, or a copy of it
            carrying the identifiers it left open.

        Raises:
            ModelError: With `failure_class="unsupported"` and `retryable=False` when
                the profile says the model is retired. Raised before any I/O, and not
                retryable because no number of attempts brings a model back.
        """
        profile = self.profile
        if isinstance(profile, ChatProfile) and profile.retired is True:
            raise ModelError(
                f"{self.model} is retired: profile row {profile.name!r} says the model"
                " is no longer served; pick a current model",
                status=None,
                provider=self.provider,
                retryable=False,
                failure_class="unsupported",
            )

        endpoint = self.connector.endpoint
        filled: dict[str, Any] = {}
        if req.provider is None:
            filled["provider"] = endpoint.provider
        if req.dialect is None:
            filled["dialect"] = endpoint.dialect
        if req.model is None:
            filled["model"] = self.model
        return req.model_copy(update=filled) if filled else req


type ChatClient = Client[PromptPackage, ModelResponse, Never]
"""A chat pipeline: a `PromptPackage` in, a `ModelResponse` out, no deltas yet.

`ChatModel` is the leaf that satisfies it and every chat middleware both consumes and
produces it. The `Never` becomes `Delta` when streaming lands in M1.
"""
