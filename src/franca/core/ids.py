"""Open identifier vocabularies: provider, dialect and capability slugs.

Open on purpose. A plugin registers a new provider, a new wire dialect or a new
capability as *data* -- an endpoint row and a key -- without editing franca, so these
cannot be enumerations. Vocabularies that drive a branch in franca's own code are
closed instead, and live in `core.enums` as `StrEnum`.

Each identifier is a `NewType` over `str`: free at runtime, but distinct to the type
checker, so a `Dialect` cannot be passed where a `Provider` is expected. `NewType`
carries no runtime validation, so pydantic model fields use the `*Field` aliases
below, which attach `validate_slug` as an `AfterValidator`.
"""

import re
from typing import Annotated, NewType

from pydantic import AfterValidator

Provider = NewType("Provider", str)
"""Who you authenticate with: `anthropic`, `openai`, a plugin's own slug."""

Dialect = NewType("Dialect", str)
"""A wire shape, not a vendor: xAI's video REST is not the OpenAI chat wire."""

Capability = NewType("Capability", str)
"""What is being asked for: `chat`, `image`, `video`."""

# No "-" and no ".": every slug must also spell as an environment-variable segment,
# because keys and overrides are addressable as FRANCA_<PROVIDER>_API_KEY. Underscores
# are the separator for the same reason -- every Dialect constant below relies on it.
_SLUG = re.compile(r"[a-z][a-z0-9_]{0,63}")


def validate_slug(value: str) -> str:
    """Validate an identifier slug.

    The failure message describes the *constraint*, never the offending value.
    pydantic copies a validator's `ValueError` text into `e["msg"]`, and
    `format_validation` renders that into `ConfigError` and `ModelError` messages;
    echoing the input there would leak a mistyped key or secret into logs and
    spans, defeating the guarantee that only a field path ever escapes.

    Args:
        value: The candidate identifier.

    Returns:
        The value unchanged, so this composes as a pydantic `AfterValidator`.

    Raises:
        ValueError: If the value is not a lowercase slug of 1-64 characters
            starting with a letter and containing only `[a-z0-9_]`.
    """
    if not _SLUG.fullmatch(value):
        raise ValueError("not a slug: expected 1-64 chars matching [a-z][a-z0-9_]*")
    return value


type ProviderField = Annotated[Provider, AfterValidator(validate_slug)]
type DialectField = Annotated[Dialect, AfterValidator(validate_slug)]
type CapabilityField = Annotated[Capability, AfterValidator(validate_slug)]

ANTHROPIC = Provider("anthropic")
OPENAI = Provider("openai")
GOOGLE = Provider("google")
XAI = Provider("xai")
DEEPSEEK = Provider("deepseek")

CHAT = Capability("chat")
IMAGE = Capability("image")
VIDEO = Capability("video")

ANTHROPIC_MESSAGES = Dialect("anthropic_messages")
OPENAI_RESPONSES = Dialect("openai_responses")
OPENAI_CHAT = Dialect("openai_chat")
GOOGLE_INTERACTIONS = Dialect("google_interactions")
GOOGLE_GENERATE_CONTENT = Dialect("google_generate_content")
OPENAI_IMAGES = Dialect("openai_images")
XAI_VIDEO = Dialect("xai_video")
GOOGLE_VEO = Dialect("google_veo")
