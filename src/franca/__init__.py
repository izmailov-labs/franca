"""franca — one request shape for every LLM wire dialect.

A lingua franca for model APIs: one intermediate representation per capability
(chat, image, video), an adapter per wire dialect, and a registry that decides
which dialect a given model actually speaks. Providers, dialects and models are
kept as three independent axes, so a compatibility surface that remaps model
names does not have to be special-cased anywhere else.

The public API is re-exported from this module; everything not listed in
``__all__`` is internal and may change without a major version bump.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("franca")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"
