"""Deterministic JSON serialisation for request bodies.

Two requests with the same logical content must produce byte-identical bodies no
matter how their dicts were assembled. That is the mechanical basis of byte-exact
cassette matching and of the deterministic-output requirement: keys are sorted at
every nesting level, separators carry no whitespace, and non-ASCII text is emitted as
UTF-8 rather than as ASCII escape sequences, so the bytes on the wire are one
canonical form.

Private to franca: adapters call `dumps` when they build a request body; nothing
outside the package should depend on it.
"""

import json


def dumps(obj: object) -> bytes:
    """Serialise `obj` to canonical, compact, UTF-8 encoded JSON bytes.

    Keys are sorted recursively, separators are `,` and `:` with no surrounding
    whitespace, and non-ASCII characters are kept as UTF-8 rather than escaped. The
    same logical content therefore always yields the same bytes.

    Args:
        obj: Any value the standard `json` encoder accepts -- `dict`, `list`, `str`,
            `int`, `float`, `bool` or `None`, nested arbitrarily.

    Returns:
        The encoded JSON document as UTF-8 bytes.

    Raises:
        TypeError: If `obj`, or something nested inside it, is not JSON
            serialisable. The standard library's error propagates unchanged.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
