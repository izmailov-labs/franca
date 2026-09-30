"""Concrete `Transport` implementations: what plugs into the byte-level HTTP seam.

`franca.core.transport` defines the seam; this package holds the bindings for it --
`httpx` for production traffic, `scripted` for replaying canned exchanges under test.

Nothing is re-exported here on purpose. Reaching a transport is always an explicit
`from franca.transports.httpx import HttpxTransport`, so importing `franca` or
`franca.core` never pulls in httpx and the `franca[http]` extra stays optional.
"""
