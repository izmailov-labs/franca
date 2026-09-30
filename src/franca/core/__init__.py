"""Core contracts: identifiers, enums, seam protocols and the wire-facing types.

Importing this package must never pull in a capability package (`franca.chat`) or a
transport implementation, so that `pip install franca` works without httpx.
"""
