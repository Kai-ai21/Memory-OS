"""Identifier generation.

UUIDv7 everywhere. The first 48 bits are a millisecond timestamp, so keys sort
by creation time and insert at the right-hand edge of a B-tree instead of
scattering across it the way UUIDv4 does. The difference is write amplification
that grows with the table, and changing a primary key type after data exists is
a full-table migration.
"""

from uuid import UUID

from uuid_utils.compat import uuid7


def new_id() -> UUID:
    """Return a fresh time-ordered identifier as a stdlib `uuid.UUID`."""
    return uuid7()
