"""Argon2id, via `argon2-cffi`.

The only implementation of `PasswordHasher` there is, and the port exists so
that stays true of the *call sites* rather than of the codebase: nothing in
`application/` knows what algorithm is in use, so replacing it is this file and
a line in the container.

**Library defaults, deliberately.** `argon2-cffi`'s defaults track the RFC 9106
recommendations and move when the guidance moves; hand-picked parameters are a
snapshot of one afternoon's reading that nobody revisits. The parameters live
inside each encoded hash, so raising them later rehashes on next login rather
than invalidating anybody.
"""

from argon2 import PasswordHasher as Argon2
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


class Argon2PasswordHasher:
    """Hash and verify passwords with Argon2id.

    `verify` returns a bool rather than raising, because every caller wants the
    same thing and the alternative is three exception types repeated at each
    one. It never propagates the library's exceptions: a malformed stored hash
    and a wrong password are both "this login fails", and distinguishing them to
    the caller would eventually distinguish them to the client.
    """

    def __init__(self) -> None:
        self._hasher = Argon2()

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password_hash: str, password: str) -> bool:
        """True when the password matches. Constant-time, inside the library.

        The comparison is `argon2-cffi`'s, which is why there is no `==` on a
        hash anywhere in this codebase. A naive comparison leaks how many
        leading bytes matched through its timing, and the leak is measurable
        over enough requests.
        """
        try:
            self._hasher.verify(password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
        return True
