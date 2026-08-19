"""Encryption at rest: a data key per memory, wrapped by one master key.

**Envelope encryption, and the shape is the whole design.** Each memory gets a
random 256-bit data key. Content is encrypted with that key; the key itself is
encrypted with the master key and stored in `memory_keys`. The master key comes
from the environment and is never written to the database.

That indirection is what makes *crypto-shredding* possible. Permanent deletion
destroys one wrapped key — a few dozen bytes — and every copy of that memory's
ciphertext, in the database, in a backup taken last month, in a replica nobody
has thought about, becomes permanently undecryptable at the same instant. There
is no way to achieve that by deleting rows, because you do not control the
copies.

**AES-256-GCM, not CBC or CTR.** GCM authenticates as well as encrypts: a
ciphertext that has been tampered with fails to decrypt rather than decrypting
to something else. Without that, an attacker with write access to the database
can flip bits in a memory's content and the system will believe the result.

**The memory id is bound in as associated data.** The AAD is authenticated but
not encrypted, so moving a ciphertext from one memory's row to another's is
detectable: the tag will not verify. Without it, somebody with database access
could swap two memories' contents and nothing would notice.

**Losing the master key is unrecoverable and that is the point.** There is no
escrow, no backdoor and no recovery path. A system that can recover your data
without your key can be made to hand it over without your key.
"""

import base64
import os
import secrets
from dataclasses import dataclass
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

#: What `MEMOS_MASTER_KEY` holds: 32 random bytes, base64. Generate one with
#: `python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`.
MASTER_KEY_ENV = "MEMOS_MASTER_KEY"

KEY_BYTES = 32
NONCE_BYTES = 12

#: Stamped into `memory_keys.algorithm`. Present so a future scheme can be
#: introduced per row rather than by a migration that must decrypt everything.
ALGORITHM = "AESGCM-256/v1"

#: Prefix on every stored ciphertext. Two jobs: it makes an encrypted column
#: visually obvious in `psql`, and it is how `encrypt-existing` tells a row it
#: has already done from one it has not — which is what makes it resumable.
CIPHERTEXT_PREFIX = "enc:v1:"


class MissingMasterKey(RuntimeError):
    """`MEMOS_MASTER_KEY` is absent or malformed.

    Raised at startup rather than at first use. **Starting with encryption
    silently disabled is worse than not starting**: the process would come up,
    serve traffic, and write plaintext into columns everything downstream
    believes are encrypted — and the damage is only visible later, in a backup.
    """


class DecryptionFailed(RuntimeError):
    """The ciphertext did not authenticate.

    Three causes and the caller cannot tell them apart, which is correct: the
    wrong master key, a tampered ciphertext, and a shredded key all mean "this
    content is not available", and distinguishing them would be a probe.
    """


def load_master_key(
    *, environ: dict[str, str] | None = None, settings: object | None = None
) -> bytes:
    """Read and validate the master key. Raises rather than returning None.

    Three sources in order of preference: an explicit `environ` mapping (tests
    and `keys rotate`, which needs to load a *second* key), a `Settings` object
    (which is how `.env` reaches this), and the process environment.
    """
    if environ is not None:
        raw = environ.get(MASTER_KEY_ENV, "").strip()
    elif settings is not None and getattr(settings, "master_key", None) is not None:
        raw = settings.master_key.get_secret_value().strip()  # type: ignore[attr-defined]
        if not raw:
            raw = os.environ.get(MASTER_KEY_ENV, "").strip()
    else:
        raw = os.environ.get(MASTER_KEY_ENV, "").strip()
    if not raw:
        raise MissingMasterKey(
            f"{MASTER_KEY_ENV} is not set. Memory OS encrypts memory content at "
            "rest and will not start without the key that decrypts it. Generate "
            'one with:\n\n  python -c "import os,base64; '
            'print(base64.b64encode(os.urandom(32)).decode())"\n\n'
            "Store it somewhere you will not lose it: there is no recovery path, "
            "and without it every encrypted memory is unreadable forever."
        )
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise MissingMasterKey(
            f"{MASTER_KEY_ENV} is not valid base64."
        ) from exc
    if len(key) != KEY_BYTES:
        raise MissingMasterKey(
            f"{MASTER_KEY_ENV} decodes to {len(key)} bytes; AES-256 needs {KEY_BYTES}."
        )
    return key


def new_data_key() -> bytes:
    """A fresh 256-bit data key, from the OS CSPRNG."""
    return secrets.token_bytes(KEY_BYTES)


@dataclass(frozen=True, slots=True)
class Cipher:
    """Wrap and unwrap data keys; encrypt and decrypt content with them.

    Holds the master key in memory and nothing else. Constructed once per
    process from `load_master_key`.
    """

    master_key: bytes

    def wrap(self, data_key: bytes, memory_id: UUID) -> bytes:
        """Encrypt a data key under the master key, bound to its memory."""
        nonce = secrets.token_bytes(NONCE_BYTES)
        sealed = AESGCM(self.master_key).encrypt(
            nonce, data_key, _aad(memory_id)
        )
        return nonce + sealed

    def unwrap(self, wrapped: bytes, memory_id: UUID) -> bytes:
        """Recover a data key. Raises `DecryptionFailed` on any mismatch."""
        try:
            return AESGCM(self.master_key).decrypt(
                wrapped[:NONCE_BYTES], wrapped[NONCE_BYTES:], _aad(memory_id)
            )
        except InvalidTag as exc:
            raise DecryptionFailed(
                "The data key did not decrypt. Either MEMOS_MASTER_KEY is the "
                "wrong key, or this key was destroyed by a permanent deletion."
            ) from exc

    def encrypt(self, plaintext: str, data_key: bytes, memory_id: UUID) -> str:
        """Encrypt content. Returns a prefixed, base64 envelope for a TEXT column.

        Base64 in `TEXT` rather than raw bytes in `BYTEA`, deliberately: every
        content column in this schema is already `TEXT`, and changing five of
        them to `BYTEA` would touch every mapper, every query and every test for
        a representation nobody reads by hand. The prefix is what makes the
        column self-describing.
        """
        nonce = secrets.token_bytes(NONCE_BYTES)
        sealed = AESGCM(data_key).encrypt(
            nonce, plaintext.encode("utf-8"), _aad(memory_id)
        )
        return CIPHERTEXT_PREFIX + base64.b64encode(nonce + sealed).decode("ascii")

    def decrypt(self, envelope: str, data_key: bytes, memory_id: UUID) -> str:
        """Decrypt content written by `encrypt`.

        Plaintext passes through untouched. That is not laxness — it is what
        lets `encrypt-existing` run against a half-encrypted corpus, and what
        keeps a reader working while the migration is in flight.
        """
        if not is_encrypted(envelope):
            return envelope
        raw = base64.b64decode(envelope[len(CIPHERTEXT_PREFIX) :])
        try:
            return AESGCM(data_key).decrypt(
                raw[:NONCE_BYTES], raw[NONCE_BYTES:], _aad(memory_id)
            ).decode("utf-8")
        except InvalidTag as exc:
            raise DecryptionFailed(
                "Content did not authenticate. The ciphertext has been altered, "
                "or it belongs to a different memory."
            ) from exc


def is_encrypted(value: str | None) -> bool:
    return bool(value) and value.startswith(CIPHERTEXT_PREFIX)  # type: ignore[union-attr]


def _aad(memory_id: UUID) -> bytes:
    """Associated data: authenticated, not encrypted.

    Binding the memory id means a ciphertext lifted from one row and pasted into
    another fails to authenticate rather than decrypting into the wrong memory.
    """
    return str(memory_id).encode("ascii")
