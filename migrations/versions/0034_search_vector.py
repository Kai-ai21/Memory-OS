"""M11.2: keep keyword search working over encrypted chunks.

`memory_chunks.search_vector` was `GENERATED ALWAYS AS to_tsvector('english',
content)`. Encrypting `content` therefore encrypted the *index*: after
`encrypt-existing` the tsvector held base64 lexemes and `plainto_tsquery`
matched nothing at all, corpus-wide. Vector search still worked, because
embeddings are deliberately not encrypted; the lexical half of hybrid retrieval
was simply gone.

That failure is quiet, which is the worst kind. Nothing errors — searches just
stop finding things — and it would have made M11.2's own headline test
("a shredded memory's text returns nothing from search") pass for the wrong
reason, by finding nothing for anything.

So the column stops being generated and is maintained by a trigger instead.

**A trigger rather than application code**, and the first attempt at this was a
SQLAlchemy mapper event that got it wrong. A generated column's real guarantee
is that *nothing* can write `content` and leave the index stale — not a bulk
`UPDATE`, not a Core statement that bypasses the ORM, not somebody at a `psql`
prompt. An ORM event covers only writers that go through the ORM, which is a
strictly weaker promise wearing the same clothes. The trigger keeps the original
guarantee and adds one clause: ciphertext is left alone, because by then the
correct vector has already been derived from the plaintext.

**This is a real disclosure and it belongs in the README rather than only
here.** A tsvector is a list of the stemmed words in a chunk with their
positions. Somebody with database access can read it. It does not give them the
sentences, the order across chunk boundaries, or anything unstemmed — but "the
content is encrypted" would be an overstatement, and the honest sentence is that
the *text* is encrypted and its *vocabulary* is not.

Revision ID: 0034
Revises: 0033
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0034"
down_revision: str | None = "0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SET_VECTOR = """
CREATE OR REPLACE FUNCTION memos_set_search_vector() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.content IS NULL THEN
    NEW.search_vector := NULL;
  ELSIF NEW.content LIKE 'enc:v1:%' THEN
    -- Already encrypted. The vector was derived from the plaintext before the
    -- encryption and is carried in on the same statement; deriving one from the
    -- envelope is the bug this whole migration exists to fix.
    NEW.search_vector := NEW.search_vector;
  ELSE
    NEW.search_vector := to_tsvector('english', NEW.content);
  END IF;
  RETURN NEW;
END;
$$
"""

ATTACH = """
CREATE TRIGGER memory_chunks_search_vector
BEFORE INSERT OR UPDATE OF content ON memory_chunks
FOR EACH ROW EXECUTE FUNCTION memos_set_search_vector()
"""


def upgrade() -> None:
    # The GIN index depends on the column, so it goes first and comes back after.
    op.execute("DROP INDEX IF EXISTS ix_memory_chunks_search")
    op.execute("ALTER TABLE memory_chunks DROP COLUMN search_vector")
    op.execute("ALTER TABLE memory_chunks ADD COLUMN search_vector tsvector")
    # Rebuilt from whatever is still plaintext. Rows already encrypted cannot be
    # recovered here — SQL has no key — and are repaired by `encrypt-existing`,
    # which decrypts, re-derives and re-encrypts.
    op.execute(
        "UPDATE memory_chunks SET search_vector = to_tsvector('english', content) "
        "WHERE content NOT LIKE 'enc:v1:%'"
    )
    op.execute("CREATE INDEX ix_memory_chunks_search ON memory_chunks USING gin (search_vector)")
    op.execute(SET_VECTOR)
    op.execute(ATTACH)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS memory_chunks_search_vector ON memory_chunks")
    op.execute("DROP FUNCTION IF EXISTS memos_set_search_vector()")
    op.execute("DROP INDEX IF EXISTS ix_memory_chunks_search")
    op.execute("ALTER TABLE memory_chunks DROP COLUMN search_vector")
    op.execute(
        "ALTER TABLE memory_chunks ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute("CREATE INDEX ix_memory_chunks_search ON memory_chunks USING gin (search_vector)")
