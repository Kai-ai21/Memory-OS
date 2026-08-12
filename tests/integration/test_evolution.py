"""M4.2's four claims.

Each is a way this could be confidently wrong. A diff computed over raw bytes
still produces a diff; a history in insertion order still lists every version; a
span mapped onto the wrong chunk still names a chunk; and a model asked what
changed in an empty diff still answers. All four failures produce output that
looks like a working feature.

The first is also a live check on something else. If M1.4's normalization ever
stops collapsing line endings, this suite fails here rather than in retrieval
three milestones later — a CRLF rewrite would start reporting every line of every
file as changed, and nothing else in the system would notice.
"""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import evolution
from memoryos.application.ports import LanguageModel
from memoryos.config import Settings
from memoryos.domain.diffing import ChangeKind, diff_spans
from tests.integration.conftest import add_source, build_harness

pytestmark = pytest.mark.integration


class RefusesToBeCalled:
    """A `LanguageModel` that fails the test if anything asks it to complete.

    The trivial-diff path must not reach a model. A stub returning a fixed string
    would let a regression through: the summary would still read "No substantive
    change" because the stub said so, not because the code decided it.
    """

    @property
    def model_id(self) -> str:
        return "must-not-be-called"

    async def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        raise AssertionError(
            "the model was called for a diff that has no changed spans; "
            "'no substantive change' must be decided from the spans, not generated"
        )


class Echoes:
    """A model whose output is fixed by the test, to check what happens to it."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "echo-1"

    async def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.reply


async def test_a_line_ending_only_change_produces_an_empty_normalized_diff(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    """New bytes, new artifact, new version — and nothing changed.

    The whole file is rewritten with CRLF, so every byte after the first line
    differs and the content hash is completely different. `difflib` over the raw
    bytes would report every line as changed. Over the normalized text it reports
    nothing, which is the correct answer and is only correct because M1.4
    collapses line endings on the way in.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    body = "# Title\n\nA paragraph about workers and leases.\nA second line here.\n"
    (root / "doc.md").write_text(body, newline="")

    source = await add_source(sessions, "crlf", root)
    harness = build_harness(root, tmp_path / "blobs", sessions, source, settings)
    await harness.ingest()

    # Same text, different line endings. Nothing else touched.
    (root / "doc.md").write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
    await harness.ingest()

    history = await evolution.version_history(sessions, source.id, "doc.md")
    assert [version.version for version in history] == [1, 2]
    # A genuinely new artifact: the bytes really are different.
    assert history[1].bytes_changed is True
    # And no semantic change at all, which is what makes the chunks adoptable.
    assert history[1].text_changed is False
    assert history[1].adopted is True
    assert history[1].summary_of_change == "chunks adopted"

    diff = await evolution.diff_versions(sessions, history[0].id, history[1].id)
    assert diff.spans == []
    assert diff.is_empty
    assert diff.added_chars == 0
    assert diff.removed_chars == 0
    assert diff.unified_diff == ""


async def test_version_history_is_ordered_with_exactly_one_current(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    """Three versions, oldest first, and only the last is current.

    Ordered by `version` rather than by `ingested_at`: the relationship fields
    are each computed against the previous element, so an ordering that put a
    later version earlier would not error — it would report the reverse of what
    happened, with every added line shown as removed.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "notes.md").write_text("# Notes\n\nfirst\n")

    source = await add_source(sessions, "history", root)
    harness = build_harness(root, tmp_path / "blobs", sessions, source, settings)
    await harness.ingest()
    (root / "notes.md").write_text("# Notes\n\nfirst\nsecond\n")
    await harness.ingest()
    (root / "notes.md").write_text("# Notes\n\nfirst\nsecond\nthird\n")
    await harness.ingest()

    history = await evolution.version_history(sessions, source.id, "notes.md")

    assert [version.version for version in history] == [1, 2, 3]
    assert [version.is_current for version in history] == [False, False, True]
    assert sum(version.is_current for version in history) == 1
    # Ingestion order matches version order, which is what makes the pairwise
    # fields meaningful.
    stamps = [version.ingested_at for version in history]
    assert stamps == sorted(stamps)
    assert history[0].adopted is None, "the first version has no predecessor"
    assert all(version.text_changed for version in history[1:])

    # And the diffs run forwards. Appending "third" is reported as one CHANGED
    # span rather than an ADDED one, and that is a real property of this corpus
    # worth pinning: M1.4 strips the trailing newline, so the last line of v2 is
    # `second` and the same line in v3 is `second\n`. Every append therefore
    # rewrites the previous final line. Asserted rather than avoided — a fixture
    # arranged to dodge it would hide the behaviour a reader will meet.
    diff = await evolution.diff_versions(sessions, history[1].id, history[2].id)
    assert [span.kind for span in diff.spans] == [ChangeKind.CHANGED]
    assert diff.added_chars > diff.removed_chars
    assert "third" in diff.spans[0].b_text
    assert diff.spans[0].a_text == "second"

    # A change in the middle, where no boundary is involved, is a plain forward
    # diff with nothing removed.
    (root / "notes.md").write_text("# Notes\n\nfirst\ninserted\nsecond\nthird\n")
    await harness.ingest()
    history = await evolution.version_history(sessions, source.id, "notes.md")
    appended = await evolution.diff_versions(sessions, history[2].id, history[3].id)
    assert [span.kind for span in appended.spans] == [ChangeKind.ADDED]
    assert appended.removed_chars == 0


async def test_diff_spans_map_onto_the_chunks_that_contain_them(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    """An edit late in a long document affects late chunks and not early ones.

    The mapping is against `char_start`/`char_end`, which tile the document, and
    not against the chunk's stored text — that carries an overlap head borrowed
    from its predecessor, so matching on text would report the chunk *before* an
    edit as affected every single time.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    paragraph = (
        "The worker claims a task from the queue and holds a lease on it while "
        "the handler runs to completion. Renewing that lease is how a long task "
        "keeps its hold on the work it started. "
    )
    original = "\n\n".join(f"## Section {n}\n\n{paragraph * 3}" for n in range(12))
    (root / "long.md").write_text(original + "\n")

    source = await add_source(sessions, "spans", root)
    harness = build_harness(root, tmp_path / "blobs", sessions, source, settings)
    await harness.ingest()

    # One edit, in the last section, so the affected chunks must be the last few.
    (root / "long.md").write_text(
        original + "\n\n## Appendix\n\nA new closing section about fencing tokens.\n"
    )
    await harness.ingest()

    history = await evolution.version_history(sessions, source.id, "long.md")
    diff = await evolution.diff_versions(sessions, history[0].id, history[1].id)

    assert diff.affected_chunks, "an appended section must land in some chunk"
    assert history[1].chunks > 1, "the fixture has to be long enough to split"

    async with sessions() as session:
        chunks = list(
            (
                await session.execute(
                    select(models.MemoryChunk)
                    .where(models.MemoryChunk.memory_id == history[1].id)
                    .order_by(models.MemoryChunk.ordinal)
                )
            ).scalars()
        )

    affected = {chunk.ordinal for chunk in diff.affected_chunks}
    # The edit is at the end, so the first chunk cannot be affected and the last
    # must be.
    assert 0 not in affected
    assert chunks[-1].ordinal in affected

    # Every reported chunk genuinely overlaps a span, checked here rather than
    # trusted: this is the assertion that would catch an off-by-one in the
    # offset arithmetic.
    by_ordinal = {chunk.ordinal: chunk for chunk in chunks}
    for reported in diff.affected_chunks:
        chunk = by_ordinal[reported.ordinal]
        assert any(
            span.b_start < chunk.char_end and span.b_end > chunk.char_start
            for span in diff.spans
        ), f"chunk #{reported.ordinal} was reported but overlaps no span"

    # And nothing that overlaps was left out.
    for chunk in chunks:
        overlaps = any(
            span.b_start < chunk.char_end and span.b_end > chunk.char_start
            for span in diff.spans
        )
        assert overlaps == (chunk.ordinal in affected)


async def test_a_trivial_diff_says_so_without_asking_the_model(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    """"No substantive change" is decided from the spans, never generated.

    A model handed an empty diff and asked what changed will answer, because
    answering is what it is for, and the answer will be fluent and invented. The
    fake here raises if it is called at all, so the test fails on the *attempt*
    rather than on the shape of the text that came back.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    body = "# Doc\n\nOne unchanging paragraph of text.\n"
    (root / "same.md").write_text(body, newline="")

    source = await add_source(sessions, "trivial", root)
    harness = build_harness(root, tmp_path / "blobs", sessions, source, settings)
    await harness.ingest()
    (root / "same.md").write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
    await harness.ingest()

    history = await evolution.version_history(sessions, source.id, "same.md")
    diff = await evolution.diff_versions(sessions, history[0].id, history[1].id)
    assert diff.is_empty

    model: LanguageModel = RefusesToBeCalled()
    summary = await evolution.SummarizeChange(sessions, model)(diff)

    assert summary.text == evolution.NO_SUBSTANTIVE_CHANGE
    assert summary.trivial is True
    assert summary.grounding.grounded is True

    # Nothing was written to the cache either: there is no model call to save,
    # and a row here would be a stored answer to a constant-time question.
    async with sessions() as session:
        assert (
            await session.execute(select(models.ChangeSummary))
        ).scalars().first() is None

    # A real diff does reach the model, so the guard above is about emptiness
    # rather than about the summarizer never running.
    (root / "same.md").write_text(body + "\nA genuinely new sentence about leases.\n")
    await harness.ingest()
    history = await evolution.version_history(sessions, source.id, "same.md")
    real = await evolution.diff_versions(sessions, history[1].id, history[2].id)
    assert not real.is_empty

    echo = Echoes("A sentence about leases was added.")
    produced = await evolution.SummarizeChange(sessions, echo)(real)
    assert echo.calls == 1
    assert produced.text == "A sentence about leases was added."
    assert produced.cached is False

    # Second call for the same pair reads the cache rather than the model.
    again = await evolution.SummarizeChange(sessions, echo)(real)
    assert echo.calls == 1
    assert again.cached is True
    assert again.text == "A sentence about leases was added."

    # `refresh` regenerates *and replaces*. The first version of this stored with
    # `ON CONFLICT DO NOTHING` on both paths, so a refresh paid for the call,
    # returned the new text to its caller, and left the old row for the next
    # reader — a fix that looked applied and was not.
    better = Echoes("A lease sentence was appended, precisely.")
    refreshed = await evolution.SummarizeChange(sessions, better)(real, refresh=True)
    assert better.calls == 1
    assert refreshed.cached is False
    assert refreshed.text == "A lease sentence was appended, precisely."

    after_refresh = await evolution.SummarizeChange(sessions, better)(real)
    assert better.calls == 1
    assert after_refresh.cached is True
    assert after_refresh.text == "A lease sentence was appended, precisely."


def test_diff_spans_is_line_oriented_and_keeps_exact_offsets() -> None:
    """The pure half, without a database.

    Offsets are recovered by summing line lengths, which is the step that would
    silently drift if `keepends` were ever dropped — the spans would still be in
    the right order and every offset after the first change would be wrong by the
    number of newlines before it.
    """
    before = "alpha\nbeta\ngamma\n"
    after = "alpha\nBETA\ngamma\ndelta\n"

    spans = diff_spans(before, after)

    assert [span.kind for span in spans] == [ChangeKind.CHANGED, ChangeKind.ADDED]
    changed, added = spans
    assert before[changed.a_start : changed.a_end] == "beta\n"
    assert after[changed.b_start : changed.b_end] == "BETA\n"
    # An insertion is a zero-width range in the old text, at the point it landed.
    assert added.a_start == added.a_end == len(before)
    assert after[added.b_start : added.b_end] == "delta\n"

    assert diff_spans(before, before) == []
