"""The structural chunker."""

import itertools

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from memoryos.adapters.chunking.sentences import split_sentences
from memoryos.adapters.chunking.structural import ChunkerConfig, StructuralChunker
from memoryos.adapters.chunking.tokens import count_tokens
from memoryos.adapters.parsers.code import CodeParser
from memoryos.adapters.parsers.markdown import MarkdownParser
from memoryos.application.ports import ParsedDocument, StructureMarker, TextChunk
from memoryos.domain.values import MemoryKind

SENTENCE = "The quick brown fox jumps over the lazy dog and keeps running onward. "


def document(
    text: str,
    structure: list[StructureMarker] | None = None,
    kind: MemoryKind = MemoryKind.NOTE,
) -> ParsedDocument:
    return ParsedDocument(
        text=text, title=None, metadata={}, structure=structure or [], kind=kind
    )


def markdown(source: str) -> ParsedDocument:
    return MarkdownParser().parse(
        source.encode(), media_type="text/markdown", external_key="doc.md"
    )


def python(source: str) -> ParsedDocument:
    return CodeParser().parse(source.encode(), media_type=None, external_key="mod.py")


def round_trips(doc: ParsedDocument, chunk: TextChunk) -> bool:
    """The chunk text ends with exactly the span it claims.

    Ends with rather than equals, because every chunk after the first carries
    an overlap prefix borrowed from its predecessor. char_start/char_end name
    the body alone — the part a citation should highlight.
    """
    return chunk.text.endswith(doc.text[chunk.char_start : chunk.char_end])


# --------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------


def test_the_version_encodes_the_parameters() -> None:
    version = StructuralChunker().version
    assert version == "structural-v1:target=640:overlap=80:min=120:max=1024"


@pytest.mark.parametrize(
    "config",
    [
        ChunkerConfig(target=512),
        ChunkerConfig(overlap=40),
        ChunkerConfig(minimum=200),
        ChunkerConfig(maximum=2048),
    ],
)
def test_changing_any_parameter_changes_the_version(config: ChunkerConfig) -> None:
    # This is what makes improving the chunker a query rather than a corpus
    # rebuild: a stale stamp is findable in SQL.
    assert StructuralChunker(config).version != StructuralChunker().version


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def test_prose_chunks_never_exceed_the_ceiling() -> None:
    doc = document(SENTENCE * 500)
    chunks = StructuralChunker().chunk(doc)

    assert chunks
    assert max(chunk.token_count for chunk in chunks) <= 1024


def test_a_single_enormous_sentence_is_split_anyway() -> None:
    # Minified text, a giant table row: no sentence boundary to use, so the
    # ceiling wins over the preference for clean breaks.
    doc = document("word " * 4000)
    chunks = StructuralChunker().chunk(doc)

    assert max(chunk.token_count for chunk in chunks) <= 1024


def test_a_short_document_is_one_chunk() -> None:
    chunks = StructuralChunker().chunk(document("A short note."))
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0


def test_an_empty_document_produces_no_chunks() -> None:
    assert StructuralChunker().chunk(document("")) == []
    assert StructuralChunker().chunk(document("   \n\n  ")) == []


def test_ordinals_are_contiguous_from_zero() -> None:
    chunks = StructuralChunker().chunk(document(SENTENCE * 400))
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_every_chunk_satisfies_the_schema_constraints() -> None:
    # memory_chunks has CHECKs for these; producing a chunk that violates one
    # would fail at insert time instead of here.
    chunks = StructuralChunker().chunk(document(SENTENCE * 300))
    for chunk in chunks:
        assert chunk.char_end > chunk.char_start
        assert chunk.char_start >= 0
        assert chunk.token_count > 0


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_headings_are_chunk_boundaries() -> None:
    source = "".join(
        f"## Section {n}\n\n{SENTENCE * 12}\n\n" for n in range(6)
    )
    doc = markdown(source)
    chunks = StructuralChunker().chunk(doc)

    # The author already said where the topic changes; the chunker uses that
    # rather than guessing.
    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.char_start :].lstrip().startswith("## Section")


def test_undersized_sections_are_merged() -> None:
    # A heading plus one line is not a retrieval unit: it matches on keyword
    # luck and carries no context.
    source = "".join(f"## S{n}\n\nshort line {n}\n\n" for n in range(12))
    doc = markdown(source)

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) < len(doc.structure)
    assert all(chunk.token_count >= 20 for chunk in chunks)


def test_merging_never_manufactures_an_oversized_chunk() -> None:
    source = "".join(f"## S{n}\n\n{SENTENCE * 8}\n\n" for n in range(20))
    chunks = StructuralChunker().chunk(markdown(source))
    assert max(chunk.token_count for chunk in chunks) <= 1024


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def test_chunks_after_the_first_carry_an_overlap_prefix() -> None:
    doc = document(SENTENCE * 300)
    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) > 1
    first, second = chunks[0], chunks[1]
    # The prefix is the tail of the preceding chunk: a concept spanning the
    # boundary appears in full in at least one of them.
    prefix = second.text[: -(second.char_end - second.char_start)]
    assert prefix
    assert first.text.endswith(prefix)


def test_the_overlap_starts_at_a_sentence_boundary() -> None:
    doc = document(SENTENCE * 300)
    chunks = StructuralChunker().chunk(doc)

    for previous, chunk in itertools.pairwise(chunks):
        prefix_length = len(chunk.text) - (chunk.char_end - chunk.char_start)
        if prefix_length == 0:
            continue
        overlap_start = chunk.char_start - prefix_length
        sentence_starts = {
            previous.char_start + start
            for start, _ in split_sentences(doc.text[previous.char_start : previous.char_end])
        }
        assert overlap_start in sentence_starts


def test_overlap_can_be_switched_off() -> None:
    config = ChunkerConfig(overlap=0)
    doc = document(SENTENCE * 300)
    chunks = StructuralChunker(config).chunk(doc)

    for chunk in chunks:
        assert chunk.text == doc.text[chunk.char_start : chunk.char_end]


# --------------------------------------------------------------------------
# Offsets
# --------------------------------------------------------------------------


def test_offsets_round_trip_for_a_realistic_document() -> None:
    doc = markdown(
        "---\ntitle: T\n---\n\n# One\n\n" + SENTENCE * 40 + "\n\n## Two\n\n" + SENTENCE * 40
    )
    chunks = StructuralChunker().chunk(doc)

    assert chunks
    for chunk in chunks:
        assert round_trips(doc, chunk)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(exclude_categories=["Cs"]), min_size=0, max_size=3000
    )
)
def test_offsets_round_trip_for_any_document(text: str) -> None:
    """The property that makes citations possible.

    `char_start`/`char_end` index into the normalized text. If they ever drift,
    a Phase 2 citation highlights the wrong span — and it would do so silently,
    because the chunk text itself would still look right.
    """
    doc = document(text)
    for chunk in StructuralChunker().chunk(doc):
        assert doc.text[chunk.char_start : chunk.char_end], "empty span"
        assert round_trips(doc, chunk)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    paragraphs=st.lists(
        st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=200),
        min_size=1,
        max_size=25,
    ),
    levels=st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=25),
)
def test_offsets_round_trip_with_headings(paragraphs: list[str], levels: list[int]) -> None:
    source = "".join(
        f"{'#' * level} Heading\n\n{paragraph}\n\n"
        for level, paragraph in zip(levels, paragraphs, strict=False)
    )
    doc = markdown(source)

    for chunk in StructuralChunker().chunk(doc):
        assert round_trips(doc, chunk)


@settings(max_examples=100, deadline=None)
@given(text=st.text(min_size=0, max_size=2000))
def test_chunks_never_overlap_their_bodies(text: str) -> None:
    # Bodies must tile the text without double-counting, or a citation could
    # point at a span two chunks both claim.
    chunks = StructuralChunker().chunk(document(text))
    for previous, chunk in itertools.pairwise(chunks):
        assert chunk.char_start >= previous.char_end


# --------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------


def test_code_is_never_split_mid_function() -> None:
    """A function cut in half embeds as neither a function nor a statement.

    Size variance is the lesser cost, so an oversized function stays whole.
    """
    body = "\n".join(f"    value_{n} = compute(value_{n - 1})" for n in range(1, 400))
    source = f"def small():\n    return 1\n\n\ndef enormous():\n{body}\n    return value_1\n"
    doc = python(source)

    chunks = StructuralChunker().chunk(doc)

    enormous_start = doc.text.index("def enormous():")
    # Whichever chunk holds the giant function holds all of it.
    holder = [c for c in chunks if c.char_start <= enormous_start < c.char_end]
    assert len(holder) == 1
    assert holder[0].char_end == len(doc.text)
    assert holder[0].token_count > 640


def test_every_code_chunk_starts_at_a_definition_or_the_top() -> None:
    source = "".join(f"def fn_{n}():\n    return {n}\n\n\n" for n in range(30))
    doc = python(source)

    boundaries = {0, *(marker.char_offset for marker in doc.structure)}
    for chunk in StructuralChunker().chunk(doc):
        assert chunk.char_start in boundaries


def test_code_offsets_round_trip() -> None:
    source = "".join(f"def fn_{n}():\n    return {n}\n\n\n" for n in range(40))
    doc = python(source)

    for chunk in StructuralChunker().chunk(doc):
        assert round_trips(doc, chunk)


def test_tiny_code_definitions_are_merged() -> None:
    source = "".join(f"def fn_{n}():\n    return {n}\n\n\n" for n in range(30))
    doc = python(source)

    chunks = StructuralChunker().chunk(doc)

    assert len(chunks) < len(doc.structure)


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------


def test_token_counting_is_words_and_punctuation() -> None:
    assert count_tokens("hello world") == 2
    assert count_tokens("hello, world!") == 4
    assert count_tokens("") == 0
    assert count_tokens("   \n  ") == 0
