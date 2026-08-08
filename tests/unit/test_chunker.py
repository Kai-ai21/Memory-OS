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
from tests.support.fakes import FakeEmbedder

# A stand-in tokenizer: deterministic, and with a configurable window so the
# boundary can be exercised without loading a model.
COUNTER = FakeEmbedder()

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


def test_the_version_encodes_the_parameters_and_the_window() -> None:
    # The window is in the string because a model with a different one produces
    # differently-sized chunks; chunks sized for the old window are stale even
    # when nothing else changed.
    version = StructuralChunker(FakeEmbedder(max_sequence_tokens=512)).version
    assert version == (
        "structural-v2:model_window=512:target=396:overlap=47:min=79:max=496"
    )


def test_sizes_are_derived_from_the_models_window() -> None:
    # Derived rather than chosen, so a model swap re-derives correct sizes
    # instead of relying on somebody remembering to.
    small = StructuralChunker(FakeEmbedder(max_sequence_tokens=256)).config
    assert small.maximum == 256 - 16
    assert small.target == int(small.maximum * 0.8)
    assert small.overlap == int(small.target * 0.12)
    assert small.minimum == int(small.target * 0.2)


def test_a_different_window_changes_the_version() -> None:
    assert (
        StructuralChunker(FakeEmbedder(max_sequence_tokens=256)).version
        != StructuralChunker(FakeEmbedder(max_sequence_tokens=512)).version
    )


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
    assert StructuralChunker(COUNTER, config).version != StructuralChunker(COUNTER).version


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def test_prose_chunks_never_exceed_the_ceiling() -> None:
    chunker = StructuralChunker(COUNTER)
    chunks = chunker.chunk(document(SENTENCE * 500))

    assert chunks
    assert max(chunk.token_count for chunk in chunks) <= chunker.max_tokens


def test_a_single_enormous_sentence_is_split_anyway() -> None:
    # Minified text, a giant table row: no sentence boundary to use, so the
    # ceiling wins over the preference for clean breaks.
    chunker = StructuralChunker(COUNTER)
    chunks = chunker.chunk(document("word " * 4000))

    assert max(chunk.token_count for chunk in chunks) <= chunker.max_tokens


def test_a_short_document_is_one_chunk() -> None:
    chunks = StructuralChunker(COUNTER).chunk(document("A short note."))
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0


def test_an_empty_document_produces_no_chunks() -> None:
    assert StructuralChunker(COUNTER).chunk(document("")) == []
    assert StructuralChunker(COUNTER).chunk(document("   \n\n  ")) == []


def test_ordinals_are_contiguous_from_zero() -> None:
    chunks = StructuralChunker(COUNTER).chunk(document(SENTENCE * 400))
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_every_chunk_satisfies_the_schema_constraints() -> None:
    # memory_chunks has CHECKs for these; producing a chunk that violates one
    # would fail at insert time instead of here.
    chunks = StructuralChunker(COUNTER).chunk(document(SENTENCE * 300))
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
    chunks = StructuralChunker(COUNTER).chunk(doc)

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

    chunks = StructuralChunker(COUNTER).chunk(doc)

    assert len(chunks) < len(doc.structure)
    assert all(chunk.token_count >= 20 for chunk in chunks)


def test_merging_never_manufactures_an_oversized_chunk() -> None:
    source = "".join(f"## S{n}\n\n{SENTENCE * 8}\n\n" for n in range(20))
    chunker = StructuralChunker(COUNTER)
    chunks = chunker.chunk(markdown(source))
    assert max(chunk.token_count for chunk in chunks) <= chunker.max_tokens


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def test_chunks_after_the_first_carry_an_overlap_prefix() -> None:
    doc = document(SENTENCE * 300)
    chunks = StructuralChunker(COUNTER).chunk(doc)

    assert len(chunks) > 1
    first, second = chunks[0], chunks[1]
    # The prefix is the tail of the preceding chunk: a concept spanning the
    # boundary appears in full in at least one of them.
    prefix = second.text[: -(second.char_end - second.char_start)]
    assert prefix
    assert first.text.endswith(prefix)


def test_the_overlap_starts_at_a_sentence_boundary() -> None:
    doc = document(SENTENCE * 300)
    chunks = StructuralChunker(COUNTER).chunk(doc)

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
    chunks = StructuralChunker(COUNTER, config).chunk(doc)

    for chunk in chunks:
        assert chunk.text == doc.text[chunk.char_start : chunk.char_end]


# --------------------------------------------------------------------------
# Offsets
# --------------------------------------------------------------------------


def test_offsets_round_trip_for_a_realistic_document() -> None:
    doc = markdown(
        "---\ntitle: T\n---\n\n# One\n\n" + SENTENCE * 40 + "\n\n## Two\n\n" + SENTENCE * 40
    )
    chunks = StructuralChunker(COUNTER).chunk(doc)

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
    for chunk in StructuralChunker(COUNTER).chunk(doc):
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

    for chunk in StructuralChunker(COUNTER).chunk(doc):
        assert round_trips(doc, chunk)


@settings(max_examples=100, deadline=None)
@given(text=st.text(min_size=0, max_size=2000))
def test_chunks_never_overlap_their_bodies(text: str) -> None:
    # Bodies must tile the text without double-counting, or a citation could
    # point at a span two chunks both claim.
    chunks = StructuralChunker(COUNTER).chunk(document(text))
    for previous, chunk in itertools.pairwise(chunks):
        assert chunk.char_start >= previous.char_end


# --------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------


def test_an_oversized_definition_is_split_rather_than_truncated() -> None:
    """M1.4's rule, revised by M1.6.1.

    Keeping an oversized function whole did not keep it whole — the model read
    its first N tokens and discarded the rest, silently. Two complete halves
    beat one truncated whole, so the ceiling is now a hard invariant.
    """
    body = "\n".join(f"    value_{n} = compute(value_{n - 1})" for n in range(1, 400))
    source = f"def small():\n    return 1\n\n\ndef enormous():\n{body}\n    return value_1\n"
    doc = python(source)
    chunker = StructuralChunker(COUNTER)

    chunks = chunker.chunk(doc)

    enormous_start = doc.text.index("def enormous():")
    parts = [c for c in chunks if c.char_start >= enormous_start]
    assert len(parts) > 1, "an oversized function must be split, not truncated"
    assert all(c.token_count <= chunker.max_tokens for c in chunks)

    # Together the parts still cover the whole function: split, not dropped.
    assert parts[0].char_start == enormous_start
    assert parts[-1].char_end == len(doc.text)


def test_every_part_of_a_split_definition_names_it() -> None:
    # So a citation can still say which function the span came from.
    body = "\n".join(f"    value_{n} = compute(value_{n - 1})" for n in range(1, 400))
    doc = python(f"def enormous():\n{body}\n    return value_1\n")

    chunks = StructuralChunker(COUNTER).chunk(doc)

    assert len(chunks) > 1
    assert all(chunk.metadata.get("definition") == "enormous" for chunk in chunks)


def test_no_code_chunk_exceeds_the_ceiling() -> None:
    source = "".join(
        f"def fn_{n}():\n"
        + "\n".join(f"    step_{m} = work(step_{m - 1})" for m in range(1, 90))
        + "\n\n\n"
        for n in range(6)
    )
    chunker = StructuralChunker(COUNTER)

    chunks = chunker.chunk(python(source))

    assert chunks
    assert max(c.token_count for c in chunks) <= chunker.max_tokens


def test_every_code_chunk_starts_at_a_definition_or_the_top() -> None:
    source = "".join(f"def fn_{n}():\n    return {n}\n\n\n" for n in range(30))
    doc = python(source)

    boundaries = {0, *(marker.char_offset for marker in doc.structure)}
    for chunk in StructuralChunker(COUNTER).chunk(doc):
        assert chunk.char_start in boundaries


def test_code_offsets_round_trip() -> None:
    source = "".join(f"def fn_{n}():\n    return {n}\n\n\n" for n in range(40))
    doc = python(source)

    for chunk in StructuralChunker(COUNTER).chunk(doc):
        assert round_trips(doc, chunk)


def test_tiny_code_definitions_are_merged() -> None:
    source = "".join(f"def fn_{n}():\n    return {n}\n\n\n" for n in range(30))
    doc = python(source)

    chunks = StructuralChunker(COUNTER).chunk(doc)

    assert len(chunks) < len(doc.structure)


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------


def test_token_counting_is_words_and_punctuation() -> None:
    assert count_tokens("hello world") == 2
    assert count_tokens("hello, world!") == 4
    assert count_tokens("") == 0
    assert count_tokens("   \n  ") == 0


# --------------------------------------------------------------------------
# The ceiling as a hard invariant
# --------------------------------------------------------------------------


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(exclude_categories=["Cs"]), min_size=0, max_size=4000
    )
)
def test_no_chunk_ever_exceeds_the_ceiling(text: str) -> None:
    """The invariant this hotfix exists to establish.

    Exceeding the window is not a large chunk, it is silent data loss: the
    model reads the first N tokens and discards the rest without complaint.
    """
    chunker = StructuralChunker(COUNTER)
    for chunk in chunker.chunk(document(text)):
        assert chunk.token_count <= chunker.max_tokens


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("one enormous word", "x" * 20000),
        ("no whitespace at all", "abc.def.ghi." * 2000),
        ("dense unicode", "日本語のテキストです。" * 800),
        ("emoji", "🙂🎉🚀" * 3000),
        ("mixed scripts", ("Ελληνικά عربى हिन्दी " * 900)),
        ("one long line of code", "x = " + " + ".join(str(n) for n in range(4000))),
    ],
)
def test_pathological_input_still_respects_the_ceiling(label: str, text: str) -> None:
    chunker = StructuralChunker(COUNTER)
    chunks = chunker.chunk(document(text))
    assert all(c.token_count <= chunker.max_tokens for c in chunks), label


@pytest.mark.parametrize("window", [64, 128, 256, 512])
def test_the_ceiling_holds_at_every_window(window: int) -> None:
    chunker = StructuralChunker(FakeEmbedder(max_sequence_tokens=window))
    chunks = chunker.chunk(document(SENTENCE * 200))
    assert chunks
    assert max(c.token_count for c in chunks) <= chunker.max_tokens
    assert chunker.max_tokens <= window


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    lines=st.lists(
        st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=80),
        min_size=1,
        max_size=120,
    )
)
def test_code_respects_the_ceiling_too(lines: list[str]) -> None:
    # Code was the exemption in M1.4 and the worst offender in practice: a
    # 2192-token function had 88% of itself discarded.
    source = "def generated():\n" + "\n".join(f"    {line}" for line in lines)
    chunker = StructuralChunker(COUNTER)
    for chunk in chunker.chunk(python(source)):
        assert chunk.token_count <= chunker.max_tokens
