"""Entity extraction over the `LanguageModel` port.

Provider-agnostic, and that is the point rather than a nicety: this talks to
whatever `MEMOS_LLM_PROVIDER` selected, so switching Groq for Gemini changes
nothing here. A provider-specific extractor would have re-created exactly the
coupling M2.6 spent a milestone removing.

Three things do the real work:

**Offsets are verified, never trusted.** The model returns a name and, if asked,
a position. The position is a guess — models cannot count characters — so it is
discarded and the name is *located* in the text instead. A name that cannot be
found is dropped and counted, because a mention at a fabricated offset is worse
than a missing one: it points at real text that says something else, and the
provenance chain M2.5 built would be reporting a span nobody wrote.

**One retry, then permanent.** Malformed JSON gets a second attempt with a
blunter instruction. A model that returns unparseable output twice will do it a
third time, and burning five worker attempts on it delays every other job in the
queue.

**Batched, because the free tier is the constraint.** 1,308 chunks is 1,308
requests one at a time, which exceeds the daily request cap before it exceeds
anything else. Chunks are numbered in one prompt and the results demultiplexed
by index — and because each chunk's offsets are verified against *that chunk's*
text, a model that misattributes an entity to the wrong index produces a name
that is not found there, which is dropped by the same rule that catches
hallucinated names.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic import BaseModel, Field, ValidationError

from memoryos.application.ports import EntityExtractor, ExtractedEntity, LanguageModel
from memoryos.domain.jobs import PermanentError
from memoryos.domain.values import EntityType, MemoryKind

logger = structlog.get_logger(__name__)

# Bump when the prompt changes in any way that could change what comes back.
# Part of `version`, so a prompt improvement becomes a query — find the mentions
# carrying the old version, redo those — rather than a corpus rebuild.
PROMPT_VERSION = "v1"

# Below this, the model is guessing. The prompt asks for it to be applied, and
# it is enforced here too: an instruction is a request, and a threshold that
# only exists in the prompt is a threshold the model may quietly ignore.
MIN_CONFIDENCE = 0.5

# How many chunks go into one request by default. Eight keeps the prompt well
# inside the context window while cutting request count by the same factor,
# which is what the free tier's per-day request cap actually cares about.
DEFAULT_BATCH_SIZE = 8

# Models wrap JSON in fences however firmly they are told not to.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

_SYSTEM = """\
You extract named entities from text. You return JSON and nothing else.

Rules, in order of importance:

1. Extract ONLY entities explicitly present in the text. Never infer an entity \
that is implied, related, or that you happen to know about. If the text does \
not name it, it does not exist.
2. Return the entity name EXACTLY as it appears in the text, character for \
character. Do not expand abbreviations, fix capitalisation, correct spelling, \
or translate. "postgres" stays "postgres"; do not return "PostgreSQL".
3. Assign a confidence between 0 and 1. Omit anything below 0.5.
4. Use only these types: person, technology, project, organization, concept, \
file, decision. If nothing fits, omit the entity rather than inventing a type.

Return this JSON shape and nothing else — no prose, no markdown fences:

{"results": [{"index": <chunk index>, "entities": [{"name": "...", \
"type": "...", "confidence": 0.0}]}]}

Every chunk index you were given must appear exactly once in "results", with an \
empty "entities" list if it contains no entities."""

# Appended for code, and the reason a `kind` is on the port at all. Without it
# the extractor returns English words from source files — "the", "return",
# "value" — because they are the most frequent tokens and it is trying to be
# helpful.
_CODE_GUIDANCE = """\

This text is source code. Prefer identifiers, function and class names, module \
and library names, and file paths. Do not extract ordinary English words, \
language keywords, or common variable names like "data", "result" or "value"."""

_RETRY_REMINDER = """\

Your previous response was not valid JSON. Return ONLY a JSON object. Start \
your response with { and end it with }. No explanation, no markdown fences, no \
text before or after the JSON."""


class _Entity(BaseModel):
    """One entity as the model returns it, before anything is believed.

    Note what is absent: any offset. The model is not asked where the name
    appears, because it cannot count characters and the answer would have to be
    thrown away. Asking would spend tokens on a field that exists only to be
    ignored.
    """

    name: str
    type: str
    confidence: float = Field(default=1.0)


class _Result(BaseModel):
    index: int
    entities: list[_Entity] = Field(default_factory=list)


class _Response(BaseModel):
    results: list[_Result] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    """What one call cost and what it threw away.

    `dropped_not_found` is the number that matters: it is the rate at which the
    model named something that is not in the text, which is the only direct
    measurement of fabrication this milestone produces.
    """

    calls: int = 0
    retries: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    returned: int = 0
    dropped_not_found: int = 0
    dropped_low_confidence: int = 0
    dropped_bad_type: int = 0

    def merged(self, other: "ExtractionStats") -> "ExtractionStats":
        return ExtractionStats(
            calls=self.calls + other.calls,
            retries=self.retries + other.retries,
            prompt_chars=self.prompt_chars + other.prompt_chars,
            response_chars=self.response_chars + other.response_chars,
            returned=self.returned + other.returned,
            dropped_not_found=self.dropped_not_found + other.dropped_not_found,
            dropped_low_confidence=(
                self.dropped_low_confidence + other.dropped_low_confidence
            ),
            dropped_bad_type=self.dropped_bad_type + other.dropped_bad_type,
        )


class LlmEntityExtractor(EntityExtractor):
    def __init__(
        self,
        model: LanguageModel,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int = 2048,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self._model = model
        self._batch_size = max(1, batch_size)
        self._max_tokens = max_tokens
        self._min_confidence = min_confidence
        # Accumulated across calls and read by the CLI for the cost report.
        self.stats = ExtractionStats()

    @property
    def version(self) -> str:
        """Extractor, prompt, and model — the three things that change output.

        All three, because any one of them changing means a mention carrying
        this string was produced by something else. The model id is in here for
        the same reason it is in the embedding cache key: without it, switching
        providers would silently reuse the old provider's extractions and
        nothing would report that the corpus now holds two incompatible
        opinions.
        """
        return f"llm-{PROMPT_VERSION}:{self._model.model_id}"

    async def extract(
        self, text: str, *, kind: MemoryKind
    ) -> list[ExtractedEntity]:
        """The port's single-text surface. One chunk, one call."""
        batched = await self.extract_batch([text], kind=kind)
        return batched[0]

    async def extract_batch(
        self, texts: list[str], *, kind: MemoryKind
    ) -> list[list[ExtractedEntity]]:
        """Several chunks per request, returning one list per input, in order.

        Beyond the port, and the milestone asks for both: the port's `extract`
        is the contract, and this is what the job uses because a request per
        chunk exhausts the free tier's daily cap on a corpus this size. The
        return is positional rather than keyed, so a caller cannot accidentally
        associate a chunk with another chunk's entities.
        """
        results: list[list[ExtractedEntity]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            results.extend(await self._extract_one_batch(batch, kind))
        return results

    async def _extract_one_batch(
        self, batch: list[str], kind: MemoryKind
    ) -> list[list[ExtractedEntity]]:
        system = _SYSTEM + (_CODE_GUIDANCE if kind is MemoryKind.CODE else "")
        user = _render(batch)

        raw = await self._model.complete(system, user, max_tokens=self._max_tokens)
        self._record(calls=1, prompt_chars=len(system) + len(user), response_chars=len(raw))

        parsed = _parse(raw)
        if parsed is None:
            # One retry, blunter. A model that cannot produce JSON twice will
            # not produce it on the fifth attempt either, and the worker's
            # remaining attempts are better spent on other jobs.
            logger.warning("extract.unparseable_retrying", chars=len(raw))
            raw = await self._model.complete(
                system + _RETRY_REMINDER, user, max_tokens=self._max_tokens
            )
            self._record(
                calls=1,
                retries=1,
                prompt_chars=len(system) + len(user),
                response_chars=len(raw),
            )
            parsed = _parse(raw)

        if parsed is None:
            raise PermanentError(
                f"{self._model.model_id} returned unparseable JSON twice for a "
                f"{len(batch)}-chunk extraction batch; retrying will not fix it. "
                f"First 200 characters: {raw[:200]!r}"
            )

        by_index = {result.index: result.entities for result in parsed.results}
        return [
            self._verify(by_index.get(index, []), text)
            for index, text in enumerate(batch)
        ]

    def _verify(
        self, candidates: list[_Entity], text: str
    ) -> list[ExtractedEntity]:
        """Keep only entities that are really in this text, at a real offset.

        The single most important function in this adapter. Everything upstream
        is a model's opinion; this is where an opinion becomes a fact with a
        span attached, or is thrown away.
        """
        verified: list[ExtractedEntity] = []
        seen: set[tuple[str, int]] = set()

        for candidate in candidates:
            name = candidate.name.strip()
            if not name:
                continue

            self._record(returned=1)

            if candidate.confidence < self._min_confidence:
                self._record(dropped_low_confidence=1)
                continue

            entity_type = _type_of(candidate.type)
            if entity_type is None:
                # A type outside the closed vocabulary. Dropped rather than
                # coerced to `concept`: a model that invented a type has
                # probably invented the entity too, and a wrong type is a
                # filter silently returning the wrong rows.
                self._record(dropped_bad_type=1)
                logger.debug("extract.unknown_type", type=candidate.type, name=name)
                continue

            located = _locate(text, name)
            if located is None:
                # The name is not in the text. Either the model paraphrased it
                # — against instruction 2 — or invented it outright. Both are
                # unusable, and this counter is the fabrication rate.
                self._record(dropped_not_found=1)
                logger.debug("extract.name_not_in_text", name=name)
                continue

            # The surface form is taken from the *text*, never from the model.
            # On a case-insensitive hit the two differ, and storing the model's
            # spelling against the text's offsets would break the one invariant
            # this whole adapter exists to uphold:
            # `text[char_start:char_end] == name`.
            start, surface = located

            # The same name twice at the same offset is one mention; the unique
            # constraint would reject the second anyway, and catching it here
            # keeps the count honest.
            if (surface, start) in seen:
                continue
            seen.add((surface, start))

            verified.append(
                ExtractedEntity(
                    name=surface,
                    type=entity_type,
                    # Clamped rather than rejected: a model returning 1.2 is
                    # expressing certainty in a broken unit, and the CHECK
                    # constraint would refuse the row on a technicality.
                    confidence=min(1.0, max(0.0, candidate.confidence)),
                    char_start=start,
                    char_end=start + len(surface),
                )
            )

        return verified

    def _record(self, **fields: int) -> None:
        self.stats = self.stats.merged(ExtractionStats(**fields))


def _render(batch: list[str]) -> str:
    """The chunks, numbered, with an explicit end marker per chunk.

    Delimiters a model is unlikely to reproduce accidentally, so a chunk
    containing something that looks like a header cannot shift the numbering.
    """
    parts = [
        f"<<<CHUNK {index}>>>\n{text}\n<<<END CHUNK {index}>>>"
        for index, text in enumerate(batch)
    ]
    return (
        f"Extract entities from the following {len(batch)} chunk(s). "
        f"Return one result object per chunk index 0 to {len(batch) - 1}.\n\n"
        + "\n\n".join(parts)
    )


def _parse(raw: str) -> _Response | None:
    """The model's text as a validated response, or None if it is not one.

    None rather than an exception, because the caller's response to "not JSON"
    is to retry once and only then give up — and distinguishing that from a real
    error is the difference between one extra call and five.
    """
    text = _FENCE.sub("", raw).strip()
    if not text:
        return None

    # Models sometimes prepend a sentence despite instruction. Take the outermost
    # braces rather than failing on a response that does contain the object.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        payload: Any = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    try:
        return _Response.model_validate(payload)
    except ValidationError:
        return None


def _type_of(raw: str) -> EntityType | None:
    try:
        return EntityType(raw.strip().lower())
    except ValueError:
        return None


def _locate(text: str, name: str) -> tuple[int, str] | None:
    """Where `name` occurs in `text`, and the text's own spelling of it.

    Returns the offset *and* the surface form, because those two must be stored
    together or not at all — the caller writes the returned string, never the
    model's, so `text[start:start + len(surface)] == surface` holds by
    construction rather than by hope.

    Exact match first, because instruction 2 asks for the surface form verbatim
    and an exact hit needs no judgement. The case-insensitive fallback exists
    because models normalise capitalisation habitually: "Neo4J" for "neo4j" is
    the model being tidy, not the model hallucinating, and dropping it would
    discard a real mention over a capital letter.

    Deliberately no fuzzy matching beyond case. "Close enough" is how a mention
    ends up pointing at a different entity that happens to share a prefix.
    """
    index = text.find(name)
    if index != -1:
        return index, name
    index = text.lower().find(name.lower())
    if index == -1:
        return None
    return index, text[index : index + len(name)]
