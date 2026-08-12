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

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError

from memoryos.application.ports import (
    EntityExtractor,
    EntityRef,
    ExtractedEntity,
    ExtractedRelationship,
    LanguageModel,
)
from memoryos.domain.backoff import wait_for
from memoryos.domain.jobs import PermanentError, TransientError
from memoryos.domain.values import EntityType, MemoryKind, Predicate

logger = structlog.get_logger(__name__)

# Retries for one batch before the memory is given up on. The caller retries too,
# per memory; this is the inner loop that keeps a rate limit from re-sending
# batches that already succeeded. See `_with_backoff`.
_BATCH_MAX_ATTEMPTS = 6

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
    dropped_unknown_entity: int = 0
    dropped_self: int = 0

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
            dropped_unknown_entity=(
                self.dropped_unknown_entity + other.dropped_unknown_entity
            ),
            dropped_self=self.dropped_self + other.dropped_self,
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
            results.extend(await self._with_backoff(batch, kind))
        return results

    async def _with_backoff(
        self, batch: list[str], kind: MemoryKind
    ) -> list[list[ExtractedEntity]]:
        """One batch, waiting out a rate limit rather than losing the memory.

        **Retried here, at batch granularity, and that is not a refinement.** The
        caller's retry is per *memory*, so a memory of fifty chunks that is rate
        limited on its seventh batch re-sends the six that already succeeded —
        spending the quota that caused the limit on work already done, which makes
        the next limit arrive sooner and the one after that sooner still. On a
        token-per-minute tier that is the difference between finishing a corpus and
        stalling partway through it: measured on this corpus, a 1,169-chunk
        extraction made three requests in twelve minutes.

        M3.3 learned this and fixed it for relationships — `_with_backoff` there
        says the same thing about chunks. Entity extraction kept the memory-level
        retry, and the fix is the same one.

        `PermanentError` is not caught, here or there. Unparseable JSON has already
        had its own retry, and a fifth attempt at a model that cannot produce JSON
        is four wasted requests.
        """
        for attempt in range(_BATCH_MAX_ATTEMPTS):
            try:
                return await self._extract_one_batch(batch, kind)
            except TransientError as exc:
                if attempt == _BATCH_MAX_ATTEMPTS - 1:
                    raise
                # The provider's own number when it gave one. A token-per-minute
                # window slides, and Groq says how far.
                delay = wait_for(exc, attempt)
                logger.info(
                    "extract.rate_limited",
                    waiting_seconds=round(delay),
                    chunks=len(batch),
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    @property
    def relationship_version(self) -> str:
        """Identifies the relationship extractor, separately from the entity one.

        Separate because the two are separately re-runnable: improving the
        relationship prompt must not invalidate every entity mention and force a
        corpus-wide re-extraction of both.
        """
        return f"rel-{RELATIONSHIP_PROMPT_VERSION}:{self._model.model_id}"

    async def extract_relationships(
        self, text: str, entities: Sequence[EntityRef]
    ) -> list[ExtractedRelationship]:
        """Typed, directed claims between the supplied entities only.

        **Entities are offered by number, and only numbers come back.** That is
        what makes an invented endpoint structurally impossible rather than
        merely discouraged: the model never writes an entity name, so there is
        no name to hallucinate. An out-of-range index is the residual failure,
        and it is dropped and counted below.

        Fewer than two entities means there is nothing to relate, and asking is
        a request spent to be told so.
        """
        if len(entities) < 2:
            return []

        user = (
            "Entities known to appear in the text:\n"
            f"{_render_entities(entities)}\n\n"
            f"Text:\n{text}"
        )
        raw = await self._model.complete(
            _RELATIONSHIP_SYSTEM, user, max_tokens=self._max_tokens
        )
        self._record(
            calls=1,
            prompt_chars=len(_RELATIONSHIP_SYSTEM) + len(user),
            response_chars=len(raw),
        )

        parsed = _parse_relationships(raw)
        if parsed is None:
            logger.warning("extract.relationships_unparseable_retrying", chars=len(raw))
            raw = await self._model.complete(
                _RELATIONSHIP_SYSTEM + _RETRY_REMINDER,
                user,
                max_tokens=self._max_tokens,
            )
            self._record(calls=1, retries=1, response_chars=len(raw))
            parsed = _parse_relationships(raw)

        if parsed is None:
            raise PermanentError(
                f"{self._model.model_id} returned unparseable JSON twice for a "
                f"relationship extraction; retrying will not fix it. "
                f"First 200 characters: {raw[:200]!r}"
            )

        return self._verify_relationships(parsed, entities, text)

    def _verify_relationships(
        self,
        parsed: "_RelationshipResponse",
        entities: Sequence[EntityRef],
        text: str,
    ) -> list[ExtractedRelationship]:
        """Keep only claims whose endpoints and predicate are real."""
        verified: list[ExtractedRelationship] = []
        seen: set[tuple[UUID, str, UUID]] = set()

        for claim in parsed.relationships:
            self._record(returned=1)

            if claim.confidence < self._min_confidence:
                self._record(dropped_low_confidence=1)
                continue

            predicate = _predicate_of(claim.predicate)
            if predicate is None:
                self._record(dropped_bad_type=1)
                logger.debug("extract.unknown_predicate", predicate=claim.predicate)
                continue

            subject = _entity_at(entities, claim.subject)
            obj = _entity_at(entities, claim.object)
            if subject is None or obj is None:
                # The model referenced an entity that was not offered. This is
                # the counter the milestone asks for: an edge to an entity that
                # does not exist looks exactly like a real edge until somebody
                # follows it.
                self._record(dropped_unknown_entity=1)
                logger.debug(
                    "extract.relationship_unknown_entity",
                    subject=claim.subject,
                    object=claim.object,
                    supplied=len(entities),
                )
                continue

            if subject.entity_id == obj.entity_id:
                # A self-relationship asserts nothing and the CHECK would refuse
                # it anyway.
                self._record(dropped_self=1)
                continue

            key = (subject.entity_id, predicate.value, obj.entity_id)
            if key in seen:
                continue
            seen.add(key)

            span = _locate(text, claim.evidence.strip()) if claim.evidence else None
            verified.append(
                ExtractedRelationship(
                    subject_id=subject.entity_id,
                    object_id=obj.entity_id,
                    predicate=predicate,
                    confidence=min(1.0, max(0.0, claim.confidence)),
                    evidence=claim.evidence.strip(),
                    char_start=span[0] if span else None,
                    char_end=span[0] + len(span[1]) if span else None,
                )
            )

        return verified

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


# Bumped independently of PROMPT_VERSION, because relationships and entities are
# separately re-runnable: improving one prompt must not invalidate the other's
# output and force a corpus-wide re-extraction of both.
RELATIONSHIP_PROMPT_VERSION = "v1"

_RELATIONSHIP_SYSTEM = """\
You extract relationships between entities from text. You return JSON and \
nothing else.

You are given a numbered list of entities that are known to appear in the text. \
Rules, in order of importance:

1. Use ONLY the supplied entity numbers. Never introduce an entity that is not \
in the list, and never invent a number.
2. Extract ONLY relationships explicitly stated in the text. If the text does \
not assert it, it does not exist. Do not infer from what you know about these \
technologies.
3. Quote the exact sentence or clause from the text that asserts the \
relationship, character for character, in "evidence".
4. Direction matters. "subject" does the thing to "object": for "A supersedes \
B", A is the subject.
5. Use only these predicates: uses, depends_on, part_of, authored_by, \
mentions, supersedes, relates_to. Prefer a specific predicate over relates_to; \
use relates_to only when the text asserts a connection you cannot type.
6. Assign a confidence between 0 and 1. Omit anything below 0.5.

Return this JSON shape and nothing else — no prose, no markdown fences:

{"relationships": [{"subject": <number>, "predicate": "...", \
"object": <number>, "confidence": 0.0, "evidence": "..."}]}

Return an empty list if the text asserts no relationships between the supplied \
entities. That is a common and correct answer."""


class _Relationship(BaseModel):
    subject: int
    predicate: str
    object: int
    confidence: float = Field(default=1.0)
    evidence: str = Field(default="")


class _RelationshipResponse(BaseModel):
    relationships: list[_Relationship] = Field(default_factory=list)


def _render_entities(entities: Sequence[EntityRef]) -> str:
    return "\n".join(
        f"{index}. {entity.name} ({entity.type.value})"
        for index, entity in enumerate(entities)
    )


def _parse_relationships(raw: str) -> "_RelationshipResponse | None":
    text = _FENCE.sub("", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload: Any = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return _RelationshipResponse.model_validate(payload)
    except ValidationError:
        return None


def _predicate_of(raw: str) -> Predicate | None:
    try:
        return Predicate(raw.strip().lower())
    except ValueError:
        return None


def _entity_at(entities: Sequence[EntityRef], index: int) -> EntityRef | None:
    """The supplied entity at this number, or None if the model invented one."""
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    if 0 <= index < len(entities):
        return entities[index]
    return None
