"""Choosing what the model is allowed to see, and numbering it.

Three rules, each defending against a specific way this goes wrong.

**Count with a real tokenizer.** M1.6.1 was a chunker sized in one unit against
a model reading in another; the text past the window was discarded silently and
retrieval was quietly worse for a milestone and a half. A character-based
estimate here has the same shape: the context overflows, the provider truncates
from one end, and the model answers from a prompt missing the passage that
mattered — with no error anywhere.

**Drop whole passages, never truncate one.** A passage cut mid-sentence is worse
than an absent one. The model cannot tell that the sentence was severed, so it
completes the thought from its own training data, and the result is a fabricated
claim carrying a citation to a real passage — the single most convincing kind of
wrong answer this system can produce.

**Return what made the cut.** The answer may only cite passages the model
actually saw, so the verifier needs the same list the prompt was built from.
Deriving it twice is how the two drift apart.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog

from memoryos.application.ports import TokenCounter
from memoryos.application.search import MemoryHit

logger = structlog.get_logger(__name__)

# Tokens of passages, not of the whole prompt. The system prompt and the
# question add a few hundred more; the budget is deliberately well inside any
# modern context window, because a fuller prompt is not a better one — the
# instruction to refuse competes with every additional passage that looks
# vaguely relevant.
DEFAULT_TOKEN_BUDGET = 6000


@dataclass(frozen=True, slots=True)
class Passage:
    """One numbered passage, as the model will see it."""

    number: int
    hit: MemoryHit
    text: str
    tokens: int

    @property
    def label(self) -> str:
        return f"[{self.number}] {self.hit.source_name}::{self.hit.external_key}"

    def render(self) -> str:
        return f"{self.label}\n{self.text}"


@dataclass(frozen=True, slots=True)
class AssembledContext:
    passages: list[Passage] = field(default_factory=list)
    # Hits that did not fit. Kept rather than discarded so a caller can say the
    # budget bound the answer, instead of the answer silently resting on less
    # than was retrieved.
    dropped: list[MemoryHit] = field(default_factory=list)
    token_budget: int = DEFAULT_TOKEN_BUDGET
    tokens_used: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.passages

    @property
    def valid_indices(self) -> set[int]:
        """The citation indices an answer is allowed to use."""
        return {passage.number for passage in self.passages}

    def render(self) -> str:
        return "\n\n".join(passage.render() for passage in self.passages)

    def passage(self, number: int) -> Passage | None:
        for candidate in self.passages:
            if candidate.number == number:
                return candidate
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "passages": len(self.passages),
            "dropped": len(self.dropped),
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
        }


def assemble_context(
    hits: Sequence[MemoryHit],
    *,
    counter: TokenCounter,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> AssembledContext:
    """Numbered passages from the highest-ranked hits that fit the budget.

    Rank order, because the budget will run out and what it runs out on should
    be the material retrieval was least sure about.

    A hit contributes the text of its best-scoring chunk — the one that put it
    where it is. Concatenating every matched chunk would spend the budget on
    near-duplicates, since chunks of one memory overlap by design.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    passages: list[Passage] = []
    dropped: list[MemoryHit] = []
    used = 0

    for hit in hits:
        text = _best_text(hit)
        if not text:
            dropped.append(hit)
            continue

        tokens = counter.count_tokens(text)
        if used + tokens > token_budget:
            # Dropped whole, and the loop continues rather than stopping: a
            # short passage further down may still fit where a long one did not,
            # and there is no reason to spend the remaining budget on nothing.
            dropped.append(hit)
            continue

        used += tokens
        passages.append(
            Passage(number=len(passages) + 1, hit=hit, text=text, tokens=tokens)
        )

    logger.info(
        "context.assembled",
        passages=len(passages),
        dropped=len(dropped),
        tokens_used=used,
        token_budget=token_budget,
    )
    return AssembledContext(
        passages=passages,
        dropped=dropped,
        token_budget=token_budget,
        tokens_used=used,
    )


def _best_text(hit: MemoryHit) -> str:
    """The chunk that set this hit's score, with its borrowed head removed.

    The overlap prefix is a copy of the previous chunk's tail. Including it
    spends budget on text the model may already have seen and invites it to
    attribute a neighbouring passage's content to this one.
    """
    if not hit.matched_chunks:
        return ""
    best = max(hit.matched_chunks, key=lambda chunk: chunk.score)
    return best.text[best.prefix_chars :].strip()
