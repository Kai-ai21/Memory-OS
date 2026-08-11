"""Checking that a generated answer stayed inside its evidence.

Pure Python. The system prompt *asks* the model to cite every claim and to
refuse when the passages do not contain the answer; this checks whether it did.
The distinction matters because a prompt is a request and a model under pressure
to be helpful will produce a fluent, well-formed, uncited assertion drawn from
its training data. Nothing in the text marks it as different from a grounded
one — which is precisely why the check has to be mechanical.

Three things are checked, in increasing order of how badly they fail.

**A citation index outside the supplied range** is unambiguous: the model
referenced a passage that was never in the prompt. There is no charitable
reading, and it is the one failure that can be detected with certainty.

**A factual sentence with no citation** is a weaker signal and is treated as
one. It is *flagged, never removed*: quietly deleting a sentence from the middle
of an answer produces prose that reads as complete while missing a step, which
is worse than prose that admits which part is unsupported.

**The citation rate** — cited factual sentences over factual sentences — is the
number to watch across runs. A single uncited sentence is noise; a rate that
falls after a prompt change is a regression.

What is deliberately *not* attempted: checking that a cited passage actually
supports the sentence. That needs a judge, the only available judge is another
language model, and a model grading its own grounding is not evidence. The
milestone's answer to that is a person reading five answers.
"""

import re
from dataclasses import dataclass, field

# `[3]` or `[1, 2]` or `[1][2]`. Bare digits so ordinary prose brackets do not
# register — a passage number is the only thing this system puts in brackets.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Sentence boundaries: terminal punctuation followed by whitespace, or a line
# break. Deliberately simple. A full sentence splitter would be more accurate
# on prose containing abbreviations, and the failure mode of being slightly
# wrong here is a slightly wrong citation *rate*, not a wrong answer.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# A sentence with no letters is punctuation, a list bullet or a stray marker,
# and asking whether it carries a citation is meaningless.
_HAS_LETTERS = re.compile(r"[A-Za-z]")

# Openings that mark a sentence as commentary about the answer rather than a
# claim drawn from the corpus. These are what a *correct* refusal looks like,
# and demanding a citation on "the passages do not say" would penalise the one
# behaviour this milestone most wants.
_NON_FACTUAL_PREFIXES = (
    "i cannot",
    "i could not",
    "i don't",
    "i do not",
    "the passages do not",
    "the passages don't",
    "the provided passages",
    "the retrieved",
    "none of the passages",
    "no passage",
    "there is no",
    "there are no",
    "this is not",
    "based on the passages provided, i",
    "unfortunately",
)

# A refusal is negation plus a verb of having-or-saying: "do not contain",
# "doesn't cover", "cannot answer", "did not mention". One pattern rather than a
# list of literal phrases, and that is a fix rather than a tidy-up.
#
# The list this replaces held "not covered" and not "do not cover", so the model
# saying "The passages do not cover this" — a textbook refusal, and the exact
# wording the system prompt asks for — was recorded as `refused=False`. Nothing
# failed. `refusal_rate` simply under-reported, which is the worst possible place
# for a silent defect: that metric is the evidence that the grounding guardrail
# works at all, and it was quietly counting successful refusals as answers.
#
# Enumerating literal phrases cannot work, because the model paraphrases freely
# and every miss looks like a model that failed to refuse. Matching the shape
# generalises over tense, contraction, and the words in between.
_REFUSAL_PATTERN = re.compile(
    r"\b(?:do(?:es)?\s+not|did\s+not|do(?:es)?n't|didn't|cannot|can\s?not|can't"
    r"|could\s+not|couldn't|unable\s+to|no|none\s+of|nothing\s+in|not)\b"
    # Up to three words of slack, so "do not appear to contain" and "does not
    # explicitly mention" match without the pattern having to know them.
    r"(?:\W+\w+){0,3}\W+"
    r"(?:contain|cover|mention|describe|discuss|address|specify|specified"
    r"|answer|include|provide|state|say|reference|detail|found|available"
    r"|information|passage)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SentenceCheck:
    text: str
    citations: list[int]
    factual: bool
    # True when a factual sentence carries no citation. The answer keeps it and
    # marks it; see the module docstring.
    unsupported: bool


@dataclass(frozen=True, slots=True)
class VerificationResult:
    sentences: list[SentenceCheck] = field(default_factory=list)
    # Indices the answer cited that were never supplied. The unambiguous
    # failure, and the one that must always be zero.
    hallucinated_indices: list[int] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def factual_sentences(self) -> int:
        return sum(1 for sentence in self.sentences if sentence.factual)

    @property
    def supported_sentences(self) -> int:
        return sum(
            1 for sentence in self.sentences if sentence.factual and not sentence.unsupported
        )

    @property
    def citation_rate(self) -> float:
        """Cited factual sentences over factual sentences.

        1.0 for a refusal, which contains no factual claims to cite — scoring
        it zero would make the safest possible answer look like the worst one.
        """
        total = self.factual_sentences
        if total == 0:
            return 1.0
        return self.supported_sentences / total

    @property
    def grounded(self) -> bool:
        """Nothing invented, and every claim attributed."""
        return not self.hallucinated_indices and self.citation_rate == 1.0

    def marked(self, marker: str = " [unsupported]") -> str:
        """The answer with unsupported sentences marked, not removed."""
        return " ".join(
            sentence.text + (marker if sentence.unsupported else "")
            for sentence in self.sentences
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "citation_rate": round(self.citation_rate, 4),
            "factual_sentences": self.factual_sentences,
            "supported_sentences": self.supported_sentences,
            "unsupported_sentences": [
                sentence.text for sentence in self.sentences if sentence.unsupported
            ],
            "hallucinated_indices": list(self.hallucinated_indices),
            "cited_indices": sorted(set(self.cited_indices)),
            "is_refusal": self.is_refusal,
            "grounded": self.grounded,
        }


def verify_citations(answer: str, valid_indices: set[int]) -> VerificationResult:
    """Check an answer against the passages it was given.

    `valid_indices` is what `AssembledContext.valid_indices` returns — the
    numbers the prompt actually contained. Any other index in the answer is a
    fabrication regardless of how plausible the surrounding sentence is.
    """
    stripped = answer.strip()
    if not stripped:
        return VerificationResult()

    # Over the whole answer and before the per-sentence walk, because the
    # refusal decision now depends on whether anything was cited at all.
    refusal = _is_refusal(stripped, cites_nothing=not _indices(stripped))
    sentences: list[SentenceCheck] = []
    hallucinated: list[int] = []
    cited: list[int] = []

    for raw in _SENTENCE.split(stripped):
        text = raw.strip()
        if not text:
            continue

        indices = _indices(text)
        cited.extend(index for index in indices if index in valid_indices)
        hallucinated.extend(index for index in indices if index not in valid_indices)

        factual = _is_factual(text) and not refusal
        sentences.append(
            SentenceCheck(
                text=text,
                citations=indices,
                factual=factual,
                unsupported=factual and not indices,
            )
        )

    return VerificationResult(
        sentences=sentences,
        # Sorted and deduplicated: the same wrong index cited twice is one
        # defect, not two.
        hallucinated_indices=sorted(set(hallucinated)),
        cited_indices=cited,
        is_refusal=refusal,
    )


def _indices(sentence: str) -> list[int]:
    found: list[int] = []
    for match in _CITATION.finditer(sentence):
        found.extend(int(part) for part in match.group(1).split(","))
    return found


def _is_factual(sentence: str) -> bool:
    """Whether this sentence asserts something the corpus should support."""
    if _HAS_LETTERS.search(sentence) is None:
        return False
    lowered = sentence.lower().lstrip("-*# ")
    return not lowered.startswith(_NON_FACTUAL_PREFIXES)


def _is_refusal(answer: str, *, cites_nothing: bool) -> bool:
    """Whether the answer declines rather than asserts.

    Two conditions, and the second is what stops this over-firing. The answer
    must *read* as a decline, and it must cite nothing — because an answer that
    cites [2] and then notes the passages do not cover some adjacent detail is a
    partial answer, not a refusal, and counting it as one would inflate the
    refusal rate with answers that did in fact answer.

    Zero citations alone would be far worse than the bug this fixes: an
    ungrounded fabrication also cites nothing, so the rate would count
    hallucinations as refusals and read best exactly when the system was
    behaving worst.

    Checked on the whole answer rather than per sentence, because a refusal is a
    property of the response: "the passages do not describe X" followed by an
    offer to help with something else is still a refusal, and marking the second
    sentence unsupported would punish exactly the behaviour being asked for.
    """
    return cites_nothing and _REFUSAL_PATTERN.search(answer) is not None
