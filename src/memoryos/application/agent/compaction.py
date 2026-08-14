"""What the model still sees of hops it has already taken.

**Tool results accumulate, and by hop four the window is mostly things the model
has already read.** A `search_memories` result is around 700 tokens; six of them
plus six tool schemas plus the question is a prompt where the instruction not to
fabricate is one sentence in eight thousand tokens of retrieved text. Something
has to go, and *which* thing decides whether the agent gets stupider or cheaper.

### Compacted, not truncated

Cutting the oldest results at a character count would leave the model reading a
sentence that stops mid-clause and a locator missing its closing paren, and it
would do so silently. Instead each older step becomes a **finding**: the tool
that was called, what the model said it was looking for, the first lines of what
came back, and the citation locators. It is short, it is complete, and it is
still attributable.

The most recent two results stay **verbatim**, which is M7.1's rule and not an
arbitrary one: the hop the model is about to plan is nearly always shaped by the
last result, and a summary of the thing you are reasoning about right now is the
one summary that costs you the answer.

### Citations survive compaction

**A finding that loses its provenance is a fabrication waiting to happen.** By
hop five the model is writing sentences about material it can no longer see, and
if the compacted form dropped the locators, every one of those sentences would be
uncited by construction — fluent, corpus-derived, and impossible to check. So the
locator list is part of the finding and is never what gets dropped to save space:
if a finding does not fit with its citations, the whole finding goes.

The `Citation` objects themselves are kept on the trajectory regardless, so the
answer's citation list is complete even for hops the model can no longer read.
That is a different guarantee from this one and both are needed: the trajectory's
citations let a *reader* check the answer, and the locators in the prompt are
what let the *model* attribute a claim while writing it.

### The budget is the real tokenizer's, and it drops whole findings

Same rules as M2.6: `count_tokens` from the configured counter rather than
`len(text) // 4`, and an item that does not fit is dropped entire. Newest first,
because the recent hops are the ones the next hop builds on — and when anything
is dropped the rendered block says how many, because a model reasoning over a
silently shortened history is a model that will report a partial search as a
complete one.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:  # pragma: no cover - types only
    from memoryos.application.agent.planner import Step

logger = structlog.get_logger(__name__)

# How much of a compacted result's own text a finding keeps.
#
# Every tool's rendering puts the shape of the answer in its first lines — "5
# memories for 'chunking'", then the numbered entries — so a head is a better
# summary than any middle would be, and it needs no second model call to
# produce. Roughly one entry's worth.
FINDING_CHARS = 420

# Locators per finding. Enough to attribute a claim, bounded because a
# `get_memory` on a large file can carry a dozen and six of those is the budget.
FINDING_LOCATORS = 4


class Counter(Protocol):
    """The counting half of `TokenCounter`.

    Narrower than the port on purpose: compaction has no business knowing a
    model's sequence limit, and a protocol that mentioned it would make this
    untestable without one.
    """

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class Finding:
    """One earlier hop, small enough to carry and still attributable."""

    hop: int
    tool: str
    arguments: dict[str, object]
    # What the model said before making the call. Kept because the chain of
    # intent is what makes the finding list read as a plan rather than as a pile
    # of results — and rendered under a label that says it was the model's own
    # words, so a later hop cannot mistake its own speculation for something a
    # tool returned.
    thought: str
    summary: str
    locators: tuple[str, ...]
    # True when the tool said it had more than it returned. Carried through
    # compaction because "we saw part of it" is exactly the qualifier an answer
    # written five hops later needs and cannot recover.
    truncated: bool

    def render(self) -> str:
        head = f"hop {self.hop} · {self.tool}({_args(self.arguments)})"
        lines = [head]
        if self.thought:
            lines.append(f"    you said: {self.thought}")
        lines.append(f"    returned: {self.summary}")
        if self.truncated:
            lines.append("    (that result was truncated — there was more)")
        if self.locators:
            lines.append("    sources: " + "; ".join(self.locators))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Compacted:
    """The history as the next prompt will carry it."""

    findings: list[Finding] = field(default_factory=list)
    # Steps whose results are replayed to the provider unchanged.
    verbatim: list["Step"] = field(default_factory=list)
    # The hop number of the first verbatim step. Carried rather than derived
    # from `len(findings)`, which is wrong the moment the budget drops one: the
    # hops still happened, and renumbering them to close the gap would tell the
    # model it made fewer calls than it did.
    verbatim_from: int = 1
    # Findings that did not fit the budget. Reported to the model in words, not
    # only counted here.
    dropped: int = 0
    tokens: int = 0

    def render(self) -> str:
        """The findings block for the user message, or empty when there is none."""
        if not self.findings and not self.dropped:
            return ""
        parts = [finding.render() for finding in self.findings]
        if self.dropped:
            parts.append(
                f"({self.dropped} earlier finding(s) dropped for space. What you "
                "have here is not everything you found.)"
            )
        return "\n".join(parts)


def compact(
    steps: Sequence["Step"],
    *,
    counter: Counter,
    budget: int,
    keep_verbatim: int = 2,
) -> Compacted:
    """Older steps as findings, the most recent few unchanged.

    Steps that made no tool call — the model narrating without acting — are not
    findings and are dropped here. There is nothing to attribute and nothing to
    replay, and carrying them would spend the budget on the agent's opinion of
    its own progress.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    acted = [step for step in steps if step.tool is not None and step.result is not None]
    verbatim = acted[-keep_verbatim:] if keep_verbatim else []
    older = acted[: len(acted) - len(verbatim)]

    candidates = [_finding(step, hop) for hop, step in enumerate(older, start=1)]

    # Newest first, so what survives a tight budget is what the next hop is most
    # likely to build on. The kept ones are put back in order afterwards: a
    # history the model reads backwards is one it will narrate backwards.
    kept: list[Finding] = []
    used = 0
    dropped = 0
    for finding in reversed(candidates):
        cost = counter.count_tokens(finding.render())
        if used + cost > budget:
            # Whole, and the loop continues: a shorter finding further back may
            # still fit, and stopping here would leave budget unspent. M2.6's
            # rule, for M2.6's reason.
            dropped += 1
            continue
        used += cost
        kept.append(finding)
    kept.reverse()

    if dropped:
        logger.info(
            "agent.findings_dropped", dropped=dropped, kept=len(kept), budget=budget
        )
    return Compacted(
        findings=kept,
        verbatim=verbatim,
        verbatim_from=len(older) + 1,
        dropped=dropped,
        tokens=used,
    )


def _finding(step: "Step", hop: int) -> Finding:
    assert step.tool is not None and step.result is not None  # `compact` filtered
    return Finding(
        hop=hop,
        tool=step.tool,
        arguments=dict(step.args),
        thought=_head(step.thought, 200),
        summary=_head(step.result.content, FINDING_CHARS),
        locators=tuple(
            citation.locator for citation in step.result.citations[:FINDING_LOCATORS]
        ),
        truncated=step.result.truncated,
    )


def _head(text: str, limit: int) -> str:
    """The first `limit` characters, cut at a line or word rather than mid-token.

    Not `_clip`'s whitespace collapse: a tool result's line structure is what
    makes it readable at a glance, and flattening a five-entry search result into
    one paragraph costs more comprehension than it saves tokens.
    """
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    window = stripped[:limit]
    cut = max(window.rfind("\n"), window.rfind(" "))
    return (window[:cut] if cut > limit // 2 else window).rstrip() + " …"


def _args(arguments: dict[str, object]) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in sorted(arguments.items()))
