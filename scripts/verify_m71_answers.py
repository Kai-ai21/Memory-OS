"""M7.1's five answers, put through M7.2's check.

**The answers are the real ones** — copied from the M7.1 runs recorded in the
README — and the trajectories are *replayed*: the traces record which tool each
hop called and with what arguments, so calling those tools again reproduces the
passages the model actually read. No model is called, which is what makes this
runnable at all on a free tier whose daily budget the original five exhausted.

Two limitations, both real and neither hidden.

**The corpus can move.** A replayed hop returns what the tools return *today*; if
the corpus has changed since those runs the support rate moves with it. That is
why this is a script rather than a test — it measures the check against a fixed
set of answers, and only the answers are fixed.

**The line breaks are reconstructed.** `agent ask` prints through `textwrap.wrap`,
which destroys the model's own line structure before it reaches a terminal, so
the transcripts these were taken from are reflowed prose. The bullet structure is
restored below from the markers still visible in the transcript, because the
sentence splitter breaks on newlines and an answer flattened into one paragraph
would be counted as three claims where the model wrote ten. Where a bullet is
restored it is restored exactly where its `*` appeared.

    uv run python scripts/verify_m71_answers.py
"""

import asyncio
from dataclasses import dataclass

from memoryos.application.agent.planner import Step, StopReason, Trajectory
from memoryos.application.agent.verify import Support, verify
from memoryos.config import Settings
from memoryos.container import Container


@dataclass(frozen=True, slots=True)
class Run:
    label: str
    question: str
    # The hops the trace recorded, replayed by calling the same tools.
    calls: tuple[tuple[str, dict[str, object]], ...]
    answer: str


RUNS = (
    Run(
        label="mistakes",
        question="what mistakes have I repeated",
        calls=(("search_memories", {"query": "repeated mistakes"}),),
        answer=(
            "The repeated mistakes identified in your memories relate to the system "
            "generating or re-showing incorrect or dismissed information.\n"
            "Specifically:\n"
            "*   The system was re-showing the same context, which was addressed by a "
            "`SIMILARITY` threshold to prevent this.\n"
            "*   There was a need to stop the system from repeating and regenerating "
            "claims that were dismissed as wrong. This was addressed by implementing "
            "`dismissed_at` with a paired reason.\n"
            "*   A specific instance of a repeated mistake was the system generating a "
            "false claim that lines in a code diff had been reordered when they were "
            "unchanged. This led to a fix in the prompt to prevent similar "
            '"fabrications."'
        ),
    ),
    Run(
        label="reversed",
        question="which decisions did I later reverse, and what did they have in common",
        calls=(
            (
                "search_memories",
                {"query": "reversed decision OR decision changed OR decision overturned"},
            ),
            ("get_decisions", {"about": "reversed decisions"}),
            ("get_decisions", {"limit": 5}),
        ),
        answer=(
            "I couldn't find any decisions explicitly marked as \"reversed\" in the "
            "corpus. However, I can see some decisions where assumptions \"broke\". "
            "These might be decisions that were later reconsidered or would need to "
            "be.\n"
            "Here are the decisions where assumptions broke:\n"
            "*   **DECISION 019ff905-c5a6-7d30-a77f-41c255651e53 (2026-08-12):** \"Is "
            "the four-layer dependency rule enforced by a tool, or by convention?\"\n"
            "    *   **Assumed:** \"The README stating the rule more strongly than the "
            "code holds it will not mislead anybody who reads both.\" (not yet "
            "evaluated)\n"
            "*   **DECISION 019ff905-c5bf-7fc0-8b63-4bcac7333c45 (2026-08-11):** "
            "\"What answers variable-depth relationship queries — Neo4j, or recursive "
            "CTEs?\"\n"
            "    *   **Assumed:** \"Entity extraction covers enough of the corpus for "
            "the graph to be dense rather than thin.\" (broke)\n"
            "    *   **Assumed:** \"Running a second database is worth it for one "
            "query shape.\" (broke)\n"
            "    *   **Assumed:** \"The corpus will contain enough typed relationships "
            "for depth-2 traversal to reach something a retriever missed.\" (broke)\n"
            "*   **DECISION 019ff905-c58a-7841-add5-3954a5ac7c60 (2026-08-11):** "
            "\"What do a chunk's char_start and char_end index into — the stored chunk "
            "text, or the memory's text?\"\n"
            "    *   **Assumed:** \"`verify-citations` is run often enough to catch the "
            "next drift between the offsets and the text.\" (not yet evaluated)\n"
            "The common theme among the \"broke\" assumptions is that they relate to "
            "the **completeness or effectiveness of data/processes**. For example, the "
            "Neo4j decision's assumptions broke because entity extraction wasn't "
            "sufficient, and the corpus didn't contain enough typed relationships."
        ),
    ),
    Run(
        label="chunking",
        question="what was I working on when the chunking approach changed",
        calls=(
            ("search_memories", {"query": "when the chunking approach changed"}),
            ("get_memory", {"memory_id": "019ffb6f-06df-74f2-8b26-4ac758ebc20b"}),
        ),
        answer=(
            "When the chunking approach changed, specifically around M1.4, the work "
            "involved moving chunks across versions rather than rebuilding them if "
            "the normalized text remained the same, even if the file itself changed. "
            "This was done to honestly represent that there was no semantic change. "
            "This change is documented in "
            "self::src/memoryos/application/evolution.py."
        ),
    ),
    Run(
        label="density",
        question="which parts of the system have the most decisions attached",
        calls=(),
        answer=(
            "I cannot directly answer which parts of the system have the most "
            "decisions attached. The available tools can retrieve decisions, but they "
            "cannot categorize or count"
        ),
    ),
    Run(
        label="abandoned",
        question="what did I abandon and never come back to",
        calls=(("find_gaps", {}),),
        answer=(
            "I haven't abandoned anything for 30 days or more; my activity has been "
            "continuous."
        ),
    ),
)


async def main() -> None:
    container = Container.build(Settings())
    try:
        registry = container.tools()
        rows = []
        for run in RUNS:
            steps = []
            for name, arguments in run.calls:
                result = await registry.call(name, dict(arguments))
                steps.append(
                    Step(thought="", tool=name, args=dict(arguments), result=result)
                )
            trajectory = Trajectory(
                question=run.question,
                steps=steps,
                answer=run.answer,
                stopped_because=StopReason.CONFIDENCE,
            )
            checked = verify(trajectory, container.embedder)
            rows.append((run, checked))

            print(f"\n=== {run.label} ({len(run.calls)} hop(s)) ===")
            print(
                f"  verdict {checked.verdict}   support "
                f"{checked.support_rate:.0%}   direct {checked.direct_rate:.0%}   "
                f"{checked.factual_claims} factual / {checked.connective_claims} connective"
            )
            for claim in checked.claims:
                if not claim.factual:
                    print(f"    ·  {claim.text[:78]}")
                    continue
                mark = {
                    Support.DIRECT: "✓",
                    Support.INFERRED: "~",
                    Support.UNSUPPORTED: "✗",
                }[claim.support]
                print(f"    {mark}  {claim.similarity:.3f}  {claim.text[:72]}")

        supported = sum(
            round(checked.support_rate * checked.factual_claims)
            for _, checked in rows
        )
        factual = sum(checked.factual_claims for _, checked in rows)
        direct = sum(
            1
            for _, checked in rows
            for claim in checked.claims
            if claim.support is Support.DIRECT
        )
        inferred = sum(
            1
            for _, checked in rows
            for claim in checked.claims
            if claim.support is Support.INFERRED
        )
        print(
            f"\nacross five answers: {supported}/{factual} factual claims supported "
            f"({supported / factual:.0%}), {direct} direct, {inferred} inferred, "
            f"{factual - supported} unsupported"
        )
        for verdict in ("grounded", "partial", "ungrounded"):
            count = sum(1 for _, checked in rows if checked.verdict == verdict)
            print(f"  {verdict:12} {count}")
    finally:
        await container.dispose()


if __name__ == "__main__":
    asyncio.run(main())
