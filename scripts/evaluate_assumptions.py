"""Which of this project's 37 assumptions held, and which were never tested.

Declared evaluations: somebody went and checked what the corpus, the milestone
reports and the code actually say, and wrote down what they found. Nothing here
came from a model — `assumptions suggest` proposes passages and this is the
judgement, which is the division M5.2 exists to keep.

**The rule applied, stated so it can be argued with.**

1. An assumption is evaluated only when something actually tested it. "Nothing
   has gone wrong" is not evidence that a belief held — it is frequently
   evidence that the belief was never exercised, which is a different fact and
   is recorded by leaving the assumption unevaluated. Twelve of the 37 are left
   that way on purpose.

2. For an assumption that is a conjunction, the **load-bearing clause** decides.
   "`domain/` staying pure is what protects testability; the rest is tidiness"
   is evaluated on the first clause, which is checkable and checked. "k=60
   transfers without tuning, and there is not enough data to tune it anyway" is
   evaluated on the first clause, which nobody tested — so it stays unevaluated
   even though the second half is confirmed.

3. `partially` means the belief was true in some circumstances and not others,
   not that it was partly examined. Exactly one assumption here earns it.

Idempotent: an assumption that already carries a verdict is skipped, so this can
be re-run after a schema change.

    uv run python scripts/evaluate_assumptions.py
"""

import asyncio
from datetime import UTC, datetime

from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.application.assumptions import evaluate
from memoryos.config import get_settings
from memoryos.container import Container
from memoryos.domain.values import AssumptionVerdict

HELD = AssumptionVerdict.HELD
FAILED = AssumptionVerdict.FAILED
PARTIALLY = AssumptionVerdict.PARTIALLY

# When these were checked. `declared`, because a person read the evidence.
EVALUATED_AT = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)

# Keyed by the assumption's statement, which is stable — the statements are
# never edited, because an assumption rewritten in hindsight is a decision
# record arguing with itself.
VERDICTS: dict[str, tuple[AssumptionVerdict, str]] = {
    # ----------------------------------------------------------------------
    # Held: something tested the belief and it survived.
    # ----------------------------------------------------------------------
    "Job throughput stays in the low thousands per second, well under what a "
    "Postgres table can claim.": (
        HELD,
        "Never approached. Every constraint this project actually hit was "
        "elsewhere: the embedder's CPU time first, then the language model's "
        "free-tier token budget. The queue has carried five phases of ingestion "
        "without being the thing anybody waited on.",
    ),
    "Nothing later needs a vector operation pgvector does not have.": (
        HELD,
        "Five phases and the only operations ever needed are ANN search and an "
        "exhaustive scan to measure it against. Nothing has asked for a filter, "
        "a quantisation or an index type pgvector lacks.",
    ),
    "pgvector's HNSW recall at a tunable ef_search is close enough to exact "
    "that the difference does not change what a user sees.": (
        HELD,
        "Measured rather than assumed: `eval-recall` reports recall@10 rising "
        "from 0.94 to 1.00 as ef_search goes 40 to 400. At any workable setting "
        "the approximate index returns what the exhaustive scan returns.",
    ),
    "Postgres is already a hard dependency, so this adds no operational "
    "surface at all.": (
        HELD,
        "No broker was ever added. The only second service in docker-compose is "
        "Neo4j, which arrived with M3.0 and for a different reason.",
    ),
    "The corpus stays small enough that HNSW in Postgres is fast enough — "
    "hundreds of thousands of chunks, not hundreds of millions.": (
        HELD,
        "1,308 chunks, five phases later. Held — and held by never being "
        "stressed, which is worth saying: this is a belief about scale that a "
        "corpus this size cannot really test.",
    ),
    "384 dimensions carry enough signal for a corpus of this size and kind.": (
        HELD,
        "nDCG@10 of 0.788 on the 41-query golden set, with the failures "
        "attributable to retrieval depth and reranking rather than to the "
        "vectors being too small to separate anything.",
    ),
    "Chunking stays deterministic, so an ordinal identifies the same span "
    "after a replay.": (
        HELD,
        "Proven three times. M5.0's replays re-linked decision evidence by "
        "ordinal, and M5.1's re-linked 47 links across two evidence tables with "
        "zero dropped — including chunk-level ones, which resolve only if the "
        "ordinal means the same span on the other side of a rebuild.",
    ),
    "Chunking stays deterministic, so an ordinal identifies the same span "
    "after a rebuild.": (
        HELD,
        "The same evidence as its twin in the golden-set decision, which is why "
        "M5.2's grouping put the two together: 47 evidence links re-resolved by "
        "ordinal across a full replay, none dropped.",
    ),
    "Judgements are worth more than the referential integrity a foreign key "
    "would give.": (
        HELD,
        "Settled by repetition. The natural-key design survived every replay in "
        "M5.0 and M5.1, and the same shape has now been applied three more "
        "times — decision, outcome and assumption evidence all key on "
        "(source_name, external_key, chunk_ordinal) for this reason.",
    ),
    "Rank is the thing both retrievers agree on, and their scores carry "
    "nothing extra worth keeping.": (
        HELD,
        "Hybrid has been the default through four phases and nothing has ever "
        "needed the raw scores back. The `ScoreBreakdown` kept for explanation "
        "is read by humans, not by the fusion.",
    ),
    "Running the model locally on CPU stays fast enough that nothing needs a "
    "GPU or a hosted embedding API.": (
        HELD,
        "A full replay re-embeds the corpus in seconds against the cache and in "
        "minutes without it, on a laptop. No milestone has been slowed by "
        "embedding since M1.6.",
    ),
    "The two retrievers fail in different directions often enough for "
    "agreement to be informative.": (
        HELD,
        "Measured in M2.1 before it was relied on: an opaque SQL token goes "
        "from recall 0.000 on the vector half to 1.000 on the lexical half, and "
        "the paraphrase queries go the other way just as sharply.",
    ),
    "A gain below the floor is genuinely not evidence, so shipping it would be "
    "shipping noise.": (
        HELD,
        "Applied three times since it was written — M2.3b's ranking signals, "
        "M3.5's graph expansion and M4.3's temporal weighting all shipped at "
        "zero on this rule, and none of the three has since been shown to have "
        "left value on the table.",
    ),
    "Nothing downstream needs a provider-specific feature — no "
    "structured-output API, no tool use — so the thin port stays sufficient.": (
        HELD,
        "Five phases of prompts — entity extraction, relationships, change "
        "summaries, grounded answers, decision drafts, outcome judgements — and "
        "every one of them is a system prompt, a user message and JSON parsed "
        "out of a string. The port has not grown a method.",
    ),
    "Keeping the graph strictly a projection stops it becoming a second source "
    "of truth nobody can reconcile.": (
        HELD,
        "Nothing has ever written to Neo4j except the sync, a full replay "
        "rebuilds it from Postgres, and `graph verify` compares the two. The "
        "graph has been wrong and empty at various points; it has never "
        "disagreed with Postgres about something only it knew.",
    ),
    "`domain/` staying pure is the part that actually protects testability; "
    "the rest of the rule is tidiness.": (
        HELD,
        "Evaluated on the load-bearing clause. `domain/` imports nothing from "
        "the other three layers — checked, still true — and the unit suite over "
        "it runs with no database. The second clause is untested and the "
        "verdict does not rest on it.",
    ),
    "Nobody adds an `adapters/` import to `domain/` without review catching "
    "it.": (
        HELD,
        "Checked directly: no module under `domain/` imports from `adapters/` "
        "or `application/` after five phases. Weak evidence for the general "
        "claim — one author, every change through the same head — but the "
        "specific thing it predicted has not happened.",
    ),
    "RRF's conservatism is the right default even when it bounds the upside.": (
        HELD,
        "It did the job it was defended for. M3.5's arithmetic shows a "
        "graph-only candidate cannot reach the middle of a fused list without "
        "being ranked first by the graph, which is what stopped expansion "
        "manufacturing answers on a corpus where the graph saw 13% of the "
        "memories.",
    ),
    # ----------------------------------------------------------------------
    # Failed: something tested the belief and it did not survive.
    # ----------------------------------------------------------------------
    "Cosmetic edits are common enough in a real corpus to be worth "
    "special-casing.": (
        FAILED,
        "M4.2 measured it directly: seven items have two versions, 155 have "
        "one, and the chunk-adoption case occurs zero times in the real corpus. "
        "The mechanism is correct and has never fired.",
    ),
    "The free tier's rate limits are workable for a corpus of this size.": (
        FAILED,
        "The clearest failure in the corpus, and it kept failing. M3.5 reported "
        "entity extraction reaching 21 of 162 memories against a "
        "100,000-token daily cap. M5.1 spent fifteen minutes rate-limited to "
        "extract twelve memories and then exhausted the daily budget outright "
        "at 99,461 of 100,000, mid-milestone.",
    ),
    "Entity extraction covers enough of the corpus for the graph to be dense "
    "rather than thin.": (
        FAILED,
        "13% at M3.5, zero after M5.0's replays truncated the entity tables, "
        "about 7% during M5.1 and zero again after its verification replay. The "
        "graph has never seen most of this corpus.",
    ),
    "Running a second database is worth it for one query shape.": (
        FAILED,
        "Graph-augmented retrieval shipped at weight zero in M3.5 and has "
        "stayed there. The one query shape Neo4j was brought in for has not "
        "been served, and the entity tables it depends on have been empty for "
        "most of two phases.",
    ),
    "The corpus will contain enough typed relationships for depth-2 traversal "
    "to reach something a retriever missed.": (
        FAILED,
        "M3.5 measured 30 relationship rows collapsing to 24 distinct edges "
        "after hub suppression. Depth-2 traversal over 24 edges reaches very "
        "little that a MENTIONS hop did not already reach.",
    ),
    "Someone will re-run this measurement once extraction covers the corpus.": (
        FAILED,
        "Three milestones later nobody has, and coverage has gone down rather "
        "than up. The assumption was recorded at 0.4 and that was generous.",
    ),
    # ----------------------------------------------------------------------
    # Partially: true in some circumstances and not others.
    # ----------------------------------------------------------------------
    "Keeping two implementations is cheap enough that the second one does not "
    "rot.": (
        PARTIALLY,
        "The Gemini adapter still exists, still typechecks and is still wired "
        "into `build_language_model` — it has not rotted in the sense of "
        "breaking. But nothing exercises it: no test runs against it, and when "
        "Groq's daily budget ran out mid-M5.1 the fallback reached for was a "
        "different Groq model rather than the second implementation. It is "
        "maintained and unproven, which is neither of the other two verdicts.",
    ),
}


async def main() -> None:
    settings = get_settings()
    container = Container.build(settings)
    written = 0
    skipped = 0
    missing = 0
    try:
        sessions = container.database.session_factory
        async with sessions() as session:
            rows = list(
                await session.execute(
                    select(
                        models.DecisionAssumption.id,
                        models.DecisionAssumption.statement,
                        models.DecisionAssumption.held,
                    )
                )
            )

        by_statement = {statement: (row_id, held) for row_id, statement, held in rows}
        for statement, (verdict, note) in VERDICTS.items():
            found = by_statement.get(statement)
            if found is None:
                print(f"  no assumption recorded for {statement[:60]!r}")
                missing += 1
                continue
            assumption_id, held = found
            if held is not None:
                skipped += 1
                continue
            await evaluate(
                sessions,
                assumption_id,
                verdict,
                note=note,
                evaluated_at=EVALUATED_AT,
            )
            written += 1
    finally:
        await container.dispose()

    total = len(rows)
    print(
        f"evaluated {written}, skipped {skipped} already judged, "
        f"{missing} statements not found"
    )
    print(
        f"{total - written - skipped} of {total} left unevaluated on purpose — "
        f"nothing has tested them"
    )


if __name__ == "__main__":
    asyncio.run(main())
