"""What actually happened to the sixteen decisions this project recorded.

Declared outcomes, which means somebody looked at each decision, went and
checked what the corpus and the milestone reports actually say, and wrote down
what they found. Not inferred: nothing here came from a model reading a document
that happened to be modified afterwards.

**The rule applied while writing this: a decision has an outcome only when its
own `expected_outcome` has been tested.** "The code still exists and nothing has
crashed" is not a result. Four of the sixteen fail that test and are recorded
`too_early`, which is a verdict rather than a gap — somebody looked, and it is
genuinely too soon to say.

Two are `mixed`, and both are worth reading. `mixed` is not a hedge here: it is
the answer for a decision that achieved exactly what it was for and cost
something the decider had written down as an assumption and turned out to be
wrong about. Collapsing either into `worked` or `failed` would throw away the
only part M5.3 can learn from.

Every outcome carries the date the evidence for it was observed, `declared`,
because a person read it. Evidence links point at memories that exist in this
corpus; several honest outcomes have no link at all, because the thing that
demonstrates them is a command's output rather than a document — `doctor`
reporting zero oversized chunks is not a file.

Idempotent: an outcome whose description is already recorded for that decision
is skipped, so this can be re-run after a schema change.

    uv run python scripts/seed_outcomes.py
"""

import asyncio
from datetime import UTC, datetime

from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.application.outcomes import (
    OutcomeDraft,
    OutcomeEvidenceInput,
    UnresolvedEvidence,
    link_evidence,
    record,
)
from memoryos.config import get_settings
from memoryos.container import Container
from memoryos.domain.values import EvidenceKind, OutcomeVerdict, TimeProvenance

SOURCE = "self"

# When these were checked, which is today rather than when each thing happened.
# `declared` is honest for that: a person asserting "I looked, and here is what
# I found" is exactly what the provenance means, and back-dating each one to a
# guess about when the evidence first existed would be inventing precision.
OBSERVED_AT = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def evidence(*keys: str) -> tuple[OutcomeEvidenceInput, ...]:
    return tuple(
        OutcomeEvidenceInput(source_name=SOURCE, external_key=key) for key in keys
    )


# --------------------------------------------------------------------------
# The outcomes, keyed by the decision's question
# --------------------------------------------------------------------------

SEED: list[dict[str, object]] = [
    {
        "question": "What stores the vectors — Postgres with pgvector, or a dedicated index?",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "Both halves of the expected outcome held. `eval-recall` measures "
            "recall@10 rising from 0.94 to 1.00 as ef_search goes 40 to 400, so "
            "recall is above the 0.95 the decision named at any workable "
            "setting — and no reconciliation code was ever written, because "
            "there has never been a second store to reconcile against. 1,308 "
            "chunks are 100% embedded with no repair pass in the codebase."
        ),
        "evidence": evidence(
            "src/memoryos/adapters/db/vector_store.py",
            "src/memoryos/application/evaluation.py",
        ),
    },
    {
        "question": "What runs background work — a table in Postgres, or a broker?",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "Throughput has never been the binding constraint, exactly as "
            "expected. Every constraint this project actually hit was somewhere "
            "else: the embedder's CPU time, then the language model's free-tier "
            "token budget. `SELECT status, count(*) FROM jobs GROUP BY 1` has "
            "been the whole monitoring story through five phases, and no broker "
            "was ever added."
        ),
        "evidence": evidence(
            "src/memoryos/adapters/db/job_queue.py",
            "src/memoryos/application/worker.py",
        ),
    },
    {
        "question": (
            "Which embedding model, after M1.6.1 found the chunker sized against "
            "the wrong window?"
        ),
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "The truncation class of defect has not recurred, which is the half "
            "of the expected outcome that mattered. `doctor` reports zero chunks "
            "over the model window across the whole corpus, and the composition "
            "root's startup assertion makes a future mismatch a failed boot "
            "rather than silently discarded text."
        ),
        "evidence": evidence(
            "src/memoryos/container.py",
            "src/memoryos/adapters/embedding/sentence_transformers.py",
        ),
    },
    {
        "question": (
            "What do a chunk's char_start and char_end index into — the stored "
            "chunk text, or the memory's text?"
        ),
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "`verify-citations` exists and passes over the corpus, which is "
            "precisely the checkable invariant the decision expected to get. "
            "The offsets and the stored text have not drifted apart again, and "
            "the M2.5 citation UI highlights the right span because "
            "`prefix_chars` is a column rather than something each reader "
            "rediscovers."
        ),
        "evidence": evidence(
            "src/memoryos/application/verify_citations.py",
            "src/memoryos/adapters/chunking/structural.py",
        ),
    },
    {
        "question": "How are two retrievers combined into one ranking?",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "Hybrid is the default and has stayed it through three further "
            "phases. The fused score has needed no renormalisation as the "
            "corpus grew, which is the property the decision was made for — and "
            "the golden-set measurements show the two retrievers failing in "
            "opposite directions, so agreement carries real information."
        ),
        "evidence": evidence(
            "src/memoryos/domain/fusion.py", "src/memoryos/application/search.py"
        ),
    },
    {
        "question": "How does a human judgement identify the search result it is about?",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "Measured twice since, and the pattern has been reused twice more. "
            "M5.0 ran two full replays and the golden set came through both "
            "intact with ids re-resolved — and the same natural-key design was "
            "then applied to `decision_evidence` and `outcome_evidence`, both "
            "of which survive a rebuild for the same reason. A decision whose "
            "shape became the house rule."
        ),
        "evidence": evidence(
            "src/memoryos/application/judgements.py",
            "tests/integration/test_judgements.py",
        ),
    },
    {
        "question": "whether to keep or truncate the cache",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "A full replay of this corpus reports 1,436 cache hits and zero "
            "vectors recomputed, finishing in about seven seconds. That is the "
            "cheap routine check the decision wanted, and `--clear-cache` is "
            "still there for the stronger periodic one."
        ),
        "evidence": evidence("src/memoryos/adapters/db/embedding_cache.py"),
    },
    {
        "question": "how to handle table schema",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "The separate schema has survived a change nobody anticipated. M5.0 "
            "added the first table outside the derived set holding a foreign key "
            "into it, which broke `DROP TABLE public.memories` in the swap — and "
            "the fix was local to `swap_in`, lifting the constraints off and "
            "putting them back, because the workspace's tables are real tables "
            "in a real schema. Suffixed table names would have needed a parallel "
            "definition for every one of them."
        ),
        "evidence": evidence("src/memoryos/adapters/db/shadow.py"),
    },
    {
        "question": "When to build indexes in the shadow workspace",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "`verify-replay` rebuilds the whole 1,308-chunk corpus into a shadow "
            "schema and compares it in about six seconds, which is what makes it "
            "a routine check rather than a ceremony. The deferred HNSW build is "
            "the reason that number is what it is."
        ),
        "evidence": evidence("src/memoryos/adapters/db/shadow.py"),
    },
    {
        "question": "Where to split an oversized definition",
        "verdict": OutcomeVerdict.WORKED,
        "description": (
            "`doctor` reports zero chunks over the model window across the whole "
            "corpus, including the code files with long definitions that "
            "motivated the rule. The ceiling holds as a hard invariant, and no "
            "chunk in the corpus is a fragment of a broken call."
        ),
        "evidence": evidence("src/memoryos/adapters/chunking/structural.py"),
    },
    # ----------------------------------------------------------------------
    # Mixed: achieved its aim, and cost something the decider had written down
    # as an assumption and was wrong about.
    # ----------------------------------------------------------------------
    {
        "question": "Which language model provider answers questions and extracts entities?",
        "verdict": OutcomeVerdict.MIXED,
        "description": (
            "The expected outcome held exactly — provider choice has never "
            "leaked past `build_language_model`, and every use case since takes "
            "a `LanguageModel` without knowing which one it got. What broke is "
            "the assumption recorded beside it at confidence 0.5: 'the free "
            "tier's rate limits are workable for a corpus of this size'. They "
            "are not. M3.5 reported entity extraction reaching 21 of 162 "
            "memories against a 100,000-token daily cap, and M5.1 spent fifteen "
            "minutes rate-limited to extract twelve. The port was right and the "
            "provider is the binding constraint on every milestone that needs a "
            "model."
        ),
        "evidence": evidence(
            "src/memoryos/adapters/llm/groq.py", "src/memoryos/application/ports.py"
        ),
    },
    {
        "question": "What answers variable-depth relationship queries — Neo4j, or recursive CTEs?",
        "verdict": OutcomeVerdict.MIXED,
        "description": (
            "The projection half worked and the retrieval half has not paid for "
            "itself. Postgres-wins-on-disagreement held: nothing writes to Neo4j "
            "except the sync, and a replay rebuilds it. But M3.5 measured "
            "graph-augmented retrieval and shipped it at weight zero, and the "
            "assumption recorded at 0.45 — 'entity extraction covers enough of "
            "the corpus for the graph to be dense rather than thin' — is the one "
            "that broke. Worse than the milestone knew: M5.0's replays emptied "
            "the entity tables entirely and nothing reported it until M5.1 went "
            "looking, so the graph spent a phase seeing none of the corpus at "
            "all. The recursive-CTE alternative would have been enough for "
            "everything this corpus has actually asked."
        ),
        "evidence": evidence("predictions.md"),
    },
    # ----------------------------------------------------------------------
    # Too early: somebody looked, and it is genuinely too soon to say. Not the
    # same as nobody having looked, which is what an absent outcome means.
    # ----------------------------------------------------------------------
    {
        "question": "Is the four-layer dependency rule enforced by a tool, or by convention?",
        "verdict": OutcomeVerdict.TOO_EARLY,
        "description": (
            "Nothing has gone wrong and nothing has tested it. `domain/` is "
            "still pure, no `adapters/` import has crept into it, and the "
            "unenforced half has not visibly cost anything — but this project "
            "has had one author and every change has gone through the same head, "
            "which is exactly the condition under which a convention holds and "
            "tells you nothing about whether it would."
        ),
        "evidence": (),
    },
    {
        "question": (
            "When a file changes cosmetically, are its chunks re-made per version "
            "or adopted by the new one?"
        ),
        "verdict": OutcomeVerdict.TOO_EARLY,
        "description": (
            "The mechanism is correct and has never fired. M4.2 measured it "
            "directly: seven items have two versions, 155 have one, and the "
            "chunk-adoption case occurs zero times in the real corpus. The "
            "assumption it rests on — that cosmetic edits are common enough to "
            "be worth special-casing, recorded at 0.6 — has had no opportunity "
            "to be right or wrong."
        ),
        "evidence": (),
    },
    {
        "question": "Do recency and importance ship as ranking signals?",
        "verdict": OutcomeVerdict.TOO_EARLY,
        "description": (
            "Half the expected outcome is confirmed and the half that matters "
            "cannot be. The weights have stayed at zero on this corpus, as "
            "predicted. Whether 'a corpus with real date spread would move them' "
            "is still untested, because no such corpus has been ingested — the "
            "whole thing still spans two days and eighteen hours of filesystem "
            "mtimes."
        ),
        "evidence": (),
    },
    {
        "question": "Does graph expansion ship on, after M3.5 measured it?",
        "verdict": OutcomeVerdict.TOO_EARLY,
        "description": (
            "The experiment the decision named has not been run. Its expected "
            "outcome was that a larger graph would change the sign of the "
            "per-query numbers without changing the arithmetic about RRF's "
            "conservatism — and extraction coverage has gone down rather than "
            "up since, from 13% to zero after M5.0's replays and back to about "
            "7% now. Nobody has re-measured, and one of its own assumptions "
            "said at 0.4 that somebody would."
        ),
        "evidence": (),
    },
]


async def main() -> None:
    settings = get_settings()
    container = Container.build(settings)
    written = 0
    skipped = 0
    missing = 0
    unlinked = 0
    try:
        sessions = container.database.session_factory
        async with sessions() as session:
            decisions = {
                row[1]: row[0]
                for row in await session.execute(
                    select(models.Decision.id, models.Decision.question)
                )
            }
            existing = {
                (row[0], row[1])
                for row in await session.execute(
                    select(
                        models.DecisionOutcome.decision_id,
                        models.DecisionOutcome.description,
                    )
                )
            }

        for entry in SEED:
            question = str(entry["question"])
            decision_id = decisions.get(question)
            if decision_id is None:
                print(f"  no decision recorded for {question!r}")
                missing += 1
                continue
            description = str(entry["description"])
            if (decision_id, description) in existing:
                skipped += 1
                continue

            verdict = entry["verdict"]
            assert isinstance(verdict, OutcomeVerdict)
            outcome_id = await record(
                sessions,
                decision_id,
                OutcomeDraft(description=description, verdict=verdict),
                observed_at=OBSERVED_AT,
                # A person read the evidence and wrote this down.
                observed_at_source=TimeProvenance.DECLARED,
                evidence_kind=EvidenceKind.DECLARED,
            )
            written += 1

            links = entry["evidence"]
            assert isinstance(links, tuple)
            # Linked one at a time, so a file that has left the corpus costs its
            # own link rather than the whole outcome.
            for link in links:
                try:
                    await link_evidence(sessions, outcome_id, link)
                except UnresolvedEvidence as exc:
                    unlinked += 1
                    print(f"  unlinked: {exc}")
    finally:
        await container.dispose()

    print(
        f"recorded {written}, skipped {skipped} already present, "
        f"{missing} decisions not found, {unlinked} links unresolved"
    )


if __name__ == "__main__":
    asyncio.run(main())
