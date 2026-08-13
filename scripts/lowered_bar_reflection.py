"""Write the strongest pattern the resolution gate refused, so it can be read.

**This script deliberately does the thing the whole milestone is arranged
against, once, on purpose, so that the result can be judged rather than
imagined.** It is not wired into the CLI, nothing calls it, and it exists to
answer one question M5.4's report has to answer: when this system finally does
say something about you, is it insight or is it a horoscope?

On this corpus, `patterns discover` emits nothing. Six candidates, five refused
because the stated confidence falls inside the Wilson interval its own sample
supports, one refused for having no supporting evidence at all. That is the
correct behaviour and it is also a dead end for reading a reflection: there is no
pattern, so there is nothing to describe, so the interesting judgement cannot be
made at all.

So this takes the strongest of the refused candidates — the one with the most
supporting decisions behind it — and writes it into `patterns` exactly as
`discover` would have, **bypassing only the resolution gate**. Every other
property is real: the evidence rows are the actual decisions, the counts are the
actual counts, the confidence is the same formula. What is missing is the one
thing that matters, which is any reason to believe the observed rate differs from
the stated one by more than a sample this size produces by chance.

Two consequences follow and both are the point.

1. The reflection generated from it is a *fair* test of the prose layer. The
   citations are real decisions; only the claim's statistical standing is
   fabricated.
2. The pattern must not survive the reading. Run `memoryos reflections dismiss`
   and `memoryos patterns dismiss` afterwards, or `--undo` here, so the corpus is
   not left holding a claim its own gates refused.

    uv run python scripts/lowered_bar_reflection.py           # write it
    uv run python scripts/lowered_bar_reflection.py --undo    # take it back out
"""

import argparse
import asyncio

from sqlalchemy import delete, select

from memoryos.adapters.db import models
from memoryos.application.patterns import (
    Candidate,
    all_detectors,
    read_corpus,
)
from memoryos.application.patterns import (
    # `_upsert` is private by convention and is reached anyway: building this
    # row any other way would mean two ways of writing a pattern, and the one
    # used here would be the untested one.
    _upsert as upsert_pattern,
)
from memoryos.config import get_settings
from memoryos.container import Container

# Marked on the dismissal reason if `--undo` is used, and printed at the end of a
# write, so a row that escapes into a real corpus can be found by grep.
MARKER = "[lowered-bar]"


def strongest_refused(candidates: list[Candidate]) -> Candidate | None:
    """The refused candidate with the most distinct decisions behind it.

    Only candidates rejected *by the interval* are eligible. One rejected for
    having no supporting evidence is not a pattern under any threshold — there is
    nothing to lower.
    """
    eligible = [
        candidate
        for candidate in candidates
        if candidate.rejected_because is not None and len(candidate.supporting) > 0
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda candidate: len(candidate.supporting))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--undo", action="store_true")
    args = parser.parse_args()

    container = Container.build(get_settings())
    sessions = container.database.session_factory
    try:
        if args.undo:
            async with sessions.begin() as session:
                found = list(
                    (
                        await session.execute(
                            select(models.Pattern).where(
                                models.Pattern.detector.like("%_calibration")
                            )
                        )
                    ).scalars()
                )
                for row in found:
                    await session.execute(
                        delete(models.Pattern).where(models.Pattern.id == row.id)
                    )
            print(f"removed {len(found)} pattern(s) and their reflections")
            return

        corpus = await read_corpus(sessions)
        candidate = strongest_refused(all_detectors(corpus))
        if candidate is None:
            print("nothing was refused by the interval; there is nothing to lower")
            return

        dates = {
            decision.id: decision.decided_at
            for decision in corpus.decisions
            if decision.id in candidate.supporting
        }
        created = await upsert_pattern(
            sessions,
            candidate,
            min_support=3,
            first_observed=min(dates.values()) if dates else None,
            last_observed=max(dates.values()) if dates else None,
        )

        async with sessions() as session:
            row = (
                await session.execute(
                    select(models.Pattern).where(
                        models.Pattern.detector == candidate.detector,
                        models.Pattern.subject_key == candidate.subject_key,
                    )
                )
            ).scalar_one()
    finally:
        await container.dispose()

    print(f"{'wrote' if created else 'updated'} {MARKER} pattern {row.id}")
    print(f"  {row.statement}")
    print(f"  {row.support_count} supporting, {row.contradiction_count} contradicting")
    print(f"  confidence {row.confidence:.2f}")
    print(f"\nthe gate this bypassed:\n  {candidate.rejected_because}")
    print(
        "\nNow: uv run memoryos reflect --pattern "
        f"{row.id}\nThen take it back out — this is not a finding."
    )


if __name__ == "__main__":
    asyncio.run(main())
