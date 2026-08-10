"""Re-record an exported golden set into an empty `query_judgements` table.

`query_judgements` is the one table nothing can regenerate, and the project's own
verification recipe opens with `docker compose down -v`. Those two facts are in
direct conflict: every clean-slate check destroys the only copy of work that
took a person an afternoon. The export exists precisely because judgements are
keyed on `(source_name, external_key, chunk_ordinal)` and therefore survive a
rebuild — this puts them back.

It writes through `application.judgements.record`, the same function the API
endpoint and the UI's buttons call, so a restored row is indistinguishable from
one that was clicked. It is not a way to *author* judgements: everything it
writes was judged by a person and exported, and the snapshots it carries
(`rank_at_judgement`, `score_at_judgement`) are the ones recorded at that
moment, never refreshed.

    uv run python scripts/restore_judgements.py var/golden-set.json
"""

import asyncio
import json
import sys
from pathlib import Path

from memoryos.application.judgements import JudgementInput, record
from memoryos.config import get_settings
from memoryos.container import Container
from memoryos.domain.values import Verdict


async def restore(path: Path) -> int:
    payload = json.loads(path.read_text())
    container = Container.build(get_settings())
    written = 0
    try:
        for query in payload["queries"]:
            for item in query["items"]:
                await record(
                    container.database.session_factory,
                    JudgementInput(
                        query_text=query["query_text"],
                        source_name=item["source_name"],
                        external_key=item["external_key"],
                        chunk_ordinal=item.get("chunk_ordinal"),
                        verdict=Verdict(item["verdict"]),
                        # Deliberately not restored: `memory_id` and `chunk_id`
                        # are snapshots of a corpus that no longer exists after a
                        # rebuild, and writing stale ids back would be worse than
                        # leaving them null. The export re-resolves them anyway.
                        rank_at_judgement=item.get("rank_at_judgement"),
                        score_at_judgement=item.get("score_at_judgement"),
                        filters=query.get("filters") or {},
                    ),
                )
                written += 1
    finally:
        await container.dispose()
    return written


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no export at {path}")
        return 1
    written = asyncio.run(restore(path))
    print(f"restored {written} judgements from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
