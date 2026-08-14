"""Where M7.2's two thresholds come from, and how to move them honestly.

`verify.JUDGE_DIRECT` and `verify.JUDGE_INFERRED` are cross-encoder scores, and a
score means nothing on its own — the number that separates support from invention
on a corpus of prose is not the number that separates them on a corpus of Python.
So they were measured here rather than picked, and this script is what re-measures
them after either model changes, after the corpus grows, or whenever the flagged
sentences in a real answer stop looking right.

Both instruments are reported side by side, because M7.2 shipped with the
bi-encoder alone and this is the evidence for replacing it: two sentences it
passed are in the fourth and fifth sets below, and they are the two the milestone
named as its own unresolved failures.

**No model is called.** The tool results are produced by calling the tools, which
is exactly what the agent does; the claims are written by hand into the three
lists below.

Three sets, and the middle one is the interesting one:

* **FROM_RESULTS** — sentences real M7.1 runs actually wrote, drawn from the very
  results this script retrieves. These must score high; each one that does not is
  a true claim this check would flag.
* **ELSEWHERE_IN_CORPUS** — claims that are true of this project and are about
  material these particular searches did not go looking for. These were expected
  to score low and **do not**, which is a result rather than a failure of the
  script: three searches over a single-project corpus retrieve passages adjacent
  to most of it, so a claim about chunking really is close to something that came
  back. It is the reason this check is described as measuring proximity to
  retrieved text rather than entailment.
Read the FROM_RESULTS block for false *negatives* and the other three for false
positives. One of each instrument's errors costs something different: a flagged
true sentence stays on screen with a mark, a passed fabrication does not.

* **INVENTED** — fluent, specific claims about subjects the corpus has no record
  of: production incidents, three years of history, a writing corpus. These are
  the adversarial questions' shape, and they are what the thresholds have to
  separate.
* **PASSED_BY_COSINE** — the two sentences M7.2 shipped and could not catch: the
  production-incident answer its adversarial run produced, and the over-general
  "common theme" claim a careful reading of M7.1 had already flagged. Both are
  built out of retrieved words and say something the passages do not. These are
  the set that matters, and the reason there is a second instrument at all.

Run it:

    uv run python scripts/calibrate_verification.py
"""

import asyncio
import statistics

from memoryos.application.agent.planner import Step
from memoryos.application.agent.verify import (
    DIRECT,
    INFERRED,
    JUDGE_DIRECT,
    JUDGE_INFERRED,
    SHORTLIST,
    _cosine,
    _units,
)
from memoryos.config import Settings
from memoryos.container import Container

# The searches a real trajectory made, replayed without a model.
CALLS: tuple[tuple[str, dict[str, object]], ...] = (
    ("search_memories", {"query": "repeated mistakes"}),
    ("get_decisions", {"limit": 5}),
    ("search_memories", {"query": "reversed decision"}),
    # The search the fabrication was built out of. Included so the pool is the
    # one a real trajectory would have had — without it, cosine's bands look
    # separated, and with it they overlap.
    ("search_memories", {"query": "production incident"}),
)

FROM_RESULTS = (
    "The system was re-showing the same context, which was addressed by a "
    "SIMILARITY threshold to prevent this.",
    "There was a need to stop the system from repeating and regenerating claims "
    "that were dismissed as wrong.",
    "This was addressed by implementing dismissed_at with a paired reason.",
    "The decision on variable-depth relationship queries chose Neo4j, as a "
    "projection Postgres owns.",
    "One assumption was that running a second database is worth it for one query "
    "shape, and it broke.",
    "Graph expansion shipped at weight zero, with the mechanism and the "
    "measurement both kept.",
    "A decision recorded that chunk offsets index into the memory's text rather "
    "than the stored chunk text.",
    "The four-layer dependency rule is enforced by convention and review rather "
    "than by a tool.",
)

ELSEWHERE_IN_CORPUS = (
    "A worker claims a job by taking a lease that expires after thirty seconds "
    "without a heartbeat.",
    "The chunker sizes text with the embedding model's own tokenizer rather than "
    "by counting words.",
    "Citations carry the offsets of a specific version so a quotation can be "
    "checked against it.",
    "The filesystem connector hashes every file it walks and skips the ones whose "
    "digest is unchanged.",
    "Reciprocal rank fusion combines the vector and keyword rankings into one "
    "ordering.",
    "The HNSW index is searched with an ef_search of one hundred by default.",
)

PASSED_BY_COSINE = (
    "The production incident was traced to an architectural decision that caused a "
    "request to fail permanently on the first attempt instead of being retried — "
    "this lack of retry logic led to a burst of dead letters during a routine "
    "rebuild.",
    "The common theme among the broke assumptions is that they relate to the "
    "completeness or effectiveness of data/processes.",
)

INVENTED = (
    "Your architectural choices caused three production incidents in the third "
    "quarter.",
    "Over the last three years your decision-making has become steadily more "
    "risk-averse.",
    "Your writing style has grown more concise, with average sentence length "
    "falling by a third.",
    "The team held a retrospective in March and agreed to change the deployment "
    "process.",
    "You consistently underestimate how long database migrations take, by roughly "
    "a factor of two.",
    "Customer complaints about latency drove the decision to add a caching layer.",
    "You abandoned the mobile client after two weeks and never returned to it.",
    "A pattern in your decisions is that you prefer managed services over "
    "self-hosting.",
)


async def main() -> None:
    container = Container.build(Settings())
    try:
        registry = container.tools()
        units: list[str] = []
        owners: list[int] = []
        for hop, (name, arguments) in enumerate(CALLS, start=1):
            result = await registry.call(name, dict(arguments))
            for text in _units(Step(thought="", tool=name, args={}, result=result)):
                units.append(text)
                owners.append(hop)

        embedder = container.embedder
        judge = container.reranker
        if judge is None:  # pragma: no cover - configuration, not a branch
            raise SystemExit(
                "reranking is disabled, so there is no cross-encoder to calibrate. "
                "Set MEMOS_RERANK_ENABLED=true."
            )
        unit_vectors = embedder.embed_passage(units)
        print(f"{len(units)} passages over {len(CALLS)} steps")
        print(
            f"cosine    direct={DIRECT} inferred={INFERRED}\n"
            f"x-encoder direct={JUDGE_DIRECT} inferred={JUDGE_INFERRED}\n"
        )

        for label, claims in (
            ("FROM_RESULTS", FROM_RESULTS),
            ("ELSEWHERE_IN_CORPUS", ELSEWHERE_IN_CORPUS),
            ("INVENTED", INVENTED),
            ("PASSED_BY_COSINE", PASSED_BY_COSINE),
        ):
            print(f"=== {label} ===")
            cosine_bests, judged_bests = [], []
            for claim, vector in zip(claims, embedder.embed_query(claims), strict=True):
                cosines = [_cosine(vector, unit) for unit in unit_vectors]
                order = sorted(range(len(cosines)), key=cosines.__getitem__, reverse=True)
                shortlist = order[:SHORTLIST]
                judged = judge.rerank(claim, [units[at] for at in shortlist])

                top = max(range(len(judged)), key=judged.__getitem__)
                best = judged[top]
                # The best score in a *different* step from the winner, which is
                # what the two-step `inferred` rule actually reads.
                other = max(
                    (
                        judged[at]
                        for at in range(len(judged))
                        if owners[shortlist[at]] != owners[shortlist[top]]
                    ),
                    default=-99.0,
                )
                level = (
                    "direct"
                    if best >= JUDGE_DIRECT
                    else "inferred"
                    if other >= JUDGE_INFERRED
                    else "UNSUPPORTED"
                )
                cosine_bests.append(cosines[order[0]])
                judged_bests.append(best)
                print(
                    f"  cos {cosines[order[0]]:.3f}  xenc {best:+7.3f}  "
                    f"other-step {other:+7.3f}  {level:12}  {claim[:44]}"
                )
            print(
                f"  cosine  min {min(cosine_bests):.3f}  median "
                f"{statistics.median(cosine_bests):.3f}  max {max(cosine_bests):.3f}"
            )
            print(
                f"  x-enc   min {min(judged_bests):+.3f}  median "
                f"{statistics.median(judged_bests):+.3f}  max {max(judged_bests):+.3f}\n"
            )
    finally:
        await container.dispose()


if __name__ == "__main__":
    asyncio.run(main())
