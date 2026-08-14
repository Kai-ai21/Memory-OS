"""Where M7.2's two thresholds come from, and how to move them honestly.

`verify.DIRECT` and `verify.INFERRED` are cosine similarities, and a cosine
similarity means nothing on its own — 0.62 is high for one embedder and low for
another, and the number that separates support from invention on a corpus of
prose is not the number that separates them on a corpus of Python. So it was
measured here rather than picked, and this script is what re-measures it after
the embedding model changes, after the corpus grows, or whenever the flagged
sentences in a real answer stop looking right.

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
* **INVENTED** — fluent, specific claims about subjects the corpus has no record
  of: production incidents, three years of history, a writing corpus. These are
  the adversarial questions' shape, and they are what the thresholds have to
  separate.

Run it:

    uv run python scripts/calibrate_verification.py
"""

import asyncio
import statistics

from memoryos.application.agent.planner import Step
from memoryos.application.agent.verify import DIRECT, INFERRED, _cosine, _units
from memoryos.config import Settings
from memoryos.container import Container

# The searches a real trajectory made, replayed without a model.
CALLS: tuple[tuple[str, dict[str, object]], ...] = (
    ("search_memories", {"query": "repeated mistakes"}),
    ("get_decisions", {"limit": 5}),
    ("search_memories", {"query": "reversed decision"}),
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
        unit_vectors = embedder.embed_passage(units)
        print(f"{len(units)} passages over {len(CALLS)} steps")
        print(f"thresholds in use: direct={DIRECT}  inferred={INFERRED}\n")

        for label, claims in (
            ("FROM_RESULTS", FROM_RESULTS),
            ("ELSEWHERE_IN_CORPUS", ELSEWHERE_IN_CORPUS),
            ("INVENTED", INVENTED),
        ):
            print(f"=== {label} ===")
            bests = []
            for claim, vector in zip(claims, embedder.embed_query(claims), strict=True):
                scores = [_cosine(vector, unit) for unit in unit_vectors]
                order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
                best = scores[order[0]]
                # The best score in a *different* step from the winner, which is
                # what the two-step `inferred` rule actually reads.
                other = max(
                    (scores[at] for at in order if owners[at] != owners[order[0]]),
                    default=0.0,
                )
                level = (
                    "direct"
                    if best >= DIRECT
                    else "inferred"
                    if other >= INFERRED
                    else "UNSUPPORTED"
                )
                bests.append(best)
                print(f"  {best:.3f}  other-step {other:.3f}  {level:12}  {claim[:56]}")
            print(
                f"  min {min(bests):.3f}   median {statistics.median(bests):.3f}   "
                f"max {max(bests):.3f}\n"
            )
    finally:
        await container.dispose()


if __name__ == "__main__":
    asyncio.run(main())
