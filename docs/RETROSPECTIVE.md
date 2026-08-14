# Full-project retrospective

Eight phases, twenty-nine milestones, one corpus: this repository ingesting
itself. 253 memories, 3,157 chunks, 16 recorded decisions, 37 assumptions.

Every phase wrote its own retrospective as it closed and those are in the
[README](../README.md), beside the code they judge. This one asks the questions
that only make sense at the end: which of the architectural bets paid, which
bugs were found by measurement rather than by tests, where the layering was real
and where it was a diagram, and which phases were worth their time.

The Phase 1 retrospective set the standard for this document when it called the
repository ports **unearned** and the layering **about one-third real**, in
Phase 1, while there was still time for that to be embarrassing. Eight phases
have not changed the verdict, and the sections below say so.

---

## 1. Which architectural decisions earned their cost

### Earned, clearly

**Bitemporal storage (M1.1).** `occurred_at` beside `ingested_at`, with
`occurred_at_source` recording how each was derived — added six milestones
before anything read them. Three of Phase 4's four milestones were then pure
query work. Adding the column in M4.0 would have been an afternoon; **recovering
the values it should have held would have been impossible**, because a file that
has been rewritten reports today's mtime for work done last March. This is the
one decision in the project where the cost was paid early and the alternative
was not "pay later" but "never".

It also produced the project's most-cited number. `occurred_at_source` is why
the corpus can say **0 of 253 memories carry a source-declared date** instead of
quietly rendering a timeline of `git checkout` as a timeline of work. Three later
capabilities decline for that reason and cite that column when they do.

**Content addressing and the event log (M1.1–M1.3).** Everything derived is
rebuildable and the rebuild is *proved* identical, twice — into a shadow schema
and in place. Replay is not a feature anyone asked for and it has caught real
defects every phase since, because a table that cannot be rebuilt from the log is
a table with undeclared state in it. The `USER_AUTHORED` set in `replay.py` — the
tables a rebuild must never truncate — became the project's clearest statement of
what is a person's judgement versus a computation, and Phases 5 and 8 both
extended it without argument because the principle was already written down.

**Refusing below an evidence bar, everywhere.** Patterns need three decisions,
gaps need two instances, facets need three distinct observations, reflections
refuse to describe a pattern that does not clear its own threshold, surfacing has
a per-focus adaptive threshold. This is the single design idea that recurs most
and it earned its cost every time. The reason is not modesty: **the output of
every one of these features is a fluent sentence about a person, and nothing in
such a sentence reveals whether three observations or zero produced it.** Putting
the bar in code rather than in the reader's judgement is what makes the zero
outputs in this project readable as results.

**Measuring before shipping a ranking signal.** Recency and importance were
grid-searched in M2.3b, measured as harmful, and **shipped at weight 0.0**. Graph
expansion was measured in M3.5 as contributing nothing and shipped at weight
zero. Both remain in the code with their weights at zero and their measurements
in the README. A project that shipped them at plausible-looking weights would
have been indistinguishable from this one on every test.

### Did not earn its cost

**The repository ports.** One implementation each, no second one ever written, no
test that substitutes a fake for a real repository in a way that caught anything.
They exist to satisfy a layering diagram. Count the violations rather than
asserting them: **43 of the 54 modules in `application/` import
`memoryos.adapters.db.models` directly** and write SQLAlchemy against it, because
the port would have had to grow a method per query and nobody was ever going to
write that. **The honest description is that this project has a three-layer
architecture with a `ports.py` in it.**

**Six agent tools with comparative descriptions.** 1,400–2,400 prompt tokens
before the question is asked, a third of every hop's bill, to route between tools
of which the graph tool has never returned anything useful. Phase 7's own
retrospective calls for three.

**The `evolution` and `tuning` modules relative to their use.** Both are real
work; both were exercised once, produced a number, and have not been read since.
Not wrong, but the shape of a project building capability ahead of demand.

### Earned, but not for the reason expected

**Neo4j.** The graph projection has never improved a retrieval score — `graph
reached 0 queries and introduced 0 results` on the current corpus, because entity
extraction has covered 5% of it. What it *did* earn is `graph verify`: divergence
detection per node and edge type **by hash rather than by count**, which catches a
corrupted node name that no count would see. The projection's value in this
project turned out to be as a second system that must agree with the first, and
therefore as a test of whether Postgres is really the system of record. That is
worth something, and it is not what the graph was built for.

---

## 2. Every bug found by measurement rather than by testing

This is the list the project exists to be able to write.

### M1.6.1 — 89% of chunks silently truncated

Chunk sizes were derived from a guess at the embedding model's window rather than
from the model's real one. **Every test passed.** The corpus embedded, search
returned results, recall looked plausible. Found by measuring recall against a
model that reported its own `max_sequence_tokens`.

**What it taught:** the whole rest of the project. The startup assertion,
`doctor`, `eval-recall`, `verify-replay`, and every baseline in `var/` exist
because of this one defect. A test asserts that code does what you thought; a
measurement asks what actually happened.

### M2.3a — the run-to-run variance floor

Not a bug, but the measurement that made every later bug visible. Repeated
identical evaluation runs move by **~0.012**, from floating-point ordering alone.

**What it taught:** every claim of improvement smaller than the noise floor is
unfalsifiable. Phase 7 then built a benchmark it could not afford to run three
times, and therefore has no floor and no way to tell an improvement from noise —
the same lesson, not learned, three phases later.

### M2.3b — recency and importance made retrieval worse

Two ranking signals everybody expects to help, grid-searched, measured as
harmful.

**What it taught:** intuition about ranking is worthless without a golden set.
Both signals would have shipped at a plausible 0.2 in any project that did not
measure.

### M6.3 → M7.0 — a 54% dismissal rate

Proactive surfacing shipped and **more than half of what it volunteered was
refused** — the wrong side of the line it set for itself, visible only because
the surfacing log records every decision and its outcome rather than only the
surfaced ones. M7.0 found the cause and fixed it.

**What it taught:** log the decisions you *didn't* act on. A system that only
records what it did cannot measure its own restraint. It also taught the reverse
of the evidence-bar lesson: this is the one feature whose threshold was set after
seeing output rather than before, and it is the one that shipped wrong.

### M7.1 — the hop limit never fired

Multi-hop planning was built with three termination conditions and a hop ceiling.
Measured over five questions needing four or five hops, the mean was **1.4** and
the ceiling **never fired once**. The loop stops too early, not too late — the
opposite of the failure the ceiling was built to prevent.

**What it taught:** a guardrail that never fires is not evidence of safety. It is
evidence you measured the wrong risk.

### M7.2 — a fluent fabrication scored 100% supported

Answer verification measures each claim against what the trajectory retrieved,
and correctly withheld one answer. It then passed a **fabrication about
production incidents** at 100% support.

**What it taught:** the instrument measures *proximity to retrieved text*, not
entailment. Those are different properties and the gap between them is exactly
where a plausible sentence lives. M7.3 tried a cross-encoder as a replacement and
that failed too, for a documented reason. Honest scope reduction — check citation
integrity, which is exact, and stop claiming to check support — is still the
right answer and was not taken.

### M7.3 — a metric that could not be measured was printed as 0.000

`tool_appropriateness` is undefined for a trajectory the provider did not
narrate. On all eight golden questions, `judgeable` is **0**, so the metric reads
0.000 across the board — which looks exactly like an agent that picks the wrong
tool every time. `overall` correctly excludes it; the table did not say so.
**Found while writing M8.2's report**, by reading the baseline rather than the
summary. Fixed in the report renderer, which now annotates it.

**What it taught:** "not measured" and "measured as zero" must not render the
same. This is the same defect as printing a facet at 0.2 confidence instead of
declining — the project had already written the rule and then broke it in a
neighbouring module.

### M8.1 — the number that explains three detectors

Not a bug, a diagnosis found by measuring silence. Gap analysis produced zero
gaps four runs out of four, and the reason turned out to be one number: **35 of
37 assumptions are ungrouped.** No belief is shared by two decisions, so there is
no belief a third can be missing.

**What it taught:** when a feature produces nothing, measure *why* rather than
lowering the bar. The cheapest thing that would make three of Phase 8's detectors
fire is an hour of grouping assumptions by hand — not more milestones.

### M8.2 — retrieval decayed 0.13 recall@10 in two days

The finding of this milestone, and the reason it is last. The same 52 golden
queries against the same answer key scored **recall@10 0.643** today against
**0.773** for the baseline recorded two days earlier. Nothing in the retriever
changed.

The mechanism is visible in the worst query. For *"why do we store two
timestamps"* the top four results are `tests/slow/test_acceptance.py`,
`tests/slow/test_replay_real.py`, `tests/slow/test_query_prefix.py` and a
migration — the first three because they contain the query string as a **literal**,
being the tests that assert on it. `src/memoryos/domain/entities.py`, which
answers the question, is at rank 9.

**What it taught:** *ingesting the repository into itself made the evaluation
harness part of the corpus it evaluates.* `eval_exclude` removes those files from
scoring and the evaluator widens `k` to compensate, so the metric is not
double-counting — but the ranking is still led by text *about* the query rather
than answers to it, and exclusion cannot fix that. It is M1.6.1's class of defect
found M1.6.1's way, eight phases later: not by a test, by a number moving.

---

## 3. Where the layering held, and where it was aspirational

**Held: the domain layer.** `domain/` is genuinely pure — arithmetic, thresholds,
value objects, no I/O — and it is where every evidence bar lives. This is why the
bars are testable without a database and why `facet_confidence` could be imported
from `pattern_confidence` rather than reinvented. The single best structural
decision after bitemporality, and the cheapest.

**Held: adapters behind real seams.** The embedder, the reranker, the graph store
and the LLM provider all have working alternatives or fakes, and the fakes are
used. `FakeEmbedder` makes the whole integration suite fast. These seams earned
their cost because a second implementation actually exists on the other side.

**Aspirational: the repository ports.** Covered above. One implementation, no
substitution, and the application layer routes around them.

**Aspirational: application/adapters separation for the database.** The rule says
the application layer talks to ports. In practice 43 of 54 modules import
`models` and write SQLAlchemy. This was the pragmatic call every time and it was
probably right every time; what is dishonest is the diagram that says otherwise.
**A layering rule violated in four files out of five is not a layering rule.**

**Real but under-used: the container.** `Container.build(settings)` genuinely
centralises construction and made `report --full` easy to write — one object,
every dependency. It would have been worth more if the ports it wires had more
than one implementation each.

**The verdict, updated.** Phase 1 said the layering was about one-third real.
After eight phases it is closer to **half** — the domain layer grew and stayed
pure, which is the half that held — and the other half is a diagram the code does
not follow.

---

## 4. Which phases were worth their time

| Phase | Verdict |
| --- | --- |
| **1 — Foundation** | **Worth it, twice over.** Bitemporality and replay could not have been retrofitted. M1.6.1 alone justified the measurement discipline. |
| **2 — Retrieval** | **Worth it.** The golden set and the variance floor are the instruments every later phase is judged on. The two negative results are as valuable as the positive ones. |
| **3 — Graph** | **Half worth it.** `graph verify` earned its place; graph-augmented retrieval contributed zero and always would have on a corpus with 5% extraction coverage. **Would cut M3.5 and keep M3.4.** |
| **4 — Time** | **Worth it.** Query work over a schema Phase 1 got right, plus the provenance measurement that three later phases cite. |
| **5 — Decisions** | **Worth it as apparatus, not as findings.** Zero patterns, zero reflections, and each printed why. The schema is the part that transfers. |
| **6 — Proactivity** | **Worth it for one number.** The 54% dismissal rate is the most useful thing Phase 6 produced, and it was a failure. Would keep the surfacing log and cut the VS Code extension. |
| **7 — Agent** | **The phase I would cut most of.** Four milestones to establish that multi-hop earns its cost on two of eight questions, at five times the token price, with a variance floor it could not afford to measure. M7.0 (tools) and M7.2 (verification) are worth keeping; M7.1 and M7.3 measured an agent this corpus cannot support. |
| **8 — The model of you** | **Worth it as a structure, honest as a result.** Zero facets, zero gaps, no stability verdict — three milestones whose deliverable is a measurement of their own emptiness. That is the correct outcome and it is also an argument for having built the corpus first. |

**If the project had to be done in half the time**: Phases 1, 2 and 5, plus
`graph verify` from Phase 3 and the surfacing log from Phase 6. That is the
apparatus. Everything else was capability built ahead of the data to exercise it.

---

## 5. What I would tell someone starting this project

**1. Get the corpus before you build the machinery that reads it.**
Six of the ten headline capabilities in this repository are waiting on data
rather than on code. One connector that declares its own dates would unlock
`habits`, the timeline, and half of the temporal retrieval work. One afternoon of
grouping assumptions by hand would unlock pattern discovery, three of the four
gap detectors, and two of the five user-model derivers. Neither is a milestone;
both are worth more than any milestone in Phase 8.

**2. Do not ingest the repository into itself.**
It is the fastest way to get a corpus and it costs the evaluation its meaning.
The tests contain the queries, the README contains the answers, and the golden
set describes a corpus that describes the golden set. Use a corpus you did not
write.

**3. Write the evidence bar before you see the output.**
Every feature here that refuses well — patterns, gaps, facets, reflections —
refuses well because the threshold existed before anyone saw what it would emit.
The one that went the other way shipped at a 54% dismissal rate. Once you have
seen the output, every threshold you choose is a threshold chosen to produce it.

**4. A number you cannot afford to reproduce is not a measurement.**
Establish the variance floor first, and if you cannot afford three runs of a
benchmark, you cannot afford the benchmark. Phase 2 learned this and Phase 7
ignored it in the same repository.

**5. "Not measured" and "measured as zero" must never render the same.**
This is the rule the whole project is organised around, and it was still being
broken in Phase 7 while Phase 8 enforced it two modules away. A dimension with no
evidence prints what it would take to fill it. A metric that could not be judged
says so. A facet below the bar is not written at a low confidence — it is not
written.

**6. Delete nothing that represents a belief.**
Superseded facets, dismissed patterns, rejected suggestions, withdrawn claims:
all kept, all readable, all carrying the reason. A clean current state is worth
less than an honest record of a state that changed, and the difference only
becomes visible at the moment somebody asks "did you used to think something
else?" — by which point the delete has already happened.

**7. Log the decisions you did not act on.**
The surfacing log records every suppression and its reason, and that is the only
reason the 54% dismissal rate was findable. A system that records only its
actions cannot measure its own restraint, and restraint is most of what a
proactive system does.

**8. Expect the honest answer to be "not yet".**
Four milestones in Phase 5 and three in Phase 8 have a headline number of zero.
Every one of them printed why, with the number it reached and what would change
it. That is not a failure mode; it is the only thing that distinguishes a system
with nothing to say from one that is broken, and it is harder to build than the
feature it guards.
