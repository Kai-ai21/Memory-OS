# Predictions

Written before the measurement, so that the result can disagree with it.

## M4.3 — does query-conditional recency beat the resolution floor?

**Prediction: no on the corpus-wide mean, yes on some of the six.** I expect the
mean over 52 queries to move by less than M2.3a's 0.0122 floor, and I expect the
six temporal queries to split rather than move together.

The reasoning, per mechanism, because the three do different things and only one
of them can plausibly move a mean.

**1. The month-range query cannot change anything, and that is structural.** The
whole corpus occurred between 7 and 10 August 2026. A filter for August 2026
therefore admits all 162 current memories, so the query runs against exactly the
set it ran against before. **Predicted delta: exactly zero, on every metric.**
If it moves at all, the filter is doing something other than filtering.

**2. The day-range query is the one explicit-range case that can act.** Filtering
to 8 August admits 50 of 162, and the discriminating fact is that the module
implementing the worker carries a 10 August mtime while everything else about
the job queue carries an 8 August one. So the filter removes a memory that is
topically relevant and outside the named day, which is precisely what a range
filter is for. **Predicted: precision improves, recall unchanged or slightly
worse** — the removed memory is one I judged not-relevant *because* of its date,
so this is partly a test of whether that judgement was fair.

**3. The relative query should improve most, and for an unflattering reason.**
The retrieval for it is near-random: every result comes back around −11, which is
the reranker saying nothing in the corpus answers "what was I working on". When
relevance has no opinion, recency is the only signal with one, and the corpus
does have a real answer — the last cluster of mtimes is the M2.6 answering work.
**Predicted: the largest single gain of the six.** It is also the weakest kind of
gain: a signal that helps most exactly where the retrievers have failed is not
evidence that the signal is good, only that noise is easy to beat.

**4. Ordering is the risky pair and I expect it to split.** Re-sorting the top ten
by date discards the relevance order inside it, so it wins only when the temporal
extreme and the relevant set coincide. For the "first version" query they do —
the queue migration and its module are both the oldest job-queue memories and the
right answers. For the "latest change" query they do too, but less cleanly: the
citation work is the newest cluster, and the top ten also contains the README and
the CLI, which are newer still and are not the answer. **Predicted: earliest
improves, latest is flat or slightly worse.**

**5. The trap must produce a delta of exactly zero.** "What may cause a chunk to
be discarded" contains a month name used as a modal verb. The parser requires a
temporal preposition in front of a month, so this parses to `None`, and intent
`None` takes the M3.5 code path unchanged. **Predicted: zero delta, and any
non-zero delta here is a defect rather than a result.**

**6. The other 46 queries must be identical.** Not approximately — identically,
on every metric, for every query. `_narrow` is not called and the weights object
is not replaced when intent is `None`, so the only way this fails is if the
parser fires on a query nobody intended as temporal. That is the single most
important number in this milestone's report, and it is a pass/fail rather than a
delta.

**What would falsify the "it does not pay for itself" reading:** a corpus-wide
mean gain above 0.0122 on any metric. Given that 6 of 52 queries can change at
all, a query would have to move by roughly 0.10 on average to drag the mean that
far, and only the relative query looks capable of it.

**What I expect the honest headline to be:** three mechanisms, of which one is
inert on this corpus by arithmetic, one works and is measurable, and one is a
coin flip — with the corpus-wide mean flat, because 46 of 52 queries are
untouched by construction.

## M3.5 — will graph expansion beat the resolution floor?

**Prediction: no, not on this corpus.** Mean nDCG@10 and MRR move by less than
0.0122, and the tuned graph weight lands at or very near zero.

The reasoning, in the order it matters:

1. **Coverage.** Extraction has reached 34 of 162 current memories, so 79% of
   the corpus has no entity mentions at all and is unreachable by expansion.
   A ranking that can only introduce candidates from a fifth of the corpus
   cannot move a mean over 41 queries very far, whatever its precision on the
   fifth it can see.
2. **The corpus is one voice about one subject.** Every file is prose about this
   system, written by one person in one month. Structural relatedness and
   semantic relatedness are therefore almost the same relation here — the
   documents that share entities are the documents that share vocabulary, which
   is exactly what the vector and keyword legs already find. Graph expansion
   earns its keep when structure and semantics *diverge*: a meeting, a commit
   and an invoice that share only a person. This corpus has no such spread.
3. **Hub dominance.** `sqlalchemy` (32 mentions), `postgres` (24) and `alembic`
   (14) are mentioned across a large share of the 34 extracted memories. After
   hub suppression removes them, what remains is a thin graph: 287 entities,
   662 mentions, 30 relationship rows collapsing to 24 distinct edges. Depth-2
   traversal over 24 typed edges reaches very little that MENTIONS did not
   already reach at depth 1.
4. **RRF is conservative by design.** At weight 0.5 a graph-only candidate
   contributes 0.5/61 = 0.0082 against a retriever's 0.0164, so a memory neither
   retriever found has to be ranked first by the graph to reach the middle of
   the fused list. That is the correct default — it is what stops expansion
   manufacturing answers — and it also bounds how much good it can do.

**Where I expect it to win anyway:** the five queries written for it in B3, and
specifically the two that name a person or a project rather than describing a
topic. Those are the queries where the answer is connected rather than similar.
I expect 2 or 3 of the 5 to improve, and the corpus-wide mean to stay flat —
which would be the honest result to report: a mechanism that works on the
queries it was designed for and does not pay for itself across the benchmark.

**What would falsify the "it does not help" reading:** a per-metric gain above
0.0122 on the full 41-query set, or a tuned weight that converges somewhere
clearly above zero with nDCG rising monotonically towards it.

## M3.4 — will a shadow swap be possible on Neo4j Community?

**Prediction: no.** M3.0 already found that Community supports exactly one user
database, and a swap needs somewhere to build the replacement. I expect
`CREATE DATABASE` to be refused outright by the edition rather than by
permissions, and the honest outcome to be documented downtime for the duration
of `graph rebuild` — which on a projection this size should be seconds.

## B3 — the five queries written for the graph

Written before they were run, and before the graph was rebuilt over the completed
extraction. Each names something that connects several documents without quoting
any of them, so the M2.1 contamination rule holds by construction.

The queries are described rather than quoted, because
`tests/unit/test_golden_hygiene.py` fails the build when a golden query appears
verbatim in a tracked file — a file holding the string becomes a lexical match for
it. The five are recorded in `var/golden-set.json`, which is where they belong.

| # | Query, described | Why the graph should win it | Prediction |
| - | ---------------- | --------------------------- | ---------- |
| 1 | which files the job queue design connects | The answer set spans a module, a migration, a model and a worker. Only some of them use the obvious words; the rest are connected by naming the same things. | graph wins |
| 2 | what was worked on around the same time as another body of work | Nothing in the corpus phrases itself that way. The answer is whatever shares entities with it — a structural question wearing a temporal phrasing. | graph wins |
| 3 | which parts of the system one component reaches into | The verb has no lexical footprint. The answer is the modules connected to it through the types and tables it names. | graph wins |
| 4 | what connects two named subsystems | The question is about the path between them, which is the one query shape a variable-depth traversal exists for. | graph wins, weakly: the corpus may contain no such path |
| 5 | which modules share the work of one design decision | Names a decision and asks for its neighbourhood. | uncertain: its central term is a rare literal, so the keyword leg may already find most of it |

**Overall prediction for these five: 3 of 5 improve, 1 flat, 1 unchanged or worse.**
The corpus-wide mean stays inside the resolution floor, for the reasons in the
section above. If the graph wins 4 or 5 of these *and* the corpus-wide mean moves
by more than 0.0122, my reasoning about this corpus was wrong and the graph earns
its keep outright.

**What I expect the failure mode to look like if it goes badly:** the expansion
returns plausible-looking neighbours for every query, including the 41 it was not
designed for, and drags a relevant result out of the top ten on one or two of them
— a small negative delta rather than a wash.
