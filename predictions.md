# Predictions

Written before the measurement, so that the result can disagree with it.

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
