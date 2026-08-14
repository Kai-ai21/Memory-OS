"""Phase 7: the corpus as tools an agent can call.

M7.0 was scaffolding: one model turn, at most one tool call, one answer, and
what it proved was that the tools work and that the descriptions route.

M7.1 replaces the loop with a planner that takes several dependent hops, each
query shaped by what the last one returned, under three termination conditions.
`SingleCallAgent` is gone rather than kept beside it — `MultiHopPlanner` with
`max_hops=1` is the same run, and a second loop nobody calls is a second set of
prompts to keep in step with the tools.
"""
