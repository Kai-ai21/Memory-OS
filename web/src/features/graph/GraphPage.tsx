/**
 * The entity explorer — the reference's three panes, and the one screen in this
 * milestone that cannot be filled with live data.
 *
 * **The API exposes no entity endpoint.** Not a thin one, not a partial one:
 * there is no `/entities`, no `/entities/{id}`, no neighbours route and no merge
 * records anywhere in the OpenAPI document. The machinery exists in the
 * application layer — `entity_stats.py`, `graph_expand.py`, `graph_projection.py`
 * and `merge_admin.py` are all there — but nothing routes to it. The only entity
 * figures any HTTP client can obtain are the two counts on `/stats`, which is
 * what this page shows. This milestone is presentation only and does not add
 * endpoints, so the gap is reported rather than filled.
 *
 * **And the counts say the screen would be empty even with the endpoint.** Live:
 * 16 entities, 0 relationships, 0 merges. Every entity is mentioned in exactly
 * one memory. There is no edge to draw, no neighbour to walk to, and nothing
 * behind the reference's "merged from 3 variants" chip — that chip is invented
 * content, and the real merge table has no rows in it.
 *
 * So this page draws the reference's architecture and tells the truth inside it.
 * The canvas is real and works — see `EntityGraph`, which is tested against
 * fixtures — and it renders the moment the two conditions above stop holding.
 * Until then the panes say which endpoint is missing and which number is zero,
 * because "nothing here" and "this is broken" have to be distinguishable, and a
 * blank three-pane layout says the second.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../../api/client";
import { Failure, Loading } from "../../components/primitives";
import { count } from "../../lib/format";
import { EntityGraph, type GraphEdge, type GraphNode } from "./EntityGraph";

export function GraphPage() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, staleTime: 60_000 });

  /* There is no request to make for these. Both stay empty until an entity
     endpoint exists; the canvas is handed them rather than being given fake
     rows, so that the day the endpoint lands this page changes by one query. */
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const entities = stats.data?.entities ?? null;
  const relationships = stats.data?.relationships ?? null;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <h1 className="display-page">Graph</h1>
        <p className="prose-lead">
          What the corpus talks about, and which claims connect two things. Entities are
          extracted per chunk; a relationship is only drawn when a span asserts it.
        </p>
      </header>

      {stats.isError ? <Failure error={stats.error} /> : null}
      {stats.isLoading ? <Loading rows={2} /> : null}

      {stats.data ? (
        <>
          {/* The live figures, first, because they are the whole story of this
              screen and everything below is an explanation of them. */}
          <section className="grid gap-4 sm:grid-cols-3" data-testid="graph-figures">
            <Figure
              label="entities"
              value={count(entities ?? 0)}
              note="extracted from chunk text"
            />
            <Figure
              label="relationships"
              value={count(relationships ?? 0)}
              note={
                relationships === 0
                  ? "nothing asserts a pair yet"
                  : "each carries the span asserting it"
              }
              empty={relationships === 0}
            />
            <Figure
              label="drawable"
              value={relationships === 0 ? "none" : count(entities ?? 0)}
              note="a node needs an edge to sit on"
              empty={relationships === 0}
            />
          </section>

          {nodes.length > 0 ? (
            <section className="glass p-6">
              <EntityGraph nodes={nodes} edges={edges} />
            </section>
          ) : (
            <GraphEmpty entities={entities} relationships={relationships} />
          )}
        </>
      ) : null}
    </div>
  );
}

function Figure({
  label,
  value,
  note,
  empty,
}: {
  label: string;
  value: string;
  note: string;
  empty?: boolean;
}) {
  return (
    <div className="glass flex flex-col gap-1 p-4">
      <span className="meta-label">{label}</span>
      <span className={`figure-value ${empty ? "text-magenta" : "text-cyan"}`}>{value}</span>
      <span className="meta text-faint">{note}</span>
    </div>
  );
}

/**
 * What is missing, what would fill it, and which of the two problems you have.
 *
 * Two separate causes and they need separate sentences, because the fix is
 * different: no endpoint is a routing job, and no relationships is an
 * extraction job. A single "no data" message would leave a reader unable to
 * tell which one they are looking at — and here, unusually, it is both.
 */
function GraphEmpty({
  entities,
  relationships,
}: {
  entities: number | null;
  relationships: number | null;
}) {
  return (
    <section
      className="glass flex flex-col gap-6 border-dashed p-8"
      data-testid="graph-empty"
    >
      <div className="flex flex-col gap-2">
        <p className="meta-label text-magenta">No graph to draw</p>
        <h2 className="display text-2xl">Two things are missing, and they are separate.</h2>
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <p className="meta-label-on">the api exposes no entity route</p>
          <p className="prose-content max-w-prose text-muted">
            The entity layer is in Postgres and projected into Neo4j, and search already
            reads it — when the graph is what put a result in front of you, the
            explanation panel names the route it took. But no HTTP route returns an
            entity, its neighbours, or its merge history, so this page has nothing to
            request. Drawing it needs an endpoint that lists entities with their mention
            counts and one that returns a node's neighbours. This milestone is
            presentation only and does not add either.
          </p>
        </div>

        <div className="flex flex-col gap-1">
          <p className="meta-label-on">and there are no relationships yet</p>
          <p className="prose-content max-w-prose text-muted">
            {entities === null ? (
              "The corpus figures are not loaded."
            ) : (
              <>
                Extraction has found{" "}
                <span className="text-ink">
                  {count(entities)} {entities === 1 ? "entity" : "entities"}
                </span>{" "}
                and{" "}
                <span className="text-magenta">
                  {count(relationships ?? 0)}{" "}
                  {relationships === 1 ? "relationship" : "relationships"}
                </span>
                . A graph of {count(entities)} unconnected nodes is a scatter of dots,
                which reads as a broken view rather than a young one — so it is not drawn.
                Relationships appear when extraction finds a span that asserts a pair, and
                it has reached a fraction of the corpus so far.
              </>
            )}
          </p>
        </div>
      </div>

      <p className="meta text-faint">
        <code className="kbd">memoryos entities</code> lists what extraction found, with
        the duplicate groups still to resolve.{" "}
        <Link className="text-cyan underline" to="/corpus">
          Corpus
        </Link>{" "}
        counts what it has covered.
      </p>
    </section>
  );
}
