/**
 * A route that exists with nothing behind it yet.
 *
 * **Not a blank page and not a mockup.** A blank page reads as a bug; a fake
 * screenshot of a graph that does not exist is a lie that survives right up
 * until somebody clicks it. What is useful is the truth: what this view will
 * show, what is already built underneath it, and what is missing — which is
 * also the only version of this page that is worth anything to somebody
 * evaluating the system rather than using it.
 */

import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { SectionHeading } from "../components/primitives";
import { count } from "../lib/format";
import type { ViewRoute } from "./routes";

export function Placeholder({
  route,
  built,
  missing,
  next,
}: {
  route: ViewRoute;
  /** What already exists underneath this view. */
  built: React.ReactNode;
  /** What has to be built before it can render. */
  missing: React.ReactNode;
  /** Where to go in the meantime. */
  next?: React.ReactNode;
}) {
  return (
    <div className="flex max-w-(--width-reading) flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-baseline gap-3">
          <h1 className="display-page">{route.label}</h1>
          <span className="meta-label border border-rule px-1.5 py-px">not built</span>
        </div>
        <p className="prose-lead">{route.blurb}</p>
      </header>

      <section className="flex flex-col gap-2">
        <SectionHeading>what is already there</SectionHeading>
        <div className="prose-content max-w-prose text-muted">{built}</div>
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading>what is missing</SectionHeading>
        <div className="prose-content max-w-prose text-muted">{missing}</div>
      </section>

      {next ? (
        <section className="flex flex-col gap-2">
          <SectionHeading>in the meantime</SectionHeading>
          <div className="prose-content max-w-prose text-muted">{next}</div>
        </section>
      ) : null}
    </div>
  );
}

/**
 * The graph view.
 *
 * The only genuinely unbuilt route in this application. Everything else the
 * sidebar names has a page behind it; this one has a schema, a projection and a
 * retriever, and nothing that draws them.
 */
export function GraphPlaceholder({ route }: { route: ViewRoute }) {
  // Live, like every other number in this interface. The argument this page
  // makes — that a graph view would currently draw unconnected dots — is only
  // honest while the counts say so, and the day extraction fills them in the
  // sentence has to change with them.
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, staleTime: 60_000 });
  const entities = stats.data?.entities ?? null;
  const relationships = stats.data?.relationships ?? null;

  return (
    <Placeholder
      route={route}
      built={
        <>
          The entity layer exists in Postgres and is projected into Neo4j: entities with
          their canonical names and types, mentions carrying the chunk that saw them, and
          typed relationships that each carry the span asserting them. Search already reads
          it — when the graph is what put a result in front of you, the explanation panel
          names the route it took.
        </>
      }
      missing={
        <>
          A rendering. Nothing here draws a node, and the honest reason is visible in the
          counts:{" "}
          {entities === null ? (
            <span className="text-faint">the corpus figures are not loaded</span>
          ) : (
            <span className="text-ink" data-testid="graph-counts">
              {count(entities)} {entities === 1 ? "entity" : "entities"} and{" "}
              {count(relationships ?? 0)}{" "}
              {relationships === 1 ? "relationship" : "relationships"}
            </span>
          )}
          . Extraction has reached a fraction of the corpus, so a graph drawn today would
          be a scatter of unconnected dots — which implies the layer is broken rather than
          young.
        </>
      }
      next={
        <>
          <Link className="text-amber underline" to="/corpus">
            Corpus
          </Link>{" "}
          counts what extraction has covered, and{" "}
          <code className="kbd">memoryos entities</code> lists what it found, with the
          duplicate groups that still need resolving.
        </>
      }
    />
  );
}
