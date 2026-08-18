/**
 * The persistent column: what this is, where you can go, and what it holds.
 *
 * A ruled column rather than a panel. No card, no shadow, no filled pills — the
 * only things separating it from the page are a one-pixel rule and a half-step
 * of surface, because a sidebar that announces itself is a sidebar competing
 * with the content it exists to navigate to.
 *
 * **The corpus figures live down here rather than only on `/corpus`.** They are
 * the answer to "is this thing loaded and working", which is a question you ask
 * constantly while using the tool and never want to change page for. They are
 * also the honest frame for every other view: three results out of a corpus of
 * 282 memories means something different from three out of 30,000.
 */

import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { count } from "../lib/format";
import { GROUPS, HOME, inGroup, type ViewRoute } from "./routes";

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    /* `nav` rather than a translucent `sunken`. Beside the content the sidebar
       only needs a half-step of tint to sit back from the page, and an alpha
       would have done — but the same element is the mobile drawer, and a
       translucent drawer over a scrim shows the article through the nav, which
       reads as a rendering fault rather than as a layer. */
    <div className="flex h-full flex-col gap-6 overflow-y-auto bg-nav pb-4">
      <Wordmark onNavigate={onNavigate} />

      <nav className="flex flex-col gap-5" aria-label="Views">
        <Item route={HOME} onNavigate={onNavigate} />
        {GROUPS.map((group) => {
          const routes = inGroup(group.id);
          if (routes.length === 0) return null;
          return (
            <div key={group.id} className="flex flex-col gap-1">
              <p className="meta-label px-3">{group.label}</p>
              {routes.map((route) => (
                <Item key={route.path} route={route} onNavigate={onNavigate} />
              ))}
            </div>
          );
        })}
      </nav>

      {/* Pushed to the bottom, which is where a status line belongs: read when
          looked for, never in the way of the nav above it. */}
      <div className="mt-auto flex flex-col gap-3 px-3 pt-4">
        <CorpusFigures />
        <Health />
      </div>
    </div>
  );
}

function Wordmark({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <NavLink
      to="/"
      onClick={onNavigate}
      className="flex flex-col gap-0.5 px-3 pt-5"
      aria-label="Memory OS, overview"
    >
      {/* The one place the display face appears at full size. */}
      <span className="display text-xl">Memory OS</span>
      <span className="meta text-faint">a corpus that remembers why</span>
    </NavLink>
  );
}

function Item({ route, onNavigate }: { route: ViewRoute; onNavigate?: () => void }) {
  return (
    <NavLink
      to={route.path}
      end={route.path === "/"}
      onClick={onNavigate}
      title={route.blurb}
      className={({ isActive }) =>
        `nav-item ${isActive ? "nav-item-on" : ""} ${route.planned ? "nav-item-soon" : ""}`
      }
    >
      <span>{route.label}</span>
      {/* Marked, not hidden. See `routes.ts` — a nav that omits what is not
          built misrepresents the shape of the application. */}
      {route.planned ? <span className="meta ml-auto text-faint">soon</span> : null}
    </NavLink>
  );
}

/**
 * The size of the corpus, live.
 *
 * Four numbers rather than the eight `/corpus` shows: this is the glance, not
 * the report. Entities is here because a zero is the single most informative
 * number in this sidebar — it means the graph layer is empty and every
 * entity-scoped view will be too, which is otherwise discovered one confusing
 * empty page at a time.
 */
function CorpusFigures() {
  const stats = useQuery({
    queryKey: ["stats"],
    queryFn: api.stats,
    staleTime: 60_000,
  });

  if (stats.isError) {
    return <p className="meta text-deny">corpus unavailable</p>;
  }
  if (!stats.data) {
    return <p className="meta text-faint">reading corpus…</p>;
  }

  return (
    <dl className="flex flex-col gap-0.5" data-testid="sidebar-figures">
      <Row label="memories" value={count(stats.data.current_memories)} />
      <Row label="chunks" value={count(stats.data.chunks)} />
      <Row label="entities" value={count(stats.data.entities)} />
      <Row label="relations" value={count(stats.data.relationships)} />
    </dl>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="meta text-faint">{label}</dt>
      <dd className="meta text-ink">{value}</dd>
    </div>
  );
}

/**
 * Health, quietly.
 *
 * One dot and one word. The detail is on `/corpus`, and a sidebar that listed
 * every check would be a sidebar nobody reads — but a corpus whose graph is
 * unreachable while its database is fine is a specific, common, and recoverable
 * state, and the whole reason `/health/ready` distinguishes them.
 */
function Health() {
  const ready = useQuery({
    queryKey: ["ready"],
    queryFn: api.ready,
    // Slow on purpose. This is reassurance, not monitoring: something that
    // repolls every few seconds is a thing you notice, and there is nothing to
    // do about a red dot from here anyway.
    staleTime: 30_000,
    refetchInterval: 60_000,
  });

  const state = ready.isError
    ? { tone: "bg-deny", label: "api unreachable", detail: "nothing is listening" }
    : ready.data?.status === "ok"
      ? { tone: "bg-affirm", label: "healthy", detail: "database and graph reachable" }
      : ready.data
        ? {
            tone: "bg-accent-bright",
            label: "degraded",
            detail: ready.data.database
              ? "graph unreachable — search and ingest still work"
              : "database unreachable",
          }
        : { tone: "bg-rule-strong", label: "checking…", detail: "" };

  return (
    <div
      className="flex items-baseline gap-2 border-t border-rule pt-3"
      title={state.detail}
      data-testid="health"
    >
      <span className={`inline-block size-1.5 shrink-0 rounded-full ${state.tone}`} aria-hidden />
      <span className="meta text-muted">{state.label}</span>
    </div>
  );
}
