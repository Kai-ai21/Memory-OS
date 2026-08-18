/**
 * The persistent column: what this is, where you can go, and what it holds.
 *
 * A pane of glass at the edge of the screen, per the Luminous reference: the
 * wordmark in glowing mono caps, one primary action, six nav items with icons,
 * and settings and help pinned to the bottom rule. The active item takes a cyan
 * left rule, a faint fill and an inner glow — which is a different argument
 * from the ruled era's 2px bar. Against a dark void a hairline alone is not
 * findable in peripheral vision, and the fill is what lets you see where you
 * are without looking directly at the nav.
 *
 * **The corpus figures live down here rather than only on `/corpus`.** They are
 * the answer to "is this thing loaded and working", which is a question you ask
 * constantly while using the tool and never want to change page for. They are
 * also the honest frame for every other view: three results out of a corpus of
 * 282 memories means something different from three out of 30,000.
 */

import { NavLink, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { Icon } from "../components/Icon";
import { count } from "../lib/format";
import { GROUPS, HOME, inGroup, type ViewRoute } from "./routes";

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    /* Glass beside the content, near-opaque over it.
       On desktop the sidebar sits *next to* the page and `--color-nav` — the
       void at 40% — lets the shell's glow move behind the nav, which is what
       makes it part of the same surface as everything else. As the mobile
       drawer the same element sits *over* the article, and there translucency
       stops being a layer and starts looking like a rendering fault: you read
       the search results through the navigation. Measured at 375px before
       fixing it. So below `md` it takes the void almost solid, and the glass
       comes back at the breakpoint where the drawer becomes a column. */
    <div className="flex h-full flex-col gap-6 overflow-y-auto bg-void/95 px-3 pt-6 pb-4 backdrop-blur-xl md:bg-nav">
      <Wordmark onNavigate={onNavigate} />
      <NewConversation onNavigate={onNavigate} />

      <nav className="flex flex-col gap-5" aria-label="Views">
        <div className="flex flex-col gap-1">
          <Item route={HOME} onNavigate={onNavigate} />
          {inGroup("primary").map((route) => (
            <Item key={route.path} route={route} onNavigate={onNavigate} />
          ))}
        </div>

        {GROUPS.filter((group) => group.id !== "primary").map((group) => {
          const routes = inGroup(group.id);
          if (routes.length === 0) return null;
          return (
            <div key={group.id} className="flex flex-col gap-1">
              <p className="meta-label px-3 pb-1">{group.label}</p>
              {routes.map((route) => (
                <Item key={route.path} route={route} onNavigate={onNavigate} />
              ))}
            </div>
          );
        })}
      </nav>

      {/* Pushed to the bottom, which is where a status line belongs: read when
          looked for, never in the way of the nav above it. */}
      <div className="mt-auto flex flex-col gap-3 border-t border-rule pt-3">
        <Pinned onNavigate={onNavigate} />
        <div className="flex flex-col gap-3 border-t border-rule px-3 pt-3">
          <CorpusFigures />
          <Health />
        </div>
      </div>
    </div>
  );
}

function Wordmark({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <NavLink
      to="/"
      onClick={onNavigate}
      className="flex flex-col gap-1 px-3"
      aria-label="Memory OS, chat"
    >
      {/* Mono rather than the display face, and glowing. The reference sets the
          wordmark in the same face as the system labels, which is what makes it
          read as a machine announcing itself rather than as a brand. */}
      <span className="glow-cyan font-mono text-sm font-bold tracking-[0.18em] text-ink">
        MEMORY OS
      </span>
      <span className="meta-label">a corpus that remembers why</span>
    </NavLink>
  );
}

/**
 * The one primary action in the interface.
 *
 * Starts a *new session* rather than merely navigating to chat — which is why
 * this and the `chat` nav item both exist and are not redundant. The reference
 * has only this button, because in the reference chat is the whole canvas; here
 * there are sessions, and "go back to what I was saying" and "start something
 * new" are different intentions. Dropping the session parameter is what makes
 * it new: see `ChatPage`, which reads the session from the URL.
 */
function NewConversation({ onNavigate }: { onNavigate?: () => void }) {
  const navigate = useNavigate();

  return (
    <button
      type="button"
      className="btn-primary mx-3 flex items-center justify-center gap-2"
      onClick={() => {
        navigate("/");
        onNavigate?.();
      }}
    >
      <Icon name="add" size={16} />
      <span>New conversation</span>
    </button>
  );
}

/**
 * The pair pinned to the bottom rule, as the reference draws them.
 *
 * **Wired to what exists rather than to what the reference invented.** The
 * mockup shows Settings and Help as two more nav rows; this application has
 * neither page. Help is real and is this: the command palette lists every view
 * in the application with a sentence saying what it answers, which is the only
 * help surface here and a better one than a page of prose would be.
 *
 * There is no settings row. Everything configurable in this system is an
 * environment variable read at startup — weights, models, connection strings —
 * and the one thing you can change from the interface, which directories get
 * read, is `sources` in the nav above. A settings row would have to lead
 * somewhere, and the honest options were a page saying "there are no settings"
 * or a second link to a screen already two inches higher. Both are worse than
 * the gap.
 */
function Pinned({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        className="nav-item w-full text-left"
        onClick={() => {
          onNavigate?.();
          // The palette owns this shortcut; dispatching the key is how a button
          // and a keystroke stay one implementation rather than two.
          window.dispatchEvent(
            new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }),
          );
        }}
      >
        <Icon name="help" size={18} />
        <span>help</span>
        <span className="kbd ml-auto normal-case">⌘K</span>
      </button>
    </div>
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
        `nav-item ${isActive ? "nav-item-on" : ""}`
      }
    >
      <Icon name={route.icon} size={18} />
      <span>{route.label}</span>
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
            tone: "bg-warn",
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
