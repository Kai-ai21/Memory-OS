/**
 * The panel: who you are, where you can go, and one thing worth reading.
 *
 * **It is a floating panel now, not a column.** Until M9.8 this was a flat
 * white column flush to the left edge, separated from the page by a hairline.
 * It is now 264px of glass inset 12px from the top, left and bottom of the
 * viewport, with a 20px radius — and the 12px of nothing around it is the point
 * of the change rather than a detail of it. The two drifting radials and the
 * cursor trail run underneath, and being able to see them past the edges and
 * through the face is the only thing that makes the material read as glass
 * instead of as a pale grey rectangle. See `.glass-panel`.
 *
 * **Every row is a glyph and a word, and nothing else.** No pill, no border, no
 * fill at rest. The group headings are gone — the groups are 20px of space now
 * — and so is the wordmark, which has moved to being what it always was: the
 * landing page's job. What is left at the top is an avatar and two controls.
 *
 * **The active item still carries no accent.** That was true before this
 * milestone and is the one thing in here that did not change: it is a tint and
 * a step up in weight, because where you already are is not something you can
 * click. See rule 1 in `tokens.css`, and the test that pins it.
 *
 * **The corpus figures moved behind `details` rather than out.** They are the
 * answer to "is this thing loaded and working", which is worth one click and is
 * not worth six permanent rows at the bottom of a panel this quiet — and the
 * disclosure remembers, so anybody who wants them up keeps them up. They are
 * also still the honest frame for every other view: three results out of a
 * corpus of 282 memories means something different from three out of 30,000.
 */

import { useEffect, useRef, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { Icon } from "../components/Icon";
import { count } from "../lib/format";
import { GROUPS, HOME, inGroup, type ViewRoute } from "./routes";

/**
 * Whether the details block is open, across reloads.
 *
 * Wrapped in `try` for the same reason the particle preference is: Safari's
 * private mode throws on `localStorage` rather than returning null, and a
 * disclosure state is not worth taking the application down over. A read that
 * fails is a closed block, which is the default anyway.
 */
const DETAILS_KEY = "memo:sidebar-details";

function readDetails(): boolean {
  try {
    return window.localStorage.getItem(DETAILS_KEY) === "open";
  } catch {
    return false;
  }
}

function writeDetails(open: boolean): void {
  try {
    window.localStorage.setItem(DETAILS_KEY, open ? "open" : "closed");
  } catch {
    // Still a preference for this session; the state lives in React either way.
  }
}

export function Sidebar({
  onNavigate,
  onCollapse,
}: {
  onNavigate?: () => void;
  /** Hide the panel. The shell owns the state; this is the handle for it. */
  onCollapse?: () => void;
}) {
  return (
    <div className="glass-panel flex h-full flex-col gap-4 overflow-hidden px-2 py-3">
      <Header onCollapse={onCollapse} />

      {/* The only scrolling region. The card and the footer are pinned, so the
          promoted card cannot be scrolled out of the panel on a short window —
          which is the entire reason it is promoted. */}
      <nav
        className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto"
        aria-label="Views"
      >
        <div className="flex flex-col gap-0.5">
          <NewConversation onNavigate={onNavigate} />
          <Item route={HOME} onNavigate={onNavigate} />
          {inGroup("primary").map((route) => (
            <Item key={route.path} route={route} onNavigate={onNavigate} />
          ))}
        </div>

        {/* 20px of nothing between the groups, and nothing else. A heading here
            would be a row you cannot click, at the top of a group whose members
            already say what they are. */}
        {GROUPS.filter((group) => group !== "primary").map((group) => {
          const routes = inGroup(group);
          if (routes.length === 0) return null;
          return (
            <div key={group} className="flex flex-col gap-0.5">
              {routes.map((route) => (
                <Item key={route.path} route={route} onNavigate={onNavigate} />
              ))}
            </div>
          );
        })}
      </nav>

      <Promoted onNavigate={onNavigate} />
      <Footer onNavigate={onNavigate} />
    </div>
  );
}

/**
 * The top row: an avatar, search, and the collapse toggle.
 *
 * **The wordmark is gone and this is the right call.** MEMO was a 22px display
 * face and a tagline sitting above the nav on every screen of the application,
 * which is a product name introducing itself to somebody who is already inside
 * the product. The name belongs on the landing page, where there is somebody
 * who does not know it yet.
 *
 * Search here opens the command palette rather than routing to `/search`. There
 * is already a `search` row four lines below that goes to the view; a second
 * control doing the identical thing would be a wasted affordance, and the
 * palette is the thing this icon means everywhere else — jump to anything.
 */
function Header({ onCollapse }: { onCollapse?: () => void }) {
  return (
    <div className="flex items-center justify-between px-1">
      <span
        className="avatar size-8 shrink-0 rounded-full"
        aria-hidden
        data-testid="avatar"
      />

      <div className="flex items-center">
        <button
          type="button"
          className="icon-button"
          aria-label="Search everything"
          title="Search everything — ⌘K"
          onClick={() => openPalette()}
        >
          <Icon name="search" size={20} />
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="Hide navigation"
          title="Hide navigation"
          onClick={onCollapse}
        >
          <Icon name="collapse" size={20} />
        </button>
      </div>
    </div>
  );
}

/** The palette owns ⌘K; dispatching the key is how a button and a keystroke
 *  stay one implementation rather than two. */
function openPalette(): void {
  window.dispatchEvent(
    new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }),
  );
}

/**
 * Starting something new, as a row rather than as a button.
 *
 * It was the one filled, uppercase, glass button in the interface and it sat
 * above the nav shouting at it. **A panel with one loud object in it is a panel
 * you read in the order the loudness dictates**, which was: button, wordmark,
 * everything else. As a row with a `+` in front of it, it is the first thing in
 * the list because it is first in the list, which is enough — the reference
 * does exactly this and it is why the reference reads calm.
 *
 * It still starts a *new session* rather than merely navigating to chat, which
 * is why this and the `chat` row are both here and are not redundant. Dropping
 * the session parameter is what makes it new: see `ChatPage`, which reads the
 * session from the URL.
 */
function NewConversation({ onNavigate }: { onNavigate?: () => void }) {
  const navigate = useNavigate();

  return (
    <button
      type="button"
      className="nav-item w-full text-left"
      onClick={() => {
        navigate("/");
        onNavigate?.();
      }}
    >
      <Icon name="add" size={20} />
      <span>new chat</span>
    </button>
  );
}

function Item({ route, onNavigate }: { route: ViewRoute; onNavigate?: () => void }) {
  return (
    <NavLink
      to={route.path}
      end={route.path === "/"}
      onClick={onNavigate}
      title={route.blurb}
      className={({ isActive }) => `nav-item ${isActive ? "nav-item-on" : ""}`}
    >
      <Icon name={route.icon} size={20} />
      <span>{route.label}</span>
    </NavLink>
  );
}

/**
 * The one surfaced thing in the panel.
 *
 * White paper with a hairline, in a panel where everything else is text on
 * glass — which is why the eye goes here and why there must never be a second
 * one. It is spent on the tour rather than on a figure or a status: somebody
 * who does not yet know what this application *is* cannot be helped by a count,
 * and `/overview` is the page that answers the question in numbers that come
 * from their own corpus.
 */
function Promoted({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <NavLink to="/overview" onClick={onNavigate} className="promo shrink-0">
      <span className="mt-px text-ink-2">
        <Icon name="help" size={20} />
      </span>
      <span className="flex flex-col gap-0.5">
        <span className="text-body-sm text-ink">What MEMO does</span>
        <span className="meta text-ink-3">A short tour of how memory works</span>
      </span>
    </NavLink>
  );
}

/**
 * Beneath the card, under a hairline: the corpus, and everything meta.
 *
 * **Two permanent rows became one three-dot button.** `help` and `sign out`
 * were pinned to the bottom of every screen — one of them a keyboard shortcut
 * that is already bound globally, the other a thing you do once a month. Under
 * `more` they cost a click and no space, which is the correct trade for both.
 * There is still no settings page in this application and this milestone does
 * not invent one; `more` holds what actually exists.
 */
function Footer({ onNavigate }: { onNavigate?: () => void }) {
  const [open, setOpen] = useState(readDetails);

  return (
    <div className="shrink-0 border-t border-rule pt-2">
      <button
        type="button"
        className="nav-item w-full text-left"
        aria-expanded={open}
        aria-controls="sidebar-details"
        data-testid="details-toggle"
        onClick={() => {
          setOpen((current) => {
            writeDetails(!current);
            return !current;
          });
        }}
      >
        <Icon name="tune" size={20} />
        <span>details</span>
        <span
          className={`ml-auto transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          <Icon name="chevron" size={16} />
        </span>
      </button>

      {open ? (
        <div id="sidebar-details" className="flex flex-col gap-3 px-3 py-2">
          <CorpusFigures />
          <Health />
        </div>
      ) : null}

      <More onNavigate={onNavigate} />
    </div>
  );
}

/**
 * The meta menu.
 *
 * Closes on Escape, on a click anywhere outside it, and on any of its own
 * items — three exits, because a menu you cannot dismiss by clicking away from
 * it is the single most common complaint about menus. `Escape` is handled here
 * rather than in the shell's global handler so that it closes this first and
 * the drawer second, which is the order a person expects when both are open.
 */
function More({ onNavigate }: { onNavigate?: () => void }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function onPointer(event: MouseEvent) {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.stopPropagation();
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="relative" ref={box}>
      <button
        type="button"
        className="nav-item w-full text-left"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen((current) => !current)}
      >
        <Icon name="more" size={20} />
        <span>more</span>
      </button>

      {open ? (
        <div
          role="menu"
          /* Opaque, unlike everything else in this panel — see `.menu`. It
             covers two rows of the footer, and glass over text is text you can
             read through the thing covering it. `bottom-full` because there is
             nothing below it: this is the last thing in a panel that ends 12px
             from the bottom of the window. */
          className="menu absolute bottom-full left-0 z-10 mb-1 flex w-full flex-col gap-0.5 p-1"
        >
          <button
            type="button"
            role="menuitem"
            className="nav-item w-full text-left"
            onClick={() => {
              setOpen(false);
              onNavigate?.();
              openPalette();
            }}
          >
            <Icon name="help" size={20} />
            <span>help</span>
            <span className="kbd ml-auto normal-case">⌘K</span>
          </button>
          <SignOut onNavigate={onNavigate} />
        </div>
      ) : null}
    </div>
  );
}

/**
 * Sign out.
 *
 * The request is what matters and the navigation is the consolation prize: the
 * cookie is `HttpOnly`, so this page cannot clear it and must ask the server
 * to revoke the session. A failure still navigates — if the API is unreachable
 * there is nothing to use anyway, and leaving somebody on a signed-in-looking
 * screen after they asked to leave is the worse of the two lies.
 *
 * `window.location` rather than the router, deliberately: every cached query in
 * this application is about a person who is no longer signed in, and a full
 * load is the one thing guaranteed to drop all of it.
 */
function SignOut({ onNavigate }: { onNavigate?: () => void }) {
  const [busy, setBusy] = useState(false);

  return (
    <button
      type="button"
      role="menuitem"
      className="nav-item w-full text-left"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        onNavigate?.();
        try {
          await api.logout();
        } catch {
          // Deliberately swallowed; see the header.
        }
        window.location.assign("/welcome");
      }}
    >
      <Icon name="close" size={20} />
      <span>{busy ? "signing out…" : "sign out"}</span>
    </button>
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
    return <p className="meta text-ink-2">reading corpus…</p>;
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
      {/* `ink-2` rather than `ink-3`, and only inside this panel. Muted
          metadata is `ink-3` everywhere in the application because everywhere
          else it sits on an opaque surface, where it measures 5.4:1. Here the
          surface is 72% white over a background that can be crossed by the
          cursor trail, and `ink-3` falls to 3.2:1 under the head of it. One
          step darker holds AA against everything the background can do. */}
      <dt className="meta text-ink-2">{label}</dt>
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
      <span className="meta text-ink-2">{state.label}</span>
    </div>
  );
}
