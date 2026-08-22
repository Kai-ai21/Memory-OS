/**
 * The panel: who you are, where you can go, and one thing worth reading.
 *
 * **It is a floating panel, not a column.** 264px of glass inset 12px from the
 * top, left and bottom of the viewport, with a 20px radius — and the 12px of
 * nothing around it is the point rather than a detail of it. The two drifting
 * radials and the cursor trail run underneath, and being able to see them past
 * the edges and through the face is what makes the material read as glass
 * instead of as a pale grey rectangle. See `.glass-panel`.
 *
 * **M9.9 takes the terminal out of the navigation.** Every label in here was
 * lowercase JetBrains Mono at 11px with 0.08em of tracking, which is the
 * register this application uses for paths, scores, offsets and hashes — and
 * using it for the nav said that a place you can go is the same kind of thing
 * as a byte offset. It is not. Navigation is prose: Inter at 14.5px, sentence
 * case, no tracking, 450 at rest and 600 where you are.
 *
 * **The mono is still here and is still right — inside `details`.** Four counts
 * that are read against each other need tabular figures and a column that
 * lines up. That is the whole of the exception, and `Sidebar.test.tsx` asserts
 * that nothing outside that block has crossed back over.
 *
 * **The icons are lucide now, one family at one weight.** They were
 * hand-drawn on Material's grid, which was the right call while they were
 * decoration beside a word in caps; with the caps gone the glyph is the first
 * thing the eye lands on, and a set drawn by hand at 1.5px next to text at
 * 14.5px is a set whose inconsistencies you can see. 18px, stroke 1.5 — not
 * lucide's default 2, which is heavy against text this size and is the single
 * most common reason an icon set looks clumsy.
 *
 * **The active item still carries no accent.** It is a `surface-tint` fill and
 * a step up in weight, because where you already are is not something you can
 * click. See rule 1 in `tokens.css`, and the test that pins it.
 *
 * **The corpus figures live behind `details` rather than in the panel.** They
 * are the answer to "is this thing loaded and working", which is worth one
 * click and is not worth six permanent rows — and the disclosure remembers, so
 * anybody who wants them up keeps them up. They are also still the honest frame
 * for every other view: three results out of a corpus of 282 memories means
 * something different from three out of 30,000.
 */

import { useEffect, useRef, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronDown,
  Compass,
  HelpCircle,
  Keyboard,
  LogOut,
  MoreHorizontal,
  PanelLeft,
  Plus,
  Search,
  SlidersHorizontal,
} from "lucide-react";

import { api } from "../api/client";
import { count } from "../lib/format";
import { Button } from "../components/primitives";
import { GROUPS, HOME, inGroup, type ViewRoute } from "./routes";

/** Every glyph in this panel, at one size and one weight. */
const GLYPH = { size: 18, strokeWidth: 1.5 } as const;

/**
 * The route table's label, as the panel draws it.
 *
 * **The table stays lowercase and this function is why.** `label` is read by
 * the command palette as well as by the sidebar, and the palette matches typed
 * text against it — a table full of capitals would mean either a palette that
 * misses `graph` or a `toLowerCase()` at every comparison. Sentence case is a
 * fact about how this panel *draws* a label, not about what the label is, so
 * it lives here.
 *
 * First letter only, deliberately. `text-transform: capitalize` would have been
 * one line of CSS and would render "New Chat" and "Sign Out", which is title
 * case and is not what was asked for.
 */
function sentence(label: string): string {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

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
  onShortcuts,
}: {
  onNavigate?: () => void;
  /** Hide the panel. The shell owns the state; this is the handle for it. */
  onCollapse?: () => void;
  /** Open the shortcuts sheet. The shell owns it, for the same reason. */
  onShortcuts?: () => void;
}) {
  return (
    /* No horizontal padding on the panel itself: the two hairlines — under the
       header and above the footer — run the full width of the glass, and a rule
       that stops 8px short of the edge on both sides reads as a mistake rather
       than as a decision. Every block below sets its own inset instead. */
    <div className="glass-panel flex h-full flex-col overflow-hidden py-3">
      <Header onCollapse={onCollapse} />

      {/* The only scrolling region. The card and the footer are pinned, so the
          promoted card cannot be scrolled out of the panel on a short window —
          which is the entire reason it is promoted. */}
      <nav
        className="flex min-h-0 flex-1 flex-col overflow-y-auto px-2 py-3"
        aria-label="Views"
      >
        <div className="nav-group flex flex-col">
          <NewConversation onNavigate={onNavigate} />
          <Item route={HOME} onNavigate={onNavigate} />
          {inGroup("primary").map((route) => (
            <Item key={route.path} route={route} onNavigate={onNavigate} />
          ))}
        </div>

        {/* 18px between the groups with a hairline through the middle of it,
            inset 12px from each side so it lines up with the rows rather than
            with the panel. Drawn by `.nav-group + .nav-group::before` — the gap
            alone read as inconsistent spacing, and a rule is what turns two
            runs of rows into two things. Still no headings: a heading is a row
            you cannot click. */}
        {GROUPS.filter((group) => group !== "primary").map((group) => {
          const routes = inGroup(group);
          if (routes.length === 0) return null;
          return (
            <div key={group} className="nav-group flex flex-col">
              {routes.map((route) => (
                <Item key={route.path} route={route} onNavigate={onNavigate} />
              ))}
            </div>
          );
        })}
      </nav>

      <Promoted onNavigate={onNavigate} />
      <Footer onNavigate={onNavigate} onShortcuts={onShortcuts} />
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
 * is already a `Search` row four lines below that goes to the view; a second
 * control doing the identical thing would be a wasted affordance, and the
 * palette is the thing this icon means everywhere else — jump to anything.
 */
function Header({ onCollapse }: { onCollapse?: () => void }) {
  return (
    <div className="flex items-center justify-between border-b border-rule px-3 pb-3.5">
      {/* 30px, no border. A ring around a gradient is a second edge on an
          object whose whole job is to be the one soft thing in the panel. */}
      <span
        className="avatar size-[30px] shrink-0 rounded-full"
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
          <Search {...GLYPH} />
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="Hide navigation"
          title="Hide navigation"
          onClick={onCollapse}
        >
          <PanelLeft {...GLYPH} />
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
 * the list because it is first in the list, which is enough.
 *
 * It still starts a *new session* rather than merely navigating to chat, which
 * is why this and the `Chat` row are both here and are not redundant. Dropping
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
      <Plus {...GLYPH} />
      <span>New chat</span>
    </button>
  );
}

function Item({ route, onNavigate }: { route: ViewRoute; onNavigate?: () => void }) {
  const Glyph = route.icon;

  return (
    <NavLink
      to={route.path}
      end={route.path === "/"}
      onClick={onNavigate}
      title={route.blurb}
      className={({ isActive }) => `nav-item ${isActive ? "nav-item-on" : ""}`}
    >
      <Glyph {...GLYPH} />
      <span>{sentence(route.label)}</span>
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
 *
 * `Compass` rather than `BookOpen`, which is what `/overview` wears four rows
 * up. Two glyphs for one destination is the same problem as two rows for it.
 */
function Promoted({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <NavLink to="/overview" onClick={onNavigate} className="promo mx-2 mb-3 shrink-0">
      <Compass size={16} strokeWidth={1.5} className="mt-px shrink-0 text-ink-3" />
      <span className="flex flex-col gap-0.5">
        <span className="promo-title">What MEMO does</span>
        <span className="promo-subtitle">A short tour of how memory works</span>
      </span>
    </NavLink>
  );
}

/**
 * Beneath the card, under a hairline: the corpus, and everything meta.
 *
 * **M9.10 adds `Shortcuts` to `More`.** It is the third thing under there that
 * is about the application rather than in it, and it exists because `?` cannot
 * advertise itself — see the note on the row.
 *
 * **Two permanent rows became one three-dot button.** `Help` and `Sign out`
 * were pinned to the bottom of every screen — one of them a keyboard shortcut
 * that is already bound globally, the other a thing you do once a month. Under
 * `More` they cost a click and no space, which is the correct trade for both.
 * There is still no settings page in this application and this milestone does
 * not invent one; `More` holds what actually exists.
 */
function Footer({
  onNavigate,
  onShortcuts,
}: {
  onNavigate?: () => void;
  onShortcuts?: () => void;
}) {
  const [open, setOpen] = useState(readDetails);

  return (
    <div className="shrink-0 border-t border-rule px-2 pt-2">
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
        <SlidersHorizontal {...GLYPH} />
        <span>Details</span>
        <ChevronDown
          size={16}
          strokeWidth={1.5}
          aria-hidden
          className={`ml-auto transition-transform duration-(--dur-state) ease-(--ease-out) ${open ? "rotate-180" : ""}`}
        />
      </button>

      {/* **The one place the mono survives in this panel, and the one place it
          belongs.** Four counts read against each other need figures that line
          up in a column; that is what `.meta` is for and what Inter, which has
          proportional digits by default here, would not give. */}
      {open ? (
        <div id="sidebar-details" className="flex flex-col gap-3 px-3 py-2">
          <CorpusFigures />
          <Health />
        </div>
      ) : null}

      <More onNavigate={onNavigate} onShortcuts={onShortcuts} />
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
function More({
  onNavigate,
  onShortcuts,
}: {
  onNavigate?: () => void;
  onShortcuts?: () => void;
}) {
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
        <MoreHorizontal {...GLYPH} />
        <span>More</span>
      </button>

      {open ? (
        <div
          role="menu"
          /* Opaque, unlike everything else in this panel — see `.menu`. It
             covers two rows of the footer, and glass over text is text you can
             read through the thing covering it. `bottom-full` because there is
             nothing below it: this is the last thing in a panel that ends 12px
             from the bottom of the window. */
          className="menu absolute bottom-full left-0 z-10 mb-1 flex w-full flex-col p-1"
        >
          {/* **The sheet needs a way in that is not the sheet's own shortcut.**
              `?` opens it, and somebody who does not know `?` exists is
              precisely the person the sheet is for — a shortcut that can only
              be discovered by pressing it is not discoverable. This row is the
              answer to that, and `More` is where it belongs: it is meta about
              the application rather than a place in it, which is the same
              reason Help and Sign out are here. */}
          <button
            type="button"
            role="menuitem"
            className="nav-item w-full text-left"
            onClick={() => {
              setOpen(false);
              onNavigate?.();
              onShortcuts?.();
            }}
          >
            <Keyboard {...GLYPH} />
            <span>Shortcuts</span>
            <span className="shortcut ml-auto">?</span>
          </button>
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
            <HelpCircle {...GLYPH} />
            <span>Help</span>
            {/* Not a `.kbd`, which is mono. macOS draws ⌘K in the system face
                in its own menus, and a key cap in this panel would be the one
                terminal artefact left in it. */}
            <span className="shortcut ml-auto">⌘K</span>
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
    <Button
      role="menuitem"
      className="nav-item w-full text-left"
      loading={busy}
      icon={<LogOut {...GLYPH} />}
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
      <span>Sign out</span>
    </Button>
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
