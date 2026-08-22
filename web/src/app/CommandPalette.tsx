/**
 * ⌘K: the fastest path to anything, not just a list of routes.
 *
 * Hand-built rather than `@headlessui/react`. The two things that library is
 * worth taking a dependency for are focus trapping and the dialog semantics,
 * and `<dialog showModal>` gives both natively — the trap, the `Esc` handler,
 * the inert background and the `aria-modal` role are the platform's, not a
 * bundle's.
 *
 * **M9.11 turns it from a navigator into a command palette.** It did one thing
 * — go to a view — and that is the thing the sidebar already does, which made
 * ⌘K a keyboard alias for a click rather than a reason to reach for the
 * keyboard. Five groups now, in the order they are worth offering:
 *
 *   **recent**    the last five things opened, shown before you type
 *   **navigate**  every view, including the five the sidebar does not name
 *   **search**    the query itself, run against the corpus
 *   **memory**    a path, matched against one fetch
 *   **action**    new chat, toggle details, sign out
 *
 * **Recents first, and it is not a close call.** Most palette use is returning
 * to something you just had open — see `lib/recents`. Everything else in here
 * is reachable another way; that is not.
 *
 * **Matching is a subsequence, not `includes`.** `dnw` reaches
 * `decisions/new`. See `lib/fuzzy` for the scoring and why there is no
 * dependency.
 *
 * **Memories are matched client-side against one fetch.** A request per
 * keystroke would be the obvious build and it is the wrong one: the whole value
 * of this control is that it answers before you finish typing, and even a local
 * API cannot beat a filter over 500 strings already in memory. The fetch
 * happens when the palette first opens, not on mount, so a session that never
 * presses ⌘K never pays for it.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { fuzzyRank } from "../lib/fuzzy";
import { readRecents, recordRecent } from "../lib/recents";
import { PALETTE_ROUTES, type ViewRoute } from "./routes";

/** How many memory paths to hold for matching. */
const MEMORY_PAGE = 500;

/** The groups, in the order they are drawn. */
const GROUP_ORDER = ["recent", "navigate", "search", "memory", "action"] as const;

type Group = (typeof GROUP_ORDER)[number];

interface Entry {
  id: string;
  group: Group;
  label: string;
  detail: string;
  /**
   * What activating it does.
   *
   * A function rather than a route string, which is the change that let actions
   * exist at all. "Sign out" and "toggle details" are not destinations, and a
   * palette whose entries are all `to:` can only ever navigate.
   */
  run: () => void;
}

/**
 * The two actions that are not navigation and not local to this component.
 *
 * Dispatched as window events rather than wired through props. The sidebar owns
 * its details disclosure and the shell owns nothing relevant, so the
 * alternative is threading two callbacks from `Shell` through `CommandPalette`
 * for one row each — and the sidebar already listens for `⌘K` this way, in the
 * other direction. One mechanism, used both ways.
 */
export const TOGGLE_DETAILS_EVENT = "memo:toggle-details";

export function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);

  // Only once the palette has been opened. See the module note.
  const memories = useQuery({
    queryKey: ["memories", MEMORY_PAGE],
    queryFn: () => api.memories(MEMORY_PAGE),
    enabled: open,
    staleTime: 5 * 60_000,
  });

  // `showModal` rather than a rendered overlay, so the browser owns the focus
  // trap and the backdrop's inertness.
  useEffect(() => {
    const node = dialog.current;
    if (!node) return;
    if (open && !node.open) node.showModal();
    if (!open && node.open) node.close();
  }, [open]);

  // A fresh query every time it opens. Reopening to find the same thing you
  // just found is the rare case; reopening to find something else is the
  // normal one, and having to clear the box first is friction on every use.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
    }
  }, [open]);

  /* Read at open rather than at module scope, so a memory opened in this
     session is in the list the next time ⌘K is pressed rather than the next
     time the page is loaded. */
  const recents = useMemo(() => (open ? readRecents() : []), [open]);

  const entries = useMemo(
    () =>
      build(query, memories.data ?? [], recents, {
        navigate,
        close: onClose,
      }),
    [query, memories.data, recents, navigate, onClose],
  );

  // The active row can outrun the list as it shrinks under a longer query.
  useEffect(() => {
    setActive((current) => (current >= entries.length ? 0 : current));
  }, [entries.length]);

  function go(entry: Entry | undefined) {
    if (!entry) return;
    entry.run();
  }

  return (
    <dialog
      ref={dialog}
      // `Esc` fires `cancel` before `close`; both are routed to the same place
      // so the parent's `open` state can never disagree with the element's.
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClose={onClose}
      // The backdrop is a click target for dismissal, which is the convention
      // every palette follows and the one thing `<dialog>` does not give free.
      onClick={(event) => {
        if (event.target === dialog.current) onClose();
      }}
      aria-label="Command palette"
      /* Flat white with a hairline, like every other panel. The palette
         overlaps the page, and on dark that earned it a heavier blur; here the
         scrim behind it does that work, and glass is reserved for one button. */
      className="panel m-0 w-full max-w-xl p-0 text-ink backdrop:bg-scrim sm:mt-[12vh] sm:ml-[max(0px,calc(50%-18rem))]"
    >
      <div
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((current) => Math.min(entries.length - 1, current + 1));
          }
          if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive((current) => Math.max(0, current - 1));
          }
          if (event.key === "Enter") {
            event.preventDefault();
            go(entries[active]);
          }
        }}
      >
        <div className="flex items-baseline gap-2 border-b border-rule px-3 py-2.5">
          <span className="meta text-ink-3" aria-hidden>
            &gt;
          </span>
          <input
            // The dialog gives focus to the first focusable child on
            // `showModal`, which is this.
            autoFocus
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
            }}
            placeholder="go to a view, search the corpus, or open a path"
            aria-label="Command"
            aria-controls="palette-results"
            className="flex-1 bg-transparent font-mono text-sm text-ink outline-none placeholder:text-ink-3"
            spellCheck={false}
            autoComplete="off"
          />
          <span className="kbd hidden sm:inline">esc</span>
        </div>

        <ul
          id="palette-results"
          role="listbox"
          aria-label="Results"
          className="max-h-[52vh] overflow-y-auto"
          data-testid="palette-results"
        >
          {entries.length === 0 ? (
            <li className="meta px-3 py-4 text-ink-3">
              Nothing matches. Press enter to search the corpus for it anyway.
            </li>
          ) : (
            entries.map((entry, index) => (
              <Row
                key={entry.id}
                entry={entry}
                index={index}
                active={index === active}
                /* The group label is drawn on the first row of each run rather
                   than as a separate heading element. A heading is a row the
                   arrow keys have to skip, and every palette that renders one
                   as a list item eventually lands the selection on it. */
                heading={index === 0 || entries[index - 1].group !== entry.group}
                onHover={() => setActive(index)}
                onPick={() => go(entry)}
              />
            ))
          )}
        </ul>

        <p className="meta border-t border-rule px-3 py-1.5 text-ink-3">
          <span className="kbd">↑</span> <span className="kbd">↓</span> to move ·{" "}
          <span className="kbd">enter</span> to open · <span className="kbd">esc</span> to
          close
          {memories.isLoading ? " · loading paths…" : ""}
        </p>
      </div>
    </dialog>
  );
}

function Row({
  entry,
  active,
  heading,
  onHover,
  onPick,
}: {
  entry: Entry;
  index: number;
  active: boolean;
  heading: boolean;
  onHover: () => void;
  onPick: () => void;
}) {
  return (
    <li>
      {heading ? (
        <p className="meta-label bg-surface-tint/50 px-3 py-1 text-ink-3">{entry.group}</p>
      ) : null}
      <button
        type="button"
        role="option"
        aria-selected={active}
        // Hover moves the selection, so the mouse and the keyboard never
        // disagree about which row enter would open.
        onMouseEnter={onHover}
        onClick={onPick}
        className={`flex w-full items-baseline gap-3 border-l-2 px-3 py-1.5 text-left ${
          active
            ? "border-edge bg-surface-tint/70"
            : "border-transparent hover:bg-surface-tint/40"
        }`}
      >
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink">
          {entry.label}
        </span>
        <span className="meta hidden truncate text-ink-3 sm:block sm:max-w-56">
          {entry.detail}
        </span>
      </button>
    </li>
  );
}

interface Deps {
  navigate: (to: string) => void;
  close: () => void;
}

/**
 * What to offer for a query, in the order the reader wants it.
 *
 * Navigation first among the typed groups and always: the views are a fixed
 * fourteen, they are what the shortcut is mostly used for, and burying them
 * under forty file paths that happen to contain "search" would make the fast
 * path the slow one. The search action sits between the views and the paths —
 * it is the fallback that always works, and it should not be at the bottom of
 * a list of 500.
 */
function build(
  query: string,
  memories: { id: string; external_key: string; kind: string; title: string | null }[],
  recents: { to: string; label: string; kind: string }[],
  deps: Deps,
): Entry[] {
  const term = query.trim();

  /** Navigate, remember it, and shut the palette. Every entry ends this way. */
  function goTo(to: string, label: string, kind: string) {
    return () => {
      recordRecent({ to, label, kind });
      deps.navigate(to);
      deps.close();
    };
  }

  const actions = buildActions(deps);

  if (term.length === 0) {
    /* The resting state: what you had open, then everywhere you can go, then
       what you can do. No memories — listing 500 paths before anything has
       been typed is a wall, not a menu. */
    return [
      ...recents.map((recent) => ({
        id: `recent:${recent.to}`,
        group: "recent" as const,
        label: recent.label,
        detail: recent.kind,
        run: goTo(recent.to, recent.label, recent.kind),
      })),
      ...PALETTE_ROUTES.map(routeEntry),
      ...actions,
    ];
  }

  const views = fuzzyRank(PALETTE_ROUTES, term, (route) => [
    route.label,
    route.path,
    ...(route.aliases ?? []),
  ]).map(routeEntry);

  const search: Entry = {
    id: "action:search",
    group: "search",
    label: term,
    detail: "search the corpus for this",
    run: goTo(`/search?q=${encodeURIComponent(term)}`, term, "search"),
  };

  const paths: Entry[] = fuzzyRank(memories, term, (memory) => [
    memory.external_key,
    memory.title ?? "",
  ])
    .slice(0, 20)
    .map((memory) => ({
      id: `memory:${memory.id}`,
      group: "memory" as const,
      label: memory.external_key,
      detail: memory.title ?? memory.kind,
      run: goTo(`/memory/${memory.id}`, memory.external_key, "memory"),
    }));

  const matchedActions = fuzzyRank(actions, term, (action) => [action.label, action.detail]);

  return [...views, search, ...paths, ...matchedActions];

  function routeEntry(route: ViewRoute): Entry {
    return {
      id: `view:${route.path}`,
      group: "navigate",
      label: route.label,
      detail: route.blurb,
      run: goTo(route.path, route.label, "view"),
    };
  }
}

/**
 * The three things the palette can do that are not going somewhere.
 *
 * Deliberately three. A palette that lists every action in the application
 * becomes a menu you scroll, and the value of this one is that the answer is
 * always in the first few rows. These are the ones with no other keyboard
 * route: new chat is a sidebar click, details is a sidebar click, and sign out
 * is two.
 */
function buildActions(deps: Deps): Entry[] {
  return [
    {
      id: "action:new-chat",
      group: "action",
      label: "new chat",
      detail: "start a fresh conversation",
      run: () => {
        deps.navigate("/");
        deps.close();
      },
    },
    {
      id: "action:toggle-details",
      group: "action",
      label: "toggle details",
      detail: "show or hide the corpus figures in the sidebar",
      run: () => {
        window.dispatchEvent(new CustomEvent(TOGGLE_DETAILS_EVENT));
        deps.close();
      },
    },
    {
      id: "action:sign-out",
      group: "action",
      label: "sign out",
      detail: "end this session",
      run: () => {
        deps.close();
        /* A full load rather than the router: every cached query in this
           application is about a person who is no longer signed in. The request
           is fire-and-forget for the same reason `SignOut` in the sidebar
           swallows its failure — if the API is unreachable there is nothing to
           use anyway. */
        void api.logout().finally(() => window.location.assign("/welcome"));
      },
    },
  ];
}
