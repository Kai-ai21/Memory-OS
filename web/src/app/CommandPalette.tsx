/**
 * ⌘K: jump to a view, run a search, or open a memory by path.
 *
 * Hand-built rather than `@headlessui/react`. The two things that library is
 * worth taking a dependency for are focus trapping and the dialog semantics,
 * and `<dialog showModal>` gives both natively — the trap, the `Esc` handler,
 * the inert background and the `aria-modal` role are the platform's, not a
 * bundle's.
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
import { PALETTE_ROUTES, type ViewRoute } from "./routes";

/** How many memory paths to hold for matching. */
const MEMORY_PAGE = 500;

interface Entry {
  id: string;
  kind: "view" | "search" | "memory";
  label: string;
  detail: string;
  to: string;
}

export function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const list = useRef<HTMLUListElement>(null);
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

  const entries = useMemo(
    () => match(query, memories.data ?? []),
    [query, memories.data],
  );

  // The active row can outrun the list as it shrinks under a longer query.
  useEffect(() => {
    setActive((current) => (current >= entries.length ? 0 : current));
  }, [entries.length]);

  function go(entry: Entry | undefined) {
    if (!entry) return;
    navigate(entry.to);
    onClose();
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
      className="m-0 w-full max-w-xl border border-rule-strong bg-float p-0 text-ink backdrop:bg-scrim sm:mt-[12vh] sm:ml-[max(0px,calc(50%-18rem))]"
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
          <span className="meta text-faint" aria-hidden>
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
            className="flex-1 bg-transparent font-mono text-sm text-ink outline-none placeholder:text-faint"
            spellCheck={false}
            autoComplete="off"
          />
          <span className="kbd hidden sm:inline">esc</span>
        </div>

        <ul
          ref={list}
          id="palette-results"
          role="listbox"
          aria-label="Results"
          className="max-h-[52vh] overflow-y-auto"
          data-testid="palette-results"
        >
          {entries.length === 0 ? (
            <li className="meta px-3 py-4 text-faint">
              Nothing matches. Press enter to search the corpus for it anyway.
            </li>
          ) : (
            entries.map((entry, index) => (
              <li key={entry.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={index === active}
                  // Hover moves the selection, so the mouse and the keyboard
                  // never disagree about which row enter would open.
                  onMouseEnter={() => setActive(index)}
                  onClick={() => go(entry)}
                  className={`flex w-full items-baseline gap-3 border-l-2 px-3 py-1.5 text-left ${
                    index === active
                      ? "border-edge bg-sunken/70"
                      : "border-transparent hover:bg-sunken/40"
                  }`}
                >
                  <span className="meta-label w-14 shrink-0">{entry.kind}</span>
                  <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink">
                    {entry.label}
                  </span>
                  <span className="meta hidden truncate text-faint sm:block sm:max-w-56">
                    {entry.detail}
                  </span>
                </button>
              </li>
            ))
          )}
        </ul>

        <p className="meta border-t border-rule px-3 py-1.5 text-faint">
          <span className="kbd">↑</span> <span className="kbd">↓</span> to move ·{" "}
          <span className="kbd">enter</span> to open · <span className="kbd">esc</span> to
          close
          {memories.isLoading ? " · loading paths…" : ""}
        </p>
      </div>
    </dialog>
  );
}

/**
 * What to offer for a query, in the order the reader wants it.
 *
 * Views first and always: they are nine fixed things, they are what the
 * shortcut is mostly used for, and burying them under forty file paths that
 * happen to contain "search" would make the fast path the slow one. The search
 * action sits between the views and the paths — it is the fallback that always
 * works, and it should not be at the bottom of a list of 500.
 */
function match(
  query: string,
  memories: { id: string; external_key: string; kind: string; title: string | null }[],
): Entry[] {
  const term = query.trim().toLowerCase();

  const views: Entry[] = PALETTE_ROUTES.filter((route) => matchesRoute(route, term)).map(
    (route) => ({
      id: `view:${route.path}`,
      kind: "view",
      label: route.planned ? `${route.label} (planned)` : route.label,
      detail: route.blurb,
      to: route.path,
    }),
  );

  if (term.length === 0) {
    // The resting state: every view, nothing else. Listing 500 paths before
    // anything has been typed is a wall, not a menu.
    return views;
  }

  const search: Entry = {
    id: "action:search",
    kind: "search",
    label: query.trim(),
    detail: "search the corpus for this",
    to: `/search?q=${encodeURIComponent(query.trim())}`,
  };

  const paths: Entry[] = memories
    .filter(
      (memory) =>
        memory.external_key.toLowerCase().includes(term) ||
        (memory.title ?? "").toLowerCase().includes(term),
    )
    .slice(0, 20)
    .map((memory) => ({
      id: `memory:${memory.id}`,
      kind: "memory",
      label: memory.external_key,
      detail: memory.title ?? memory.kind,
      to: `/memory/${memory.id}`,
    }));

  return [...views, search, ...paths];
}

function matchesRoute(route: ViewRoute, term: string): boolean {
  if (term.length === 0) return true;
  if (route.label.includes(term)) return true;
  if (route.path.includes(term)) return true;
  return (route.aliases ?? []).some((alias) => alias.includes(term));
}
