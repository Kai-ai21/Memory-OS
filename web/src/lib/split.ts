/**
 * The second panel's state, and who is allowed to open it.
 *
 * Split from `app/SplitProvider` so this file exports no components: the
 * provider is one, everything here is a hook or a constant, and mixing the two
 * in one module breaks fast refresh for every consumer.
 *
 * **The problem it solves: following a citation means leaving the answer.**
 * An agent answer cites four memories. Reading one of them navigates away from
 * the answer, and coming back re-renders it — so checking the evidence costs
 * you the thing you were checking. The same is true of a search result read
 * against its neighbours, and of a graph node read against the claim that drew
 * it.
 *
 * A context rather than props, because the openers are scattered: a citation
 * inside a streamed answer, a row in a search result list, an entity in the
 * graph. Threading a callback from the shell to each of those is six prop
 * chains through components that have nothing else to do with layout.
 *
 * **Only memories.** The panel takes a memory id and nothing else. A split
 * that can host any route is a second router, a second set of scroll positions
 * and a second history stack, and none of that is what the feature is for —
 * what you want beside your work is the thing your work refers to, and in this
 * corpus that is always a memory.
 *
 * The width is a percentage of the shell rather than pixels, so a window
 * resized narrower does not leave a 900px panel over a 400px column. Persisted,
 * because a width is a working preference and re-dragging it every session is
 * the kind of small tax that makes people stop using a feature.
 */

import { createContext, useContext, useEffect } from "react";

import { KEYS, read } from "./local";

/** Percent of the shell the panel takes. Clamped — see `clampWidth`. */
export const DEFAULT_WIDTH = 40;
const MIN_WIDTH = 22;
const MAX_WIDTH = 68;

/**
 * Kept inside a range rather than merely non-negative.
 *
 * Below about a fifth the panel cannot show a line of a path without wrapping,
 * and above about two thirds the thing you opened it *beside* is the one that
 * stops being usable — at which point you wanted a navigation, not a split.
 * Also the guard against a corrupt stored value: `NaN` from a hand-edited
 * `localStorage` would otherwise become `width: NaN%` and collapse the layout.
 */
export function clampWidth(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_WIDTH;
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, value));
}

export interface SplitState {
  /** The memory on the right, or `null` when the panel is closed. */
  memoryId: string | null;
  width: number;
  open: (memoryId: string) => void;
  close: () => void;
  /** `⌘\` — opens the last memory you looked at, or closes what is open. */
  toggle: () => void;
  setWidth: (percent: number) => void;
}

export const SplitContext = createContext<SplitState | null>(null);

/**
 * The split, from anywhere inside the shell.
 *
 * Returns a no-op shape rather than throwing where there is no provider. Every
 * page in this application is rendered directly in a dozen tests that mount the
 * component rather than the shell, and a hook that throws would turn "this
 * component now offers a split" into twelve unrelated test failures. A button
 * that does nothing outside the shell is the correct behaviour: outside the
 * shell there is no panel for it to open.
 */
export function useSplit(): SplitState {
  const context = useContext(SplitContext);
  return context ?? FALLBACK;
}

const FALLBACK: SplitState = {
  memoryId: null,
  width: DEFAULT_WIDTH,
  open: () => {},
  close: () => {},
  toggle: () => {},
  setWidth: () => {},
};

/** The newest `recents` entry that is a memory, for the `⌘\` fallback. */
export function readRecentMemoryId(): string | null {
  const recents = read<{ to?: string }[]>(KEYS.recents, []);
  if (!Array.isArray(recents)) return null;
  for (const entry of recents) {
    const match = /^\/memory\/(.+)$/.exec(entry?.to ?? "");
    if (match) return match[1];
  }
  return null;
}

/**
 * Bind `⌘\`, and close on `Esc`.
 *
 * Lives here rather than in the shell's keyboard effect only because it needs
 * the context; the shell calls it, so the keyboard model is still assembled in
 * one place.
 */
export function useSplitShortcuts(): void {
  const { toggle, close, memoryId } = useSplit();

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key === "\\") {
        // Claimed even while typing, like ⌘K: it is not a character anybody
        // is trying to enter with a modifier held.
        event.preventDefault();
        toggle();
        return;
      }
      /* `Esc` closes the split only when nothing is over it. The palette and
         the shortcuts sheet are modal and are dismissed first — closing the
         panel underneath at the same time would make one keypress undo two
         things, and `<dialog>` stops the event before it reaches here. */
      if (event.key === "Escape" && memoryId) close();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, close, memoryId]);
}

/**
 * Close the split when the route changes to the memory it is showing.
 *
 * Navigating to `/memory/x` while the panel is already showing `x` would leave
 * the same document rendered twice side by side, which reads as a bug.
 */
export function useCloseSplitOnDuplicate(pathname: string): void {
  const { memoryId, close } = useSplit();
  useEffect(() => {
    if (memoryId && pathname === `/memory/${memoryId}`) close();
  }, [pathname, memoryId, close]);
}
