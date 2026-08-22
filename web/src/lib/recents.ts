/**
 * The last five things you opened.
 *
 * **This is the part of the palette that earns the shortcut.** Navigating to a
 * view you can see in the sidebar is a convenience; getting back to the memory
 * you had open twenty minutes ago, whose path you cannot remember, is the thing
 * there is otherwise no route to at all. So recents are shown *before anything
 * is typed*, above every other group.
 *
 * Five, not ten. The list is meant to be read at a glance without scrolling,
 * and past about five the thing you want is faster to find by typing.
 *
 * **What counts as "opened" is deliberately narrow.** Recording every route
 * change would fill this with whatever you clicked through on the way somewhere
 * — the sidebar is right there for those, and a recents list that is mostly
 * `/search` and `/timeline` is a recents list nobody reads. Two things are
 * recorded: anything activated *from the palette*, because that is an explicit
 * act of going somewhere, and any memory opened by any route, because a memory
 * is the thing whose address you cannot reconstruct.
 */

import { KEYS, pushRecent, read, write } from "./local";

export interface Recent {
  /** Where it goes. Also the identity — one entry per destination. */
  to: string;
  /** What to show. A path for a memory, a view name for a view. */
  label: string;
  /** The group label the palette draws beside it. */
  kind: string;
}

const CAP = 5;

export function readRecents(): Recent[] {
  const stored = read<Recent[]>(KEYS.recents, []);
  // Written by this module, but `localStorage` is shared with every other tab
  // and every past version of this application. A malformed entry renders as
  // `undefined` in a list row rather than throwing, which is worse than
  // dropping it.
  if (!Array.isArray(stored)) return [];
  return stored.filter(
    (entry): entry is Recent =>
      !!entry && typeof entry.to === "string" && typeof entry.label === "string",
  );
}

export function recordRecent(entry: Recent): void {
  write(KEYS.recents, pushRecent(readRecents(), entry, CAP, (a, b) => a.to === b.to));
}

export function clearRecents(): void {
  write(KEYS.recents, []);
}
