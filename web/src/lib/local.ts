/**
 * Everything this application remembers on its own.
 *
 * **The rule for this file: nothing in here has ever been sent to the API.**
 * M9.11 adds five pieces of state that are opinions about *this browser* rather
 * than facts about the corpus — which memories you pinned, what you searched
 * for, how wide you dragged the split, how dense you want rows, what you opened
 * last. None of them belongs in the database. A pin is not a property of a
 * memory; it is a property of the person looking at it on this machine, and
 * sending it would make it everybody's, would need a migration, an endpoint and
 * a scoping rule, and would still be wrong.
 *
 * One module rather than a `try` block at each call site. There were already
 * two hand-rolled versions of this before M9.11 — the sidebar's details
 * disclosure and the particle preference — and each had its own copy of the
 * same `try`/`catch`, which is the point at which the third copy should become
 * a function instead.
 *
 * **Everything is wrapped, and the reason is Safari.** A private window throws
 * on `localStorage` access rather than returning `null`, so an unguarded read
 * at module scope takes the whole application down in a mode people actually
 * use. A read that fails returns the fallback; a write that fails is dropped
 * and the state lives in React for the session, which is the correct
 * degradation for every one of these — none is worth an error message.
 *
 * Keys are namespaced `memo:` and listed in `KEYS` so that "what does this
 * application store about me" has one answer, and so the settings sheet can
 * clear them without knowing what they mean.
 */

export const KEYS = {
  /** Feature 5 — the memories pinned to the top of the sidebar. */
  pins: "memo:pins",
  /** Feature 4 — the last ten search queries, newest first. */
  history: "memo:search-history",
  /** Feature 1 — the last five things opened, newest first. */
  recents: "memo:recents",
  /** Feature 2 — the split panel's width, as a percentage of the shell. */
  splitWidth: "memo:split-width",
  /** Feature 8 — "comfortable" or "compact". */
  density: "memo:density",
} as const;

/** Read and parse a JSON value, or the fallback if anything at all goes wrong. */
export function read<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    // Unavailable, or holding something this version cannot parse — a shape
    // written by an older build is the common case and is not an error worth
    // surfacing. The fallback is always a valid value.
    return fallback;
  }
}

export function write(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Quota, or a private window. The state lives in React either way.
  }
}

export function remove(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Nothing to do and nothing worth saying.
  }
}

/**
 * Push onto the front of a capped, de-duplicated list.
 *
 * The shape three of the five features want: recents, history and pins are all
 * "most recent first, no repeats, keep the last N". De-duplication happens
 * before the cap, so re-opening the thing at position four moves it to the
 * front rather than adding a sixth entry and evicting something.
 */
export function pushRecent<T>(list: T[], item: T, cap: number, same: (a: T, b: T) => boolean): T[] {
  return [item, ...list.filter((existing) => !same(existing, item))].slice(0, cap);
}
