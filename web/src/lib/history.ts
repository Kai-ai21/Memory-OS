/**
 * The last ten things you searched for.
 *
 * **Local only, and this one is not a shrug.** A search history is the most
 * revealing thing this application could store — it is a list of what somebody
 * was trying to find out, in their own words — and the fact that it never
 * leaves the browser is the reason it can exist at all. No endpoint, no column,
 * nothing to leak, and a single button in settings that removes it. If this
 * ever needs to sync across machines, that is a conversation about consent,
 * not a migration.
 *
 * Ten, and newest first. Long enough to hold a session's worth of narrowing —
 * the usual pattern is four or five variations on one question — and short
 * enough to sit under the input without becoming a page.
 *
 * A repeated query moves to the front rather than appearing twice; see
 * `pushRecent`. Searching the same thing three times in a row should leave one
 * entry, not three.
 */

import { KEYS, pushRecent, read, write } from "./local";

const CAP = 10;

export function readHistory(): string[] {
  const stored = read<string[]>(KEYS.history, []);
  if (!Array.isArray(stored)) return [];
  // Filtered rather than trusted: this is shared with every past version of
  // the application, and a non-string here renders as `undefined` in a button.
  return stored.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
}

export function recordQuery(query: string): void {
  const trimmed = query.trim();
  if (!trimmed) return;
  write(KEYS.history, pushRecent(readHistory(), trimmed, CAP, (a, b) => a === b));
}

export function forgetQuery(query: string): string[] {
  const next = readHistory().filter((entry) => entry !== query);
  write(KEYS.history, next);
  return next;
}

export function clearHistory(): void {
  write(KEYS.history, []);
}
