/**
 * The three or four memories you keep coming back to.
 *
 * **Frontend state, and it stays that way.** A pin is not a property of a
 * memory — it is a property of the person looking at it, on this machine, this
 * month. Sending it would make it a column, a migration, an endpoint and a
 * scoping rule, and it would still be modelling the wrong thing: two people
 * looking at one corpus do not keep returning to the same four files.
 *
 * **A store with subscribers rather than a context.** Pins are written from the
 * memory page and from a search result, and read by the sidebar — three places
 * with no common ancestor below the shell. A context would work and would mean
 * every consumer re-renders when any pin changes; `useSyncExternalStore` is the
 * API React added for exactly this shape, and it is what makes the pin button
 * on result forty not re-render when you pin result one.
 *
 * The label is stored beside the id on purpose. The sidebar has to draw a list
 * of pins on first paint, and the alternative is four requests before the nav
 * can render — for a list whose whole point is that it is instant.
 */

import { useSyncExternalStore } from "react";

import { KEYS, read, write } from "./local";

export interface Pin {
  id: string;
  /** The path, as the sidebar draws it. Stored so the nav needs no fetch. */
  label: string;
}

/** Nothing enforces this, but a pin list past a dozen is a bookmarks bar. */
const CAP = 20;

let cache: Pin[] | null = null;
const listeners = new Set<() => void>();

function load(): Pin[] {
  if (cache) return cache;
  const stored = read<Pin[]>(KEYS.pins, []);
  cache = Array.isArray(stored)
    ? stored.filter(
        (pin): pin is Pin =>
          !!pin && typeof pin.id === "string" && typeof pin.label === "string",
      )
    : [];
  return cache;
}

function commit(next: Pin[]): void {
  cache = next;
  write(KEYS.pins, next);
  for (const listener of listeners) listener();
}

/* The three functions `useSyncExternalStore` needs. `getSnapshot` must return
   a stable reference for an unchanged store or React re-renders forever, which
   is why `cache` exists rather than re-reading `localStorage` on every call. */
export function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getPins(): Pin[] {
  return load();
}

export function isPinned(id: string): boolean {
  return load().some((pin) => pin.id === id);
}

/** Pin, or unpin if it is already there. Returns the state it ended in. */
export function togglePin(pin: Pin): boolean {
  const current = load();
  const existing = current.some((entry) => entry.id === pin.id);
  commit(existing ? current.filter((entry) => entry.id !== pin.id) : [pin, ...current].slice(0, CAP));
  return !existing;
}

/** Put one back exactly where it was. The undo path — see `features/undo`. */
export function restorePin(pin: Pin, index: number): void {
  const current = load().filter((entry) => entry.id !== pin.id);
  commit([...current.slice(0, index), pin, ...current.slice(index)]);
}

export function indexOfPin(id: string): number {
  return load().findIndex((pin) => pin.id === id);
}

export function clearPins(): void {
  commit([]);
}

/**
 * The pins, as a component sees them.
 *
 * `useSyncExternalStore` rather than a context: pins are written from the
 * memory page and from a search result and read by the sidebar, three places
 * with no common ancestor below the shell — and this is the API React added
 * for precisely that shape. The server snapshot is the same function because
 * there is no server render; if there ever is, an unpinned list is the right
 * thing to send.
 */
export function usePins(): Pin[] {
  return useSyncExternalStore(subscribe, getPins, getPins);
}
