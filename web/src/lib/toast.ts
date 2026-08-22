/**
 * The small confirmation in the corner, and the rules that keep it bearable.
 *
 * **Only for actions with no visible result.** Pinning a memory moves something
 * into a sidebar you may not be looking at; copying a path changes nothing on
 * screen at all; archiving a session removes a row you were not watching. Those
 * need a word. Anything whose result is already visible does not — a toast
 * saying "filter applied" beside a list that visibly filtered is noise, and
 * noise is how people learn to ignore the corner of the screen where the
 * important messages appear.
 *
 * **Never for errors, and this is enforced rather than documented.** There is
 * no `variant` and no `error()`; the only thing you can pass is a message and
 * an optional undo. A failure belongs inline, next to the control that failed,
 * where it is still there when you look back — a toast for a failure is a
 * message you can miss, and the person who misses it believes the thing
 * worked. `Failure` in `components/primitives` is where errors go.
 *
 * **One at a time.** A new toast replaces the one before it rather than
 * stacking. Two stacked toasts are already a list, three are a notification
 * centre, and the value of this pattern is entirely that it is one line you can
 * read without stopping — which stops being true the moment it can be two.
 * Replacing also means a burst of five pins leaves the fifth message on screen
 * rather than a queue that outlives the action by ten seconds.
 *
 * **2.5 seconds, or 5 with an undo.** Long enough to read six words, short
 * enough not to sit in the corner while you work. An undo needs longer, because
 * the sequence is read it, decide, then move the pointer — see `features/undo`.
 */

import { createContext, useContext } from "react";

/** Plain confirmation. */
export const TOAST_MS = 2500;
/** With an undo attached — the window has to survive reading and deciding. */
export const TOAST_UNDO_MS = 5000;

export interface Toast {
  /** Monotonic, so replacing a toast with identical text still restarts it. */
  id: number;
  message: string;
  /** Present iff the action can be taken back. See `features/undo`. */
  undo?: () => void;
}

export interface ToastApi {
  /**
   * Show a confirmation.
   *
   * `undo` is the only option, deliberately — see the header on why there is no
   * severity. Returns nothing: a toast is fire-and-forget, and a caller that
   * wants to know what happened should be rendering it inline instead.
   */
  show: (message: string, options?: { undo?: () => void }) => void;
  dismiss: () => void;
}

export const ToastContext = createContext<ToastApi | null>(null);

/**
 * A no-op outside the provider, for the same reason `useSplit` is: a dozen
 * tests mount a page rather than the shell, and a component gaining a
 * confirmation should not turn those into failures. Silence is the correct
 * degradation — the action still happens, it just says nothing.
 */
export function useToast(): ToastApi {
  return useContext(ToastContext) ?? SILENT;
}

const SILENT: ToastApi = { show: () => {}, dismiss: () => {} };
