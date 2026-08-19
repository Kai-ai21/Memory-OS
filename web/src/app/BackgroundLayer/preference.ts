/**
 * The three questions asked before a single particle is drawn.
 *
 * Two of them are not preferences at all and are not negotiable:
 *
 * **Reduced motion.** Drifting specks in the periphery are a vestibular
 * trigger, and this is the effect a person who set that switch was setting it
 * for. `reduce` means the canvas is never created — not "created and paused",
 * not "created with a shorter animation". The gradients render and nothing
 * else does.
 *
 * **No cursor, no effect.** The whole thing follows a pointer. On a touch
 * screen there is nothing to follow, so a loop would burn battery to render
 * particles at the last place a finger happened to lift. Gated on a *coarse*
 * primary pointer rather than on `ontouchstart`, which is true on plenty of
 * laptops with a trackpad and a touchscreen both.
 *
 * The third is a real preference, stored in `localStorage` under one key, and
 * defaulting to on. Reading it is wrapped because `localStorage` throws rather
 * than returns null in a Safari private window, and a background effect is not
 * worth taking the application down over.
 */

export const STORAGE_KEY = "memo:background-particles";

/** A media query, or `null` where there is no `matchMedia` (jsdom, SSR). */
function matches(query: string): boolean | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return null;
  }
  return window.matchMedia(query).matches;
}

export function prefersReducedMotion(): boolean {
  return matches("(prefers-reduced-motion: reduce)") === true;
}

/**
 * Whether the primary pointer is a finger.
 *
 * Unknown counts as "not touch": where `matchMedia` is absent the effect stays
 * available, because the alternative is silently disabling it everywhere the
 * query cannot be asked.
 */
export function isTouchPrimary(): boolean {
  return matches("(pointer: coarse)") === true;
}

export function readPreference(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) !== "off";
  } catch {
    return true;
  }
}

export function writePreference(on: boolean): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, on ? "on" : "off");
  } catch {
    // A preference that cannot be stored is still a preference for this
    // session; the state lives in React either way.
  }
}

/** Whether the two gating queries can be asked at all. */
export function canAskMedia(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function";
}

/**
 * Whether the effect may run at all, before the stored preference is consulted.
 *
 * Separate from `readPreference` because the two failures are different: this
 * one means "there must be no toggle either", since offering to switch on an
 * effect that is forbidden is worse than offering nothing.
 *
 * **No `matchMedia`, no effect.** Every browser this ships to has it; the
 * environments that do not are jsdom and a server render. Defaulting to on
 * there would mean an accessibility gate that fails open — running a
 * vestibular-trigger animation precisely where the question "do you want
 * motion?" could not be asked — and it would mount a canvas in every component
 * test in the suite for no benefit to any of them.
 */
export function particlesPermitted(): boolean {
  return canAskMedia() && !prefersReducedMotion() && !isTouchPrimary();
}
