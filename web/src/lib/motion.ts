/**
 * Motion, and the one question every piece of it has to ask first.
 *
 * **`prefers-reduced-motion` is not a nice-to-have here.** This application
 * already treats it as a hard gate for the background canvas — see
 * `BackgroundLayer/preference` — and M9.12 adds motion to the parts of the
 * interface somebody actually has to read: the page they navigated to, the
 * results they are scanning, the number they are checking. Motion in the
 * reading path is exactly what the setting exists to switch off.
 *
 * Two mechanisms, deliberately, because there are two kinds of motion:
 *
 *   **CSS animations** are covered by the blanket rule in `index.css`, which
 *   collapses every `animation-duration` and `transition-duration` to 0.01ms.
 *   That is the right tool: it needs no component to remember, and it cannot be
 *   forgotten by the next one added. Everything in this milestone that is a
 *   keyframe is handled there and nowhere else.
 *
 *   **JavaScript-driven motion** cannot be reached by CSS. A number counting
 *   from 271 to 274 over 400ms is a `requestAnimationFrame` loop writing text,
 *   and no stylesheet can shorten it. Those ask this hook and jump straight to
 *   the final value.
 *
 * The hook is live rather than read once. Somebody turning the setting on
 * mid-session — which is exactly what somebody who has started feeling unwell
 * does — should not have to reload to be believed.
 */

import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

/**
 * Whether motion should be suppressed.
 *
 * **Defaults to `true` where the question cannot be asked.** jsdom and a server
 * render have no `matchMedia`, and the failure this guards against is
 * one-directional: defaulting to "motion is fine" in an environment that cannot
 * answer means an accessibility gate that fails *open*. It also keeps three
 * hundred component tests free of animation frames they never asked for.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(prefersReducedMotion);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(QUERY);
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    // Re-read on mount: the value may have changed between the initial state
    // and this effect, and on the very first render there was no listener.
    onChange();
    return () => media.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

/** The same question, for code that is not a component. */
export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return true;
  return window.matchMedia(QUERY).matches;
}

/**
 * How long the entering content is given, and how far it travels.
 *
 * Exported so the components and the tests spend the same numbers. Every one of
 * these is under 200ms: past that a transition stops reading as the interface
 * responding and starts reading as the interface deciding whether to.
 */
export const MOTION = {
  /** A route change. 180ms, 4px. */
  routeMs: 180,
  /** Between staggered items. See `STAGGER_CAP`. */
  stepMs: 30,
  /** One arriving list item. */
  arriveMs: 180,
  /** A number counting to its new value. */
  countMs: 400,
  /** The press on the send button. */
  pressMs: 120,
} as const;

/**
 * How many items stagger before the rest arrive together.
 *
 * **The cap is the whole reason a stagger is tolerable.** At 30ms a step,
 * fifty rows put the last one on screen 1.5 seconds after the first — which is
 * slower than no animation at all, and the delay lands on exactly the row
 * somebody scrolled down to find. Eight is a screenful; past that the effect
 * has already been communicated and only the cost is left.
 */
export const STAGGER_CAP = 8;

/**
 * The `animation-delay` for item `index`, in ms, capped.
 *
 * Returns 0 under reduced motion so the caller does not have to branch — an
 * item with no delay and a duration the stylesheet has already collapsed is an
 * item that simply appears.
 */
export function staggerDelay(index: number, reduced = false): number {
  if (reduced) return 0;
  return Math.min(index, STAGGER_CAP) * MOTION.stepMs;
}

/**
 * Whether a number change is worth animating.
 *
 * **Under 100, and the cap is not arbitrary.** A count-up is legible when the
 * eye can follow the digits; from 3,833 to 3,901 over 400ms it is a blur that
 * reads as a slot machine, and the reader waits for it to stop before they can
 * trust the figure. A large jump should simply *be* the new number.
 *
 * A first render is never animated either: counting up from zero on page load
 * is an animation about nothing, and it delays the first honest reading of
 * every figure on the screen.
 */
export const COUNT_LIMIT = 100;

export function shouldCount(from: number, to: number): boolean {
  return from !== to && Math.abs(to - from) < COUNT_LIMIT;
}

/**
 * The same stagger as props, for an element that already exists.
 *
 * `Arrive` wraps; this one does not. Several lists in this application render
 * their own `<li>` — the chat transcript is one — and wrapping those would put
 * an `<li>` inside an `<li>`, which is invalid and changes what a screen reader
 * counts. Spreading `className` and `style` onto the element that is already
 * there costs nothing and avoids the extra node entirely.
 *
 * Returns an empty object under reduced motion, so the caller spreads nothing.
 */
export function useArrive(index: number): {
  className?: string;
  style?: React.CSSProperties;
} {
  const reduced = useReducedMotion();
  if (reduced) return {};
  return {
    className: "arrive",
    style: { animationDelay: `${staggerDelay(index)}ms` },
  };
}
