/**
 * A number that moves to its new value instead of snapping to it.
 *
 * The point is not decoration. A figure that changes while you are not looking
 * at it changes silently; a figure that counts is one you notice moved, which
 * for a corpus total or a result count is the whole information content of the
 * change.
 *
 * **Three cases where it must not animate, and all three are the difference
 * between this being useful and being a toy:**
 *
 *   1. **A jump of 100 or more.** Counting 3,833 → 3,901 over 400ms is a blur
 *      that reads as a slot machine, and the reader waits for it to settle
 *      before they can trust the figure. A large change should simply *be* the
 *      new number. See `COUNT_LIMIT`.
 *
 *   2. **The first render.** Counting up from zero on load is an animation
 *      about nothing, and it delays the first honest reading of every figure on
 *      the screen — on a dashboard that is the entire screen.
 *
 *   3. **Reduced motion.** No stylesheet can shorten a `requestAnimationFrame`
 *      loop writing text, so this is one of the few places that has to ask —
 *      see `lib/motion`.
 *
 * `requestAnimationFrame` rather than a CSS transition because there is no
 * animatable property here: the thing changing is the *content* of a text node.
 * `@property` and `counter-reset` can do this in CSS, and both cost more to
 * read than a twelve-line loop.
 *
 * Figures are already `tabular-nums` application-wide — see `index.css` — which
 * is what stops the surrounding layout twitching as the digits change width.
 * Without that this animation would be actively worse than snapping.
 */

import { useEffect, useRef, useState } from "react";

import { count as formatCount } from "../lib/format";
import { MOTION, shouldCount, useReducedMotion } from "../lib/motion";

/** Decelerating: fast at first, settling into the final value. */
function easeOut(t: number): number {
  return 1 - (1 - t) ** 3;
}

export function Count({
  value,
  className = "",
}: {
  value: number;
  className?: string;
}) {
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(value);

  /* What is currently on screen, read inside the animation loop. State would
     be a render behind by the time the next frame runs, and the loop needs the
     value it is actually interpolating from. */
  const from = useRef(value);
  const frame = useRef<number | undefined>(undefined);

  useEffect(() => {
    const start = from.current;

    if (reduced || !shouldCount(start, value)) {
      from.current = value;
      setShown(value);
      return;
    }

    const startedAt = performance.now();

    function step(now: number) {
      const progress = Math.min(1, (now - startedAt) / MOTION.countMs);
      const next = Math.round(start + (value - start) * easeOut(progress));
      setShown(next);

      if (progress < 1) {
        frame.current = requestAnimationFrame(step);
      } else {
        // Pinned exactly, not left on whatever the last interpolation rounded
        // to. An animated counter that stops one short of the real figure is
        // worse than no animation, because the number on screen is then wrong.
        from.current = value;
        setShown(value);
      }
    }

    frame.current = requestAnimationFrame(step);
    return () => {
      if (frame.current !== undefined) cancelAnimationFrame(frame.current);
      /* Interrupted by a newer value: the next run must interpolate from what
         is on screen right now, not from where this one started, or the number
         visibly jumps backwards before counting forwards again. */
      from.current = shown;
    };
    // `shown` is deliberately not a dependency — it changes on every frame, and
    // including it would restart the animation sixty times a second.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, reduced]);

  return (
    <span className={className} data-testid="count" data-value={value}>
      {formatCount(shown)}
    </span>
  );
}
