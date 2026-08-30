/**
 * A list item arriving, staggered by its position.
 *
 * **The cap is the feature.** At 30ms a step, fifty rows put the last one on
 * screen a second and a half after the first — slower than no animation, and
 * the delay lands on exactly the row somebody scrolled to find. Eight steps is
 * a screenful; everything past that shares the eighth delay and arrives
 * together. See `STAGGER_CAP`.
 *
 * The delay is an inline `animationDelay` rather than a class per index,
 * because the alternative is eight utility classes that exist only to encode a
 * number. `backwards` fill — set on `.arrive` in the stylesheet — keeps a
 * waiting item invisible instead of drawing it at full opacity and snapping it
 * back to zero when its turn comes.
 *
 * **Not for content that was already on screen.** This animates on mount, so
 * anything that remounts for another reason — a filter changing, a refetch
 * replacing the array — animates again. That is correct for a list of results
 * arriving and wrong for a row updating in place, which is why this wraps items
 * in freshly-loaded lists and nothing else.
 */

import { staggerDelay, useReducedMotion } from "../lib/motion";

export function Arrive({
  index,
  children,
  as: Tag = "div",
  className = "",
}: {
  /** Position in the list. Delays are capped, so large values are safe. */
  index: number;
  children: React.ReactNode;
  /** `li` inside a list, so the wrapper does not break list semantics. */
  as?: "div" | "li";
  className?: string;
}) {
  const reduced = useReducedMotion();

  /* Under reduced motion the element renders with no animation class at all
     rather than with a collapsed duration. The stylesheet would have handled
     the duration, but not the `backwards` fill — which starts an item at
     `opacity: 0` and would leave it there for the 0.01ms the guard allows,
     producing a flash of nothing on every list. */
  if (reduced) {
    return <Tag className={className}>{children}</Tag>;
  }

  return (
    <Tag
      className={`arrive ${className}`}
      style={{ animationDelay: `${staggerDelay(index)}ms` }}
      data-testid="arrive"
    >
      {children}
    </Tag>
  );
}
