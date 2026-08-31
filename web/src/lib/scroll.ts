/**
 * Following a growing transcript without hijacking it.
 *
 * **The rule: follow the bottom only while the reader is already there.**
 * Auto-scrolling somebody who has deliberately scrolled back to re-read
 * something is one of the most irritating things an interface can do, and it is
 * a specific bug rather than a matter of taste — the reader loses their place,
 * and the only way to get it back is to scroll up again and lose it again on
 * the next token.
 *
 * So "stuck to the bottom" is a piece of state, it is set by *the reader's own
 * scrolling*, and nothing else may set it. Scrolling up unsticks; scrolling
 * back down within a threshold re-sticks, which is what makes it feel like a
 * transcript rather than a mode.
 *
 * **The threshold exists because "at the bottom" is never exact.** Sub-pixel
 * layout, a fractional device pixel ratio and a partly-visible last line all
 * mean `scrollTop + clientHeight` lands a pixel or two short of `scrollHeight`
 * while looking flush. A zero-tolerance check would unstick the reader who has
 * not moved at all, which is the same bug in the other direction.
 */

import { useCallback, useEffect, useRef } from "react";

/** How close to the bottom still counts as being at it. */
export const STICK_THRESHOLD_PX = 64;

/**
 * The element that actually scrolls around `node`.
 *
 * Usually the document, but not always: the shell gives `<main>` its own
 * `overflow-y: auto` while the split panel is open, so the transcript's
 * scroller changes depending on whether a second pane is showing. Walking up
 * from the node answers that without anything having to be told.
 */
export function scrollableAncestor(node: HTMLElement | null): HTMLElement | null {
  let current: HTMLElement | null = node?.parentElement ?? null;
  while (current) {
    const overflow = getComputedStyle(current).overflowY;
    if (
      (overflow === "auto" || overflow === "scroll") &&
      current.scrollHeight > current.clientHeight
    ) {
      return current;
    }
    current = current.parentElement;
  }
  // The page itself. `scrollingElement` is `<html>` in standards mode and is
  // the thing whose `scrollTop` actually moves.
  return (document.scrollingElement as HTMLElement | null) ?? document.documentElement;
}

/** Whether `element` is scrolled to within `threshold` of its bottom. */
export function isNearBottom(
  element: Pick<HTMLElement, "scrollTop" | "scrollHeight" | "clientHeight">,
  threshold = STICK_THRESHOLD_PX,
): boolean {
  const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
  return distance <= threshold;
}

/**
 * Keep `anchor` in view as content grows — but only while the reader is there.
 *
 * Returns nothing and takes a dependency list, so it reads at the call site
 * like the `useEffect` it replaces. The stickiness lives in a ref rather than
 * state on purpose: it changes on every scroll event, and re-rendering a long
 * transcript on every frame of a scroll is exactly the cost this milestone is
 * supposed to avoid.
 *
 * **The scroll listener is what sets the flag, and it is the only thing that
 * does.** A programmatic `scrollIntoView` also fires `scroll`, but by the time
 * it does the element is at the bottom, so it re-affirms `true` rather than
 * fighting the reader — the flag only ever flips to `false` because somebody
 * moved away from the bottom themselves.
 */
export function useStickToBottom(
  anchor: React.RefObject<HTMLElement | null>,
  deps: unknown[],
): void {
  /* Starts stuck: a transcript that has just loaded is at the bottom, and the
     first message must not require a scroll to see. */
  const stuck = useRef(true);

  useEffect(() => {
    /* **Listening on `document` in the capture phase, not on the scroller.**
       Scroll events do not bubble, so a listener on an ancestor never sees
       them — but they do propagate downward through capture, so `document`
       catches scrolling from any element on the page.

       That matters here for a concrete reason: which element scrolls the
       transcript is not fixed. The shell gives `<main>` its own `overflow-y`
       only while the split panel is open, so binding to whatever scrolled at
       mount would bind to the wrong element for the rest of the session the
       moment somebody pressed ⌘\. Resolving per event costs a handful of
       `getComputedStyle` calls on a shallow tree, in a passive handler. */
    function onScroll() {
      const scroller = scrollableAncestor(anchor.current);
      if (!scroller) return;

      const isPage =
        scroller === document.scrollingElement || scroller === document.documentElement;

      stuck.current = isNearBottom(
        isPage
          ? {
              scrollTop: window.scrollY,
              scrollHeight: document.documentElement.scrollHeight,
              clientHeight: window.innerHeight,
            }
          : scroller,
      );
    }

    document.addEventListener("scroll", onScroll, { passive: true, capture: true });
    return () =>
      document.removeEventListener("scroll", onScroll, { capture: true });
  }, [anchor]);

  const follow = useCallback(() => {
    if (!stuck.current) return;
    // Optional-called, because jsdom has no layout and therefore no
    // `scrollIntoView`. A missing scroll must never be the reason a test about
    // what was stored fails.
    anchor.current?.scrollIntoView?.({ block: "end" });
  }, [anchor]);

  useEffect(() => {
    follow();
    // The dependency list is the caller's — this hook exists to wrap exactly
    // the effect it replaces.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
