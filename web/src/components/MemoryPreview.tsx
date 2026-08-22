/**
 * What a memory reference looks like before you commit to opening it.
 *
 * Wraps any reference — a citation, a search result path, a timeline entry, a
 * graph node — and shows a small card after a delay: the path, the kind, and
 * the first few lines. The question it answers is "is this the one I want",
 * which otherwise costs a navigation and a navigation back.
 *
 * **400ms, and the number is the feature.** Instant previews fire constantly
 * as the cursor crosses a list on its way somewhere else, so the screen flickers
 * with cards nobody asked for and the feature reads as broken. Past about 600ms
 * the card arrives after you have already decided to click, so it reads as
 * broken in the other direction. 400 is long enough to mean "I stopped here"
 * and short enough to still be an answer.
 *
 * **Dismissal has no delay at all**, which is deliberately asymmetric. A card
 * that lingers after the cursor has left is a card covering the thing you moved
 * to, and the usual reason to delay a hide — letting the pointer travel into
 * the card itself — does not apply here: this is a preview, not a menu, and
 * there is nothing in it to click.
 *
 * **The fetch is the same query the memory page uses.** `["memory", id]` with
 * React Query, so previewing a memory and then opening it costs one request,
 * and previewing the same row twice costs none. It does not start until the
 * delay has elapsed — the whole point of the delay is to avoid work for
 * references the cursor merely passed over.
 *
 * Focus opens it too. Every reference this wraps is a link or a button, so a
 * keyboard walking a result list gets the same information a mouse does; that
 * is nearly free here and is the sort of thing that never gets added later.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { isCode } from "../lib/format";

/** How long the pointer must rest before anything happens. See the header. */
export const PREVIEW_DELAY_MS = 400;

/** How much of the content the card shows. */
const EXCERPT = 240;

export function MemoryPreview({
  memoryId,
  children,
  className = "",
}: {
  memoryId: string;
  children: React.ReactNode;
  className?: string;
}) {
  const [shown, setShown] = useState(false);
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // A reference that unmounts inside the delay window — a result list
  // re-rendering under a new query is the common case — must not fire its
  // timer into a component that is gone.
  useEffect(() => () => clearTimeout(timer.current), []);

  const arm = useCallback((element: HTMLElement) => {
    clearTimeout(timer.current);
    // Measured when the timer is armed rather than when it fires: by the time
    // it fires the list may have scrolled, and a card pinned to where the row
    // used to be is worse than no card.
    const box = element.getBoundingClientRect();
    timer.current = setTimeout(() => {
      setAnchor(box);
      setShown(true);
    }, PREVIEW_DELAY_MS);
  }, []);

  const disarm = useCallback(() => {
    clearTimeout(timer.current);
    setShown(false);
  }, []);

  return (
    <span
      className={`relative inline-flex ${className}`}
      data-testid="memory-preview-anchor"
      onMouseEnter={(event) => arm(event.currentTarget)}
      onMouseLeave={disarm}
      onFocus={(event) => arm(event.currentTarget)}
      onBlur={disarm}
    >
      {children}
      {shown ? <Card memoryId={memoryId} anchor={anchor} /> : null}
    </span>
  );
}

/**
 * The card itself, mounted only once the delay has elapsed.
 *
 * A separate component so that `useQuery` is not called until the preview is
 * actually wanted — a hook in the wrapper would subscribe every reference on
 * the page to a query the moment it rendered, which is the cost the delay
 * exists to avoid.
 *
 * `position: fixed` against the anchor's viewport rect rather than absolute
 * inside it: the references this wraps live in rows with `overflow: hidden`
 * and inside the split panel's own scroll region, and an absolutely positioned
 * card gets clipped by both.
 */
function Card({ memoryId, anchor }: { memoryId: string; anchor: DOMRect | null }) {
  const memory = useQuery({
    queryKey: ["memory", memoryId],
    queryFn: () => api.memory(memoryId),
    staleTime: 5 * 60_000,
  });

  if (!anchor) return null;

  /* Below the reference by default, above it when there is no room — a card
     that opens downward off the bottom of the window shows its top two lines
     and nothing else. 320px is the card's max height plus its margin. */
  const below = anchor.bottom + 320 < window.innerHeight;
  const style: React.CSSProperties = {
    position: "fixed",
    left: Math.max(8, Math.min(anchor.left, window.innerWidth - 380)),
    ...(below ? { top: anchor.bottom + 6 } : { bottom: window.innerHeight - anchor.top + 6 }),
  };

  return (
    <div
      role="tooltip"
      data-testid="memory-preview"
      style={style}
      /* `.popover`, not `.panel-raised` — see the note on the class. This
         genuinely floats over the page, which is the one condition rule 2 in
         `tokens.css` allows a shadow for. */
      className="panel popover pointer-events-none z-50 w-90 max-w-[92vw] p-3"
    >
      {memory.isPending ? (
        <div className="flex flex-col gap-1.5">
          <span className="skeleton h-3 w-2/3" />
          <span className="skeleton h-3 w-full" />
          <span className="skeleton h-3 w-4/5" />
        </div>
      ) : memory.isError ? (
        <p className="meta text-ink-3">could not load this memory</p>
      ) : (
        <>
          <p className="font-mono text-xs break-all text-ink">{memory.data.external_key}</p>
          {memory.data.title ? (
            <p className="meta mt-0.5 text-ink-2">{memory.data.title}</p>
          ) : null}
          <p
            className={`mt-2 line-clamp-6 text-xs leading-relaxed text-ink-2 ${
              isCode(memory.data.kind) ? "font-mono whitespace-pre-wrap" : "font-prose"
            }`}
          >
            {memory.data.content.slice(0, EXCERPT)}
            {memory.data.content.length > EXCERPT ? "…" : ""}
          </p>
        </>
      )}
    </div>
  );
}
