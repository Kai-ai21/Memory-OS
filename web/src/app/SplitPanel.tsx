/**
 * The right-hand panel and the handle that sizes it.
 *
 * It renders `MemoryPage` — the same component the `/memory/:id` route renders,
 * handed an id instead of reading one from the URL. **Not a second, smaller
 * memory view.** A "compact" fork would have started as three fields and would
 * be six months behind the real one by the next milestone, and the thing you
 * open beside your work is exactly the thing you want in full: the chunk
 * boundaries, the provenance, the hashes.
 *
 * **The divider is a `separator` with arrow keys, not just a drag target.**
 * A handle that can only be dragged is a control a keyboard cannot operate, and
 * this application is driven from the keyboard. `role="separator"` with
 * `aria-valuenow` is what a screen reader needs to say what it is and where it
 * is; left and right move it in 2% steps, Home and End take it to its limits.
 *
 * Pointer events rather than mouse events, so a trackpad, a mouse and a stylus
 * are one code path. `setPointerCapture` is what keeps the drag alive when the
 * cursor outruns the 5px handle, which it does immediately on any real drag —
 * without it the divider drops the moment you move fast.
 */

import { useCallback, useRef } from "react";
import { X } from "lucide-react";

import { MemoryPage } from "../features/memory/MemoryPage";
import { useSplit } from "../lib/split";

/** How far one arrow-key press moves the divider, in percent. */
const STEP = 2;

export function SplitPanel() {
  const { memoryId, width, close, setWidth } = useSplit();
  const shell = useRef<HTMLDivElement>(null);

  /* The panel's own width, from a pointer position measured against the shell.
     Right-anchored: the divider is on the panel's left edge, so the panel is
     everything from the pointer to the right-hand side. */
  const widthFromPointer = useCallback(
    (clientX: number) => {
      const box = shell.current?.parentElement?.getBoundingClientRect();
      if (!box || box.width === 0) return null;
      return ((box.right - clientX) / box.width) * 100;
    },
    [],
  );

  if (!memoryId) return null;

  return (
    <div
      ref={shell}
      className="flex min-w-0 shrink-0 grow-0"
      style={{ width: `${width}%` }}
      data-testid="split-panel"
      data-split-width={width}
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize the split panel"
        aria-valuenow={Math.round(width)}
        aria-valuemin={22}
        aria-valuemax={68}
        tabIndex={0}
        data-testid="split-divider"
        /* 9px of target around a 1px rule. A 1px hit area is a hairline
           somebody has to aim at; the rule stays hairline-thin and the
           padding is what you actually grab. */
        className="group relative w-2.5 shrink-0 cursor-col-resize"
        onPointerDown={(event) => {
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
          const next = widthFromPointer(event.clientX);
          if (next !== null) setWidth(next);
        }}
        onPointerUp={(event) => event.currentTarget.releasePointerCapture(event.pointerId)}
        onKeyDown={(event) => {
          // Right shrinks the panel, left grows it — the divider moves the way
          // the key points, which is the opposite of the width changing.
          if (event.key === "ArrowLeft") setWidth(width + STEP);
          else if (event.key === "ArrowRight") setWidth(width - STEP);
          else if (event.key === "Home") setWidth(68);
          else if (event.key === "End") setWidth(22);
          else return;
          event.preventDefault();
        }}
      >
        <span
          aria-hidden
          className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-rule transition-colors duration-(--dur-state) ease-(--ease-out) group-hover:bg-accent group-focus-visible:bg-accent"
        />
      </div>

      <div className="panel my-3 mr-3 flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="flex shrink-0 items-center justify-between border-b border-rule px-3 py-2">
          <span className="meta-label text-ink-2">reading beside</span>
          <button
            type="button"
            className="icon-button"
            onClick={close}
            aria-label="Close the split panel"
            title="Close — esc"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>

        {/* Its own scroll region. The whole point is that the left side keeps
            its position while you read down the right. */}
        <div className="min-w-0 flex-1 overflow-y-auto px-4 py-4">
          <MemoryPage id={memoryId} />
        </div>
      </div>
    </div>
  );
}

/**
 * The control that opens a memory beside what you are doing.
 *
 * Small and quiet, and appears on hover of the row it belongs to like the copy
 * button does — a list of twenty results should not be a list of twenty split
 * icons. Rendered as a real button at all times so a keyboard can reach it;
 * see the note on `CopyButton`, which this follows exactly.
 */
export function SplitOpenButton({ memoryId, label }: { memoryId: string; label?: string }) {
  const { open, memoryId: current } = useSplit();
  const isOpen = current === memoryId;

  return (
    <button
      type="button"
      data-testid="split-open"
      className="inline-flex shrink-0 items-center justify-center rounded-sm p-0.5 text-ink-3 opacity-0 transition-opacity duration-(--dur-state) ease-(--ease-out) hover:text-accent group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100"
      aria-label={`Open ${label ?? "this memory"} beside this`}
      title="Open beside this — ⌘\\"
      aria-pressed={isOpen}
      onClick={(event) => {
        // Almost always inside a link or an expander. Opening the split must
        // not also navigate.
        event.preventDefault();
        event.stopPropagation();
        open(memoryId);
      }}
    >
      <SplitGlyph />
    </button>
  );
}

/** Two panes. Drawn here for the same reason the copy glyph is — see `Icon`. */
function SplitGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M13 4v16" />
    </svg>
  );
}
