/**
 * Pin a memory to the top of the sidebar.
 *
 * Visible at all times when pinned, and on hover of its row otherwise — the
 * same reveal as the copy and split controls. A pinned thing has to advertise
 * itself: the whole value is knowing at a glance that this is one of the four
 * you keep, and a state you can only see by hovering is not a state.
 *
 * `aria-pressed` rather than a label that changes, so the control is one thing
 * in two states rather than two controls that swap places.
 */

import { Pin as PinIcon } from "lucide-react";

import { indexOfPin, restorePin, togglePin, usePins } from "../lib/pins";
import { useToast } from "../lib/toast";

export function PinButton({
  memoryId,
  label,
  onToggled,
}: {
  memoryId: string;
  /** The path. Stored with the pin so the sidebar needs no fetch to draw it. */
  label: string;
  /** Told what the new state is, so a toast can say which way it went. */
  onToggled?: (pinned: boolean) => void;
}) {
  const pins = usePins();
  const pinned = pins.some((pin) => pin.id === memoryId);
  const toast = useToast();

  return (
    <button
      type="button"
      data-testid="pin"
      aria-pressed={pinned}
      aria-label={pinned ? `Unpin ${label}` : `Pin ${label}`}
      title={pinned ? "Unpin" : "Pin to the sidebar"}
      className={`inline-flex shrink-0 items-center justify-center rounded-sm p-0.5 transition-opacity duration-(--dur-state) ease-(--ease-out) group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 ${
        pinned ? "text-accent opacity-100" : "text-ink-3 opacity-0 hover:text-accent"
      }`}
      onClick={(event) => {
        // Nearly always inside a link or an expander.
        event.preventDefault();
        event.stopPropagation();
        /* Read the position *before* toggling. Undo has to put a pin back
           where it was, not on the front — a list that silently reorders
           itself when you take an action back has not undone anything. */
        const wasAt = indexOfPin(memoryId);
        const nowPinned = togglePin({ id: memoryId, label });

        /* Pinning moves something into a sidebar you may not be looking at,
           and unpinning removes it from one — neither has a visible result
           where the click happened, which is the whole test for whether an
           action deserves a toast.

           Only the removal gets an undo. Pinning is undone by clicking the
           same button again, which is right there and already means that;
           offering both would be two ways to do one thing. */
        toast.show(nowPinned ? "Pinned" : "Unpinned", {
          undo: nowPinned
            ? undefined
            : () => restorePin({ id: memoryId, label }, Math.max(0, wasAt)),
        });
        onToggled?.(nowPinned);
      }}
    >
      <PinIcon size={12} strokeWidth={2} fill={pinned ? "currentColor" : "none"} />
    </button>
  );
}
