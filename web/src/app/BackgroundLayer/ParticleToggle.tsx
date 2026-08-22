/**
 * The switch, and the one piece of chrome this milestone adds.
 *
 * **This application has no settings surface, deliberately.** `Sidebar` argues
 * the case in its own header: everything configurable is an environment
 * variable, and a settings row would have had to lead to a page saying so. That
 * argument holds for weights and models. It does not hold for an ambient
 * animation, which is the one thing here a person may simply not want on their
 * screen — and "go and set reduced motion in the OS" is not an answer, because
 * that switch also stops the wash, the palette and the drawer.
 *
 * **Where it belongs is the sidebar, beside `help`, and it is not there.** That
 * is the pinned block this interface already keeps its one meta-control in, and
 * a row there would cost nothing and overlap nothing. This milestone was scoped
 * to `BackgroundLayer` alone, so the control had to live in this component's own
 * subtree — which means a fixed overlay, and a fixed overlay is over content by
 * construction. Moving it is a two-line change in `Sidebar.tsx` and everything
 * below is what makes the overlay survivable until somebody makes it.
 *
 * **So it is a dot.** Collapsed it is twelve pixels in the page's own right
 * gutter, outside the content column at every width the sidebar layout uses,
 * at 45% opacity and coloured only by state: an ink fill for on, a bare ring
 * for off. It becomes a labelled control on hover or keyboard focus and only
 * then — an expansion you asked for, over content you are pointing at, rather
 * than a permanent label sitting in the corner of every screenshot.
 *
 * It carries a real accessible name at both sizes, so what it is never depends
 * on the label being visible.
 *
 * **Portalled to the end of `body`, and that is not a detail.** `Shell` mounts
 * `BackgroundLayer` first, above even the skip link — correctly, since a wash
 * has to be behind everything. Rendering a button in that position would have
 * made a background-effect switch the first tab stop in the entire application,
 * ahead of "skip to content", which is the one thing the shell went out of its
 * way to put first. The portal moves it to the end of the document, where the
 * least important control on the page belongs, without moving the layer it is
 * declared in.
 */

import { createPortal } from "react-dom";

export function ParticleToggle({
  on,
  onChange,
}: {
  on: boolean;
  onChange: (next: boolean) => void;
}) {
  return createPortal(
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? "Background particles, on" : "Background particles, off"}
      title={
        on
          ? "Background particles are on — click to turn them off"
          : "Background particles are off — click to turn them on"
      }
      data-testid="particle-toggle"
      onClick={() => onChange(!on)}
      /* `pointer-events-auto` re-enables the pointer for this one element,
         inside a component that switches it off everywhere else. The wash and
         the canvas must stay untouchable; this is the only thing here that is
         not light.

         Anchored right, so the label expands leftward into the gutter rather
         than pushing the dot off the edge of the window. */
      className="group pointer-events-auto fixed right-1.5 bottom-1.5 z-20 flex items-center gap-1.5 rounded-full border border-rule-strong bg-surface p-[2px] opacity-45 transition-opacity duration-(--dur-state) ease-(--ease-out) hover:opacity-100 focus-visible:opacity-100"
    >
      <span className="meta-label hidden pl-2 whitespace-nowrap group-hover:inline group-focus:inline">
        {on ? "particles on" : "particles off"}
      </span>
      <span
        aria-hidden
        className={`size-3 rounded-full ${on ? "bg-ink-3" : "border border-rule-strong"}`}
      />
    </button>,
    document.body,
  );
}
