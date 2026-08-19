/**
 * The wash behind everything, and the field that drifts over it.
 *
 * **The gradients exist for one element.** The NEW CONVERSATION button is
 * frosted glass, and frosted glass over a flat colour is a flat colour — the
 * blur has nothing to blur, and the button collapses into a white rectangle
 * with a white border. These two soft radials are what it frosts. Take them out
 * and the only piece of glass in the application stops reading as glass.
 *
 * That is also the whole of their job. They are not a background *treatment*:
 * pale blue and pale violet at under 10% opacity, which is enough for the
 * frosting to have something to pick up and not enough to turn a light theme
 * into a gradient. If you can see it without looking for it, it is too strong.
 *
 * **M9.3 draws two more layers over them, and they are not the same effect.**
 * They share one canvas and one frame loop and nothing else, which is the
 * point: either can be re-tuned without the other moving.
 *
 * `ambient.ts` is the weather — eight hundred half-pixel-to-two-pixel specks
 * riding a Perlin flow field, peak 0.08 each, with no idea the cursor exists.
 * It is running before you touch the mouse and it does not stop.
 *
 * `components/BrushLayer` is the gesture, and as of M9.6 it is not particles
 * at all. Emitting discs and hoping they merge was an approximation of a
 * stroke, and the approximation was what you saw; it keeps the last 1.5s of
 * cursor positions and strokes one tapered, blurred path through them.
 *
 * `ParticleCanvas.tsx` owns the drift's loop and the thing that makes drift
 * read as weather: the buffer is faded rather than cleared, so a speck overlaps
 * its own recent past. Read the header there before changing the clear — it is
 * done by erasing alpha rather than by painting the page colour over the top,
 * and the reason is the wash directly underneath it.
 *
 * **The stroke is dark enough to be unreadable-over, and that is handled by
 * position rather than by opacity.** The reading column declares itself a
 * shelter and the stroke is multiplied by zero inside it — see `lib/mask`.
 *
 * **Three gates, and only one of them is a preference.** Reduced motion and a
 * touch-primary pointer both mean the canvas is never created — see
 * `preference.ts`. The stored preference is the third, and it is the only one
 * the toggle can change. All three matter more since the ambient layer landed:
 * it never stops on its own, so there is no moment at which opting out is
 * merely cosmetic.
 *
 * **Kept deliberately isolated.** Everything about how this is drawn lives
 * inside this one directory — no other file imports from it, references its
 * internals, or assumes it renders divs or a canvas. The shell's only contract
 * with it is "render something fixed and behind, and do not take the pointer".
 */

import { useEffect, useState } from "react";

import { BrushLayer } from "../../components/BrushLayer";
import { APP_SHELTER_FEATHER, ParticleCanvas } from "./ParticleCanvas";
import { ParticleToggle } from "./ParticleToggle";
import { particlesPermitted, readPreference, writePreference } from "./preference";

export function BackgroundLayer() {
  /**
   * Resolved in an effect rather than in the initial state.
   *
   * Both gates read `matchMedia` and `localStorage`, neither of which exists
   * during a server render or during the first pass of a hydration — and a
   * first render that guessed "on" would create a canvas for a fraction of a
   * second on exactly the machines that asked not to have one.
   */
  const [permitted, setPermitted] = useState(false);
  const [on, setOn] = useState(false);

  useEffect(() => {
    const allowed = particlesPermitted();
    setPermitted(allowed);
    setOn(allowed && readPreference());
  }, []);

  return (
    <>
      <div
        /* Fixed rather than absolute, so the light stays where it is while the
           page scrolls under it — a wash that scrolls with the content reads as
           a coloured shape *in* the document, which is exactly what it must not
           look like.

           `overflow-hidden` is what stops the two oversized circles,
           deliberately positioned off their corners, from giving the document a
           horizontal scrollbar. `pointer-events-none` and `aria-hidden` because
           this is two divs of pure light: neither the mouse nor a screen reader
           should ever find them. */
        className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
        aria-hidden
        data-testid="background-layer"
      >
        <div
          className="absolute -top-[15%] -left-[10%] size-[55vw] animate-[drift_24s_ease-in-out_infinite_alternate] rounded-full blur-[90px]"
          style={{ backgroundImage: "var(--wash-blue)" }}
        />
        <div
          className="absolute -right-[10%] -bottom-[20%] size-[50vw] animate-[drift_24s_ease-in-out_-12s_infinite_alternate] rounded-full blur-[90px]"
          style={{ backgroundImage: "var(--wash-violet)" }}
        />
      </div>

      {/* Mounted as a sibling of the wash rather than a child of it: the wash
          clips its own overflow, and a full-viewport canvas inside a clipped
          box is a canvas with its edges cut off. */}
      {on ? <ParticleCanvas /> : null}

      {/* The cursor mark, on its own canvas above the drift and still below
          every piece of content. Separate because the two composite in
          opposite ways — the drift accumulates onto a faded buffer, the stroke
          is cleared and redrawn from a path every frame — and because the blur
          belongs to the stroke alone. */}
      {on ? <BrushLayer feather={APP_SHELTER_FEATHER} className="-z-10" /> : null}

      {permitted ? (
        <ParticleToggle
          on={on}
          onChange={(next) => {
            setOn(next);
            writePreference(next);
          }}
        />
      ) : null}
    </>
  );
}
