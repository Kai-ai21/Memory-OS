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
 * **M9.3 adds a second layer over them, on the same rule.** Soft dark specks
 * that drift off the cursor and dissipate. The intent is atmosphere, so the
 * measure of success is that you cannot count them: peak opacity 0.06–0.12,
 * radial-gradient edges, one particle per 40px of travel rather than one per
 * frame. Every one of those numbers is low enough to feel like a mistake until
 * you see the alternative, which on a light ground is soot. The arithmetic is
 * in `field.ts` and the loop is in `ParticleCanvas.tsx`.
 *
 * **Three gates, and only one of them is a preference.** Reduced motion and a
 * touch-primary pointer both mean the canvas is never created — see
 * `preference.ts`. The stored preference is the third, and it is the only one
 * the toggle can change.
 *
 * **Kept deliberately isolated.** Everything about how this is drawn lives
 * inside this one directory — no other file imports from it, references its
 * internals, or assumes it renders divs or a canvas. The shell's only contract
 * with it is "render something fixed and behind, and do not take the pointer".
 */

import { useEffect, useState } from "react";

import { ParticleCanvas } from "./ParticleCanvas";
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
