/**
 * The wash behind everything.
 *
 * **This exists for one element.** The NEW CONVERSATION button is frosted
 * glass, and frosted glass over a flat colour is a flat colour — the blur has
 * nothing to blur, and the button collapses into a white rectangle with a white
 * border. These two soft radials are what it frosts. Take this component out
 * and the only piece of glass in the application stops reading as glass.
 *
 * That is also the whole of its job. It is not a background *treatment*: it is
 * pale blue and pale violet at under 10% opacity, which is enough for the
 * frosting to have something to pick up and not enough to turn a light theme
 * into a gradient. If you can see it without looking for it, it is too strong.
 *
 * **Kept deliberately isolated.** Everything about how this is drawn lives
 * inside this one component — no other file imports from it, references its
 * internals, or assumes it renders divs. A canvas, a shader, a cursor-following
 * field: any of those can replace the contents here without touching another
 * line anywhere else. The shell's only contract with it is "render something
 * fixed and behind, and do not take the pointer".
 */
export function BackgroundLayer() {
  return (
    <div
      /* Fixed rather than absolute, so the light stays where it is while the
         page scrolls under it — a wash that scrolls with the content reads as a
         coloured shape *in* the document, which is exactly what it must not
         look like.

         `overflow-hidden` is what stops the two oversized circles, deliberately
         positioned off their corners, from giving the document a horizontal
         scrollbar. `pointer-events-none` and `aria-hidden` because this is two
         divs of pure light: neither the mouse nor a screen reader should ever
         find them. */
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
  );
}
