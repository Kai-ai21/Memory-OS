/**
 * The shelter: where particles are not allowed to be dark.
 *
 * **This is what makes a heavy effect survivable.** A cursor trail at 0.55 ink
 * is deliberately unmistakable, and unmistakable ink behind a paragraph is a
 * paragraph you have to work to read. The wrong fix is to turn the whole effect
 * down until the worst case over text is acceptable — that pays for one region
 * with every other region, and the margins end up as timid as the text column.
 * The right fix is positional: full strength where there is nothing to read,
 * nothing at all where there is.
 *
 * A shelter is a rectangle plus a feather. Inside the rectangle the multiplier
 * is zero; it climbs to one over `feather` pixels measured outward from the
 * nearest edge, which makes the falloff a rounded rectangle rather than a
 * circle — text columns are rectangular, and a circular mask around a tall
 * column either leaves the top and bottom exposed or over-protects the sides.
 *
 * The ramp is smoothstep rather than linear. A linear ramp has a visible crease
 * at both ends: particles brighten at a constant rate and then stop, and the eye
 * finds the discontinuity in the derivative even though the value is continuous.
 */

/** How much ink is permitted at a point, in `0..1`. */
export type Mask = (x: number, y: number) => number;

export interface Shelter {
  left: number;
  top: number;
  right: number;
  bottom: number;
  /** Distance over which the multiplier climbs from 0 to 1, in CSS pixels. */
  feather: number;
}

/** Hermite ease, zero derivative at both ends. */
function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Build a multiplier function for a set of shelters.
 *
 * Overlapping shelters take the *minimum*, so a region protected by either one
 * stays protected. Returning a closure rather than a class because this is
 * called once per particle per frame — a couple of thousand times a frame — and
 * the shape of the hot loop is worth being deliberate about.
 */
export function makeMask(shelters: readonly Shelter[]): Mask {
  if (shelters.length === 0) return () => 1;

  return (x, y) => {
    let lowest = 1;
    for (const shelter of shelters) {
      // Distance from the point to the rectangle: zero inside, and the
      // perpendicular gap outside on each axis independently.
      const dx = Math.max(shelter.left - x, 0, x - shelter.right);
      const dy = Math.max(shelter.top - y, 0, y - shelter.bottom);
      const distance = Math.hypot(dx, dy);
      if (distance >= shelter.feather) continue;
      const value = smoothstep(distance / shelter.feather);
      if (value < lowest) lowest = value;
      if (lowest === 0) return 0;
    }
    return lowest;
  };
}

/** The attribute an element carries to say "keep the field off me". */
export const SHELTER_ATTRIBUTE = "data-particle-shelter";

/**
 * Measure every sheltered element currently in the document.
 *
 * Read from the DOM rather than configured, so the mask follows the layout
 * instead of duplicating it: a sticky composer that moves, a route that changes
 * the column width, a window that resizes — none of them need to tell anybody.
 * The caller re-runs this a few times a second, which is a `getBoundingClientRect`
 * on one or two elements and does not show up in a frame profile.
 */
export function readShelters(feather: number): Shelter[] {
  if (typeof document === "undefined") return [];
  return Array.from(document.querySelectorAll(`[${SHELTER_ATTRIBUTE}]`)).map((node) => {
    const rect = node.getBoundingClientRect();
    const own = node.getAttribute(SHELTER_ATTRIBUTE);
    const parsed = own ? Number(own) : Number.NaN;
    return {
      left: rect.left,
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      // An element may name its own feather; the attribute's value is the
      // distance in pixels, and anything unparseable falls back to the caller's.
      feather: Number.isFinite(parsed) && parsed > 0 ? parsed : feather,
    };
  });
}
