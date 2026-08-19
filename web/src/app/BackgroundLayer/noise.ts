/**
 * The flow field the two background layers ride.
 *
 * The noise itself lives in `lib/noise` — it is generic machinery and the
 * landing page samples it too. What is here is the part that is a *look*: how
 * far apart two points have to be before the field points them differently,
 * and how fast the whole thing turns over. Both are tuned for a wash behind an
 * application, and neither should be shared with anything that wants a
 * different one.
 */

import { noise3 } from "../../lib/noise";

/**
 * How far apart two points have to be before the field points them differently.
 *
 * At 1/380 a full turn of the field is roughly a third of a screen, which is
 * the scale at which drift reads as weather rather than as either a vortex or a
 * parallel wind.
 */
const SPACE_SCALE = 1 / 380;

/** How fast the field itself turns over. One noise unit per twelve seconds. */
const TIME_SCALE = 1 / 12000;

/**
 * The flow direction at a point, in radians.
 *
 * Multiplied by 2π rather than π so the field can turn a particle all the way
 * back on itself; a half-turn field has no eddies in it, only bends.
 */
export function flowAngle(x: number, y: number, timeMs: number): number {
  return (
    noise3(x * SPACE_SCALE, y * SPACE_SCALE, timeMs * TIME_SCALE) * Math.PI * 2
  );
}
