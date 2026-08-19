/**
 * What this directory's layers share — now mostly somewhere else.
 *
 * The sprite, the envelope and the blitter moved to `lib/particles` in M9.5,
 * when the landing page started drawing the same cursor trail this does. Kept
 * as a re-export rather than deleted: the tests in this directory assert on
 * their own module, and a test that has to reach into `lib` to check a
 * behaviour of `BackgroundLayer` is a test that has stopped describing it.
 */

import { INK_FALLBACK, readInk } from "../../lib/tokens";

export { INK_FALLBACK, readInk };
export {
  between,
  buildSprite,
  drawAll,
  envelope,
  type Drawable,
  type Random,
} from "../../lib/particles";
