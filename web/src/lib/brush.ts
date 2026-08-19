/**
 * The cursor mark, as a brush stroke rather than as particles.
 *
 * **This replaces an emitter, and the replacement is the point.** Every version
 * of the particle trail through M9.5 traded the same two failures against each
 * other: sparse enough to read as separate marks, or dense and dark enough to
 * read as dirt. Neither is a stroke, because a stroke is not a set of marks at
 * all — it is one mark whose width and darkness vary along its length. Emitting
 * two hundred discs and hoping they merge is an approximation of that, and the
 * approximation is exactly what you see.
 *
 * So: keep the recent path of the cursor and draw the path.
 *
 * **"A single path" and "the width tapers along it" cannot both be literal.**
 * `lineWidth` and `globalAlpha` are properties of a stroke *call*, not of a
 * point, so a canvas path has one width and one opacity for its whole length.
 * The stroke below is therefore drawn as a run of short overlapping segments,
 * each with its own width and alpha — which is what every implementation of a
 * tapered stroke does, and which reads as one continuous mark because the caps
 * and joins are round and the whole thing is blurred afterwards. The blur is
 * not decoration; it is what guarantees the seams are invisible.
 *
 * **Quadratic curves through midpoints.** Mouse samples arrive far apart on a
 * fast sweep — eighty pixels between frames is normal — and joining them with
 * `lineTo` puts a visible corner at every sample. Drawing instead from midpoint
 * to midpoint with the sample itself as the control point gives a curve that is
 * tangent-continuous everywhere, which is the difference between a brush mark
 * and a polyline.
 */

import type { Mask } from "./mask";
import { readInk } from "./tokens";

/**
 * The rolling window, in milliseconds.
 *
 * Also the fade: a point's opacity is a function of its age over this, so the
 * stroke is fully gone 1.5 seconds after the cursor stops. There is no separate
 * decay to keep in step with it.
 */
export const LIFETIME = 1500;

/**
 * Hard cap on retained points.
 *
 * A high-polling mouse reports at 1000Hz; without this, 1.5 seconds of it is
 * fifteen hundred points and fifteen hundred stroke calls per frame for a mark
 * that is sixty segments long to look at. At the cap the oldest is dropped,
 * which shortens the *tail* — the part already faded to nothing.
 */
export const MAX_POINTS = 60;

/** Width in CSS pixels: newest point to oldest. */
export const HEAD_WIDTH = 28;
export const TAIL_WIDTH = 4;

/** Opacity at the head. The tail is zero, by definition of the taper. */
export const HEAD_ALPHA = 0.85;

/** Gaussian radius applied to the whole layer. */
export const BLUR_RADIUS = 12;

/** A jump this large is a tab switch, not a gesture. Break rather than draw it. */
const TELEPORT = 400;

export interface Sample {
  x: number;
  y: number;
  /** Milliseconds, from the same clock `prune` and `draw` are given. */
  at: number;
}

export class BrushStroke {
  readonly points: Sample[] = [];

  /** True while there is anything left to draw. */
  get active(): boolean {
    return this.points.length > 1;
  }

  /**
   * Record a cursor position.
   *
   * Deliberately not de-duplicated by distance: a slow, deliberate movement
   * should leave a short dark stroke rather than nothing, and dropping samples
   * that are close together is how you lose it.
   */
  push(x: number, y: number, at: number): void {
    const last = this.points[this.points.length - 1];
    if (last && Math.hypot(x - last.x, y - last.y) > TELEPORT) {
      // The cursor did not travel from there to here, so there is no stroke
      // between them. Start again from the new position.
      this.points.length = 0;
    }
    this.points.push({ x, y, at });
    if (this.points.length > MAX_POINTS) {
      this.points.splice(0, this.points.length - MAX_POINTS);
    }
  }

  /** Drop everything older than the window. Cheap, and called every frame. */
  prune(now: number): void {
    const cutoff = now - LIFETIME;
    let keep = 0;
    while (keep < this.points.length && this.points[keep].at < cutoff) keep += 1;
    if (keep > 0) this.points.splice(0, keep);
  }

  /** Forget the path — used when the pointer leaves the window. */
  clear(): void {
    this.points.length = 0;
  }

  /**
   * Stroke the path.
   *
   * `now` is what the taper is measured against, so a stroke keeps fading while
   * the cursor is still: the newest point ages like every other one, and the
   * head thins and lightens rather than sitting at full strength until it
   * vanishes.
   */
  draw(context: CanvasRenderingContext2D, now: number, mask?: Mask): void {
    const points = this.points;
    if (points.length < 2) return;

    const ink = readInk();
    context.lineCap = "round";
    context.lineJoin = "round";

    for (let i = 1; i < points.length; i += 1) {
      const previous = points[i - 1];
      const point = points[i];

      // Age of this segment, as a fraction of the window: 0 at the head, 1 at
      // the tail. Taken from the *newer* of the two so the head stays the head.
      const age = Math.min(Math.max((now - point.at) / LIFETIME, 0), 1);
      const fresh = 1 - age;

      // Linear in both, which is the literal reading of "28 down to 4" and
      // "0.85 fading to 0" — and it is also the one that looks like a brush.
      // A squared falloff was tried first and puts effectively all the weight
      // in the last fifth of the path: the head is right and the other 80% is
      // a thin grey whisker, so a 1400px sweep reads as a 250px dash.
      const width = TAIL_WIDTH + (HEAD_WIDTH - TAIL_WIDTH) * fresh;
      let alpha = HEAD_ALPHA * fresh;
      if (mask) {
        // Sampled at the segment's midpoint rather than at either end, so a
        // stroke crossing into a reading column dims symmetrically instead of
        // one frame late.
        alpha *= mask((previous.x + point.x) / 2, (previous.y + point.y) / 2);
      }
      if (alpha <= 0.004) continue;

      context.lineWidth = width;
      context.strokeStyle = `rgba(${ink}, ${alpha})`;
      context.beginPath();

      if (i === 1) {
        context.moveTo(previous.x, previous.y);
      } else {
        const before = points[i - 2];
        context.moveTo((before.x + previous.x) / 2, (before.y + previous.y) / 2);
      }

      if (i === points.length - 1) {
        // The last segment runs to the cursor itself rather than to a midpoint,
        // so the head of the stroke is where the pointer actually is.
        context.quadraticCurveTo(previous.x, previous.y, point.x, point.y);
      } else {
        context.quadraticCurveTo(
          previous.x,
          previous.y,
          (previous.x + point.x) / 2,
          (previous.y + point.y) / 2,
        );
      }
      context.stroke();
    }
  }
}
