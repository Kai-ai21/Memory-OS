/**
 * The machinery both particle fields are made of: a sprite, an envelope, and a
 * blitter.
 *
 * **In `lib` because there are now three callers**, not two. The application
 * background draws an ambient drift and a cursor trail; the landing page draws
 * its own ambient drift and, as of M9.5, the same cursor trail. Nothing in this
 * file is a *look* — every number that decides how something reads lives with
 * the layer that reads that way.
 */

import { readInk } from "./tokens";

/** Anything with a position, a size and an opacity. Both layers draw as these. */
export interface Drawable {
  x: number;
  y: number;
  /** Radius in CSS pixels. */
  radius: number;
  /** Composited opacity for this frame, in `0..1`. */
  alpha: number;
}

/**
 * A short fade in and a long, eased fade out, over `0..1`.
 *
 * Both ends matter: a particle that appears at full strength pops, and one that
 * vanishes at full strength is a dot being deleted rather than smoke
 * dispersing. The exponent puts most of the life in the faded tail, which is
 * what makes it read as dissipation instead of a dimmer switch.
 */
export function envelope(t: number, rise = 0.12, fallPower = 1.6): number {
  if (t <= 0 || t >= 1) return 0;
  return Math.min(t / rise, 1) * Math.pow(1 - t, fallPower);
}

/** Injectable so a test can make emission deterministic. */
export type Random = () => number;

export function between(random: Random, low: number, high: number): number {
  return low + random() * (high - low);
}

/**
 * The one soft dot every particle in both layers is a scaled copy of.
 *
 * Built once and blitted, rather than a `createRadialGradient` per particle per
 * frame — at a thousand particles and 60fps that would be sixty thousand
 * gradient objects a second for an effect whose entire claim is that it is
 * cheap. The three stops are what make the edge soft: a hard-edged `arc` at
 * these opacities reads as a dead pixel rather than as smoke.
 */
export function buildSprite(size = 64): HTMLCanvasElement | null {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) return null;

  const ink = readInk();
  const half = size / 2;
  const gradient = context.createRadialGradient(half, half, 0, half, half, half);
  gradient.addColorStop(0, `rgba(${ink}, 1)`);
  gradient.addColorStop(0.45, `rgba(${ink}, 0.32)`);
  gradient.addColorStop(1, `rgba(${ink}, 0)`);
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);
  return canvas;
}

/** Blit one batch of particles. Both layers hand their own array to this. */
export function drawAll(
  context: CanvasRenderingContext2D,
  sprite: CanvasImageSource,
  particles: readonly Drawable[],
): void {
  for (const particle of particles) {
    if (particle.alpha <= 0.002) continue;
    context.globalAlpha = particle.alpha;
    context.drawImage(
      sprite,
      particle.x - particle.radius,
      particle.y - particle.radius,
      particle.radius * 2,
      particle.radius * 2,
    );
  }
  context.globalAlpha = 1;
}
