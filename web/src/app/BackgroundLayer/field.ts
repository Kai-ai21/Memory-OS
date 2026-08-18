/**
 * What the two layers share: a colour, a sprite, and an envelope.
 *
 * **Deliberately small.** M9.3's second step split the effect into an ambient
 * drift and a cursor trail specifically so each could be re-tuned without
 * touching the other, and that split is only real if the tunable numbers live
 * in `ambient.ts` and `cursor.ts` rather than here. Nothing in this file is a
 * look; everything in it is machinery both looks are made of.
 */

/**
 * The particle colour, read off `--color-ink` at sprite time.
 *
 * **Ink, not black.** Pure black on a warm white ground reads as a hole punched
 * in the paper rather than as smoke over it; the ink token is the same darkness
 * the type is, so the field looks like it belongs to the same document.
 *
 * Read from the stylesheet rather than written here, because a canvas cannot
 * take a `var()` and a copied literal is a colour that stays behind when the
 * palette is revised — which this palette has now been twice. The fallback is
 * the token's current value, kept in channel form so it is one string to
 * update and not a second definition of the colour.
 */
export const INK_FALLBACK = "15, 23, 42";

export function readInk(): string {
  if (typeof window === "undefined" || typeof getComputedStyle !== "function") {
    return INK_FALLBACK;
  }
  const declared = getComputedStyle(document.documentElement)
    .getPropertyValue("--color-ink")
    .trim();

  // Hex in the token file today, but a browser may hand back `rgb(...)` and a
  // future revision may declare one; both are cheap to accept and the cost of
  // not accepting them is a silently wrong colour.
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(declared);
  if (hex) {
    const digits =
      hex[1].length === 3
        ? hex[1].split("").map((digit) => digit + digit)
        : [hex[1].slice(0, 2), hex[1].slice(2, 4), hex[1].slice(4, 6)];
    return digits.map((pair) => parseInt(pair, 16)).join(", ");
  }

  const channels = declared.match(/\d+(\.\d+)?/g);
  if (channels && channels.length >= 3) return channels.slice(0, 3).join(", ");

  return INK_FALLBACK;
}

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
