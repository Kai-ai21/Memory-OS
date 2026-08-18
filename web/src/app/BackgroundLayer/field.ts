/**
 * The particle field: the arithmetic, with no canvas and no React in it.
 *
 * Kept as a plain class so the two properties that actually matter — the cap
 * holds, and the field goes empty — are testable without a DOM, a frame loop
 * or a rendering context. `ParticleField` knows how to be stepped and how to
 * be drawn onto a context it is handed; it never looks one up.
 *
 * **Every constant here is a restraint rather than a tuning knob.** On a light
 * ground, dark particles stop being atmosphere and become dirt very quickly.
 * The numbers below are the ones that survived being looked at: a peak opacity
 * you have to go looking for, a lifetime long enough that nothing pops, and an
 * emission rate tied to distance travelled rather than to frames elapsed — so
 * a still cursor emits nothing at all, and a fast one does not emit a stripe.
 */

/** Beyond this, the oldest particle is dropped to make room. */
export const MAX_PARTICLES = 150;

/** One particle per this many pixels of cursor travel. Not per frame. */
export const EMIT_DISTANCE = 40;

/** Lifetime bounds, in milliseconds. */
export const LIFE_MIN = 1500;
export const LIFE_MAX = 2500;

/**
 * Peak opacity bounds.
 *
 * The upper bound is the number to distrust, and **it is 0.09 rather than the
 * 0.12 this was built at, for a measured reason.** A particle is ink laid over
 * both the glyph and the paper, so it costs contrast; the specified ceiling was
 * checked against every text role in `tokens.css` and one of them does not
 * survive it. `accent` — links, and the only colour in the interface that means
 * "you can act on this" — measures 4.86:1 on the ground, clearing WCAG AA by
 * 0.36. An ink veil at 0.12 takes it to 4.44:1, and the crossing point is
 * α = 0.105. At 0.09 it holds at 4.55:1, and `ink-3`, the next tightest, at
 * 4.75:1.
 *
 * That is a third of a stop the effect gives up, and it is not visible as one:
 * a real sweep peaks around 0.078 anyway, because a particle only reaches its
 * own ceiling at the top of the envelope and only at its centre pixel. What the
 * old ceiling bought was a worst case that quietly undid a contrast contract
 * this theme has an argued test file for. `contrast.test.ts` checks the tokens;
 * `particles.test.tsx` checks them under this veil.
 *
 * Above these numbers it stops being weather. At 0.2 a particle is a smudge,
 * and at 0.3 you read the individual dots instead of the sentence under them.
 */
export const ALPHA_MIN = 0.06;
export const ALPHA_MAX = 0.09;

/** Radius bounds, in CSS pixels, before the lifetime's slow expansion. */
export const SIZE_MIN = 3;
export const SIZE_MAX = 12;

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

export interface Particle {
  x: number;
  y: number;
  /** Velocity in CSS pixels per millisecond. */
  vx: number;
  vy: number;
  radius: number;
  /** Age and lifetime, both in milliseconds. */
  age: number;
  life: number;
  /** Opacity at the peak of the envelope. */
  peak: number;
}

/**
 * The opacity envelope over a particle's life, in `0..1`.
 *
 * A short fade in and a long, eased fade out. Both ends matter: a particle
 * that appears at full strength pops, and one that vanishes at full strength
 * is a dot being deleted rather than smoke dispersing. The `1.6` exponent puts
 * most of the life in the faded tail, which is what makes it read as
 * dissipation instead of a dimmer switch.
 */
export function envelope(t: number): number {
  if (t <= 0 || t >= 1) return 0;
  const rise = Math.min(t / 0.12, 1);
  const fall = Math.pow(1 - t, 1.6);
  return rise * fall;
}

/** Injectable so a test can make emission deterministic. */
export type Random = () => number;

function between(random: Random, low: number, high: number): number {
  return low + random() * (high - low);
}

export class ParticleField {
  readonly particles: Particle[] = [];

  /** Cursor travel not yet spent on a particle. */
  private pending = 0;
  private last: { x: number; y: number } | null = null;
  private readonly random: Random;

  constructor(random: Random = Math.random) {
    this.random = random;
  }

  /**
   * Record a cursor position, emitting along the segment since the last one.
   *
   * Emission walks the segment rather than dropping everything at the new
   * point, so a fast flick across the screen leaves a dispersed line instead of
   * a clump at the far end — which is the difference between smoke and a
   * cursor trail.
   */
  push(x: number, y: number): void {
    const previous = this.last;
    this.last = { x, y };
    if (!previous) return;

    const dx = x - previous.x;
    const dy = y - previous.y;
    const distance = Math.hypot(dx, dy);
    // A teleport — a tab switch, a window move — is not travel, and treating it
    // as such would emit a full field's worth of particles in one frame.
    if (distance === 0 || distance > 400) return;

    this.pending += distance;
    while (this.pending >= EMIT_DISTANCE) {
      this.pending -= EMIT_DISTANCE;
      const at = 1 - this.pending / distance;
      this.emit(previous.x + dx * at, previous.y + dy * at);
    }
  }

  /** Forget where the cursor was, so the next move is not a segment from it. */
  breakTrail(): void {
    this.last = null;
    this.pending = 0;
  }

  emit(x: number, y: number): void {
    const random = this.random;
    this.particles.push({
      x,
      y,
      // Sideways drift is small and signed; without it the field rises in a
      // column and reads as a machine venting.
      vx: between(random, -0.018, 0.018),
      // Up, always, but by varying amounts — roughly 15 to 60px over a life.
      vy: -between(random, 0.008, 0.026),
      radius: between(random, SIZE_MIN, SIZE_MAX),
      age: 0,
      life: between(random, LIFE_MIN, LIFE_MAX),
      peak: between(random, ALPHA_MIN, ALPHA_MAX),
    });

    // The cap is a hard ceiling on both memory and per-frame cost. Oldest goes
    // first: it is the one already closest to invisible, so dropping it is the
    // only choice a viewer cannot see.
    while (this.particles.length > MAX_PARTICLES) this.particles.shift();
  }

  /** Advance by `dt` milliseconds and drop anything that has finished. */
  step(dt: number): void {
    for (let i = this.particles.length - 1; i >= 0; i -= 1) {
      const particle = this.particles[i];
      particle.age += dt;
      if (particle.age >= particle.life) {
        this.particles.splice(i, 1);
        continue;
      }
      particle.x += particle.vx * dt;
      particle.y += particle.vy * dt;
    }
  }

  draw(context: CanvasRenderingContext2D, sprite: CanvasImageSource): void {
    for (const particle of this.particles) {
      const t = particle.age / particle.life;
      const alpha = particle.peak * envelope(t);
      if (alpha <= 0.001) continue;
      // Ash spreads as it cools. 40% growth over a life, which is not visible
      // as growth — it is visible as the edge going soft.
      const radius = particle.radius * (1 + t * 0.4);
      context.globalAlpha = alpha;
      context.drawImage(
        sprite,
        particle.x - radius,
        particle.y - radius,
        radius * 2,
        radius * 2,
      );
    }
    context.globalAlpha = 1;
  }
}

/**
 * The one soft dot every particle is a scaled copy of.
 *
 * Built once and blitted, rather than a `createRadialGradient` per particle per
 * frame — at 150 particles and 60fps that would be nine thousand gradient
 * objects a second for an effect whose entire claim is that it is cheap. The
 * three stops are what make the edge soft: a hard-edged `arc` at this opacity
 * reads as a dead pixel rather than as smoke.
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
