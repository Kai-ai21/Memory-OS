/**
 * The cursor layer: the one the eye is meant to follow.
 *
 * Everything the ambient layer is not. It is four times as dark, an order of
 * magnitude denser along its path, and it exists only where you have just moved
 * the pointer. Split from `ambient.ts` so this file can be made heavier or
 * lighter without the weather changing underneath it.
 *
 * **Density before darkness.** A trail that does not read as a stroke is almost
 * always emitting too sparsely rather than too faintly — eight pixels between
 * particles at a radius of two to six is what makes overlapping blobs merge
 * into one continuous smear. Turning the opacity up instead gives you a line of
 * countable dots, which is the failure this is tuned against.
 *
 * **Three things make a fast sweep look different from a slow one.** Emission
 * is per distance, so speed does not change the spacing. What changes is the
 * momentum each particle is born with: it inherits a fraction of the pointer's
 * own velocity and then bleeds it off, so a flick throws its particles a
 * hundred-odd pixels further along the sweep and the tail is visibly longer.
 *
 * **And they end up in the weather.** As a particle fades it is handed over to
 * the same flow field the ambient layer rides, so the last third of its life is
 * spent drifting rather than coasting. The trail dissolves into the background
 * instead of switching off.
 */

import { between, drawAll, envelope, type Drawable, type Random } from "./field";
import { flowAngle } from "./noise";

/** One particle per this much cursor travel, in CSS pixels. Not per frame. */
export const EMIT_DISTANCE = 8;

/** Beyond this, the oldest is dropped to make room. */
export const MAX_PARTICLES = 200;

/** Peak opacity of a single particle, before the trail buffer accumulates it. */
export const CURSOR_ALPHA_MAX = 0.35;
export const CURSOR_ALPHA_MIN = 0.22;

/** Radius in CSS pixels. */
export const CURSOR_SIZE_MIN = 2;
export const CURSOR_SIZE_MAX = 6;

/** Lifetime bounds, in milliseconds. */
export const LIFE_MIN = 1000;
export const LIFE_MAX = 1400;

/** How much of the pointer's own velocity a new particle is born with. */
const INHERIT = 0.32;

/** Momentum half-life, in milliseconds. Higher throws a longer tail. */
const MOMENTUM_TAU = 210;

/** Drift speed once the flow field has taken over, in px/ms. */
const FLOW_SPEED = 0.022;

/** How fast a particle is handed to the field, per millisecond. */
const FLOW_RATE = 0.006;

/** Ignore a jump this large: a tab switch is not cursor travel. */
const TELEPORT = 400;

interface Trail extends Drawable {
  vx: number;
  vy: number;
  age: number;
  life: number;
  peak: number;
  /** Radius at birth, and the fraction it grows by over a life. */
  born: number;
  spread: number;
}

export class CursorLayer {
  readonly particles: Trail[] = [];

  private pending = 0;
  private last: { x: number; y: number; at: number } | null = null;
  private readonly random: Random;

  constructor(random: Random = Math.random) {
    this.random = random;
  }

  /** True while there is anything to draw. Used by the loop's idle rule. */
  get active(): boolean {
    return this.particles.length > 0;
  }

  /**
   * Record a pointer position, emitting along the segment since the last one.
   *
   * Emission walks the segment rather than dropping everything at the new
   * point: at 60fps a fast sweep covers eighty pixels between two events, and
   * putting ten particles at the far end of that gives a row of clumps where
   * the whole point was a continuous stroke.
   */
  push(x: number, y: number, at: number): void {
    const previous = this.last;
    this.last = { x, y, at };
    if (!previous) return;

    const dx = x - previous.x;
    const dy = y - previous.y;
    const distance = Math.hypot(dx, dy);
    if (distance === 0 || distance > TELEPORT) return;

    // Pointer velocity over this segment, in px/ms, as a direction and a speed.
    const elapsed = Math.max(at - previous.at, 1);
    const speed = distance / elapsed;
    const ux = dx / distance;
    const uy = dy / distance;

    this.pending += distance;
    while (this.pending >= EMIT_DISTANCE) {
      this.pending -= EMIT_DISTANCE;
      const along = 1 - this.pending / distance;
      this.emit(previous.x + dx * along, previous.y + dy * along, ux, uy, speed);
    }
  }

  /** Forget where the pointer was, so the next move is not a segment from it. */
  breakTrail(): void {
    this.last = null;
    this.pending = 0;
  }

  emit(x: number, y: number, ux = 0, uy = 0, speed = 0): void {
    const random = this.random;
    // A little scatter across the stroke, so the trail has a soft edge rather
    // than being a ruled line one sprite wide.
    const offset = between(random, -2.5, 2.5);
    const inherited = speed * INHERIT;
    const born = between(random, CURSOR_SIZE_MIN, CURSOR_SIZE_MAX);

    this.particles.push({
      x: x - uy * offset,
      y: y + ux * offset,
      vx: ux * inherited + between(random, -0.012, 0.012),
      vy: uy * inherited + between(random, -0.012, 0.012),
      radius: born,
      born,
      // Ash spreads as it cools. The growth is not visible as growth; it is
      // visible as the back of the trail going soft.
      spread: between(random, 0.5, 1.1),
      age: 0,
      life: between(random, LIFE_MIN, LIFE_MAX),
      peak: between(random, CURSOR_ALPHA_MIN, CURSOR_ALPHA_MAX),
      alpha: 0,
    });

    // Oldest first: it is the one already closest to invisible, so dropping it
    // is the only choice a viewer cannot see.
    while (this.particles.length > MAX_PARTICLES) this.particles.shift();
  }

  step(dt: number, now: number): void {
    const drag = Math.exp(-dt / MOMENTUM_TAU);

    for (let i = this.particles.length - 1; i >= 0; i -= 1) {
      const particle = this.particles[i];
      particle.age += dt;
      if (particle.age >= particle.life) {
        this.particles.splice(i, 1);
        continue;
      }
      const t = particle.age / particle.life;

      // Inherited momentum bleeds off …
      particle.vx *= drag;
      particle.vy *= drag;

      // … and the flow field takes over, with a weight that rises over the
      // life. Squared, so the handover happens in the fading tail rather than
      // immediately, where it would erase the thrown-tail effect entirely.
      const handover = t * t * (1 - Math.exp(-FLOW_RATE * dt));
      const angle = flowAngle(particle.x, particle.y, now);
      particle.vx += (Math.cos(angle) * FLOW_SPEED - particle.vx) * handover;
      particle.vy += (Math.sin(angle) * FLOW_SPEED - particle.vy) * handover;

      particle.x += particle.vx * dt;
      particle.y += particle.vy * dt;
      particle.radius = particle.born * (1 + t * particle.spread);
      particle.alpha = particle.peak * envelope(t, 0.06, 1.4);
    }
  }

  draw(context: CanvasRenderingContext2D, sprite: CanvasImageSource): void {
    drawAll(context, sprite, this.particles);
  }
}
