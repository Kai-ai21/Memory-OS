/**
 * The cursor trail, shared by the application background and the landing page.
 *
 * **One implementation because it is one effect.** M9.5 asked for the trail on
 * both surfaces with identical tuning; two copies of these numbers would have
 * been two places for them to drift, and the drift would be invisible until
 * somebody noticed the landing page felt different from the app and could not
 * say why. What each surface still owns is its *field* and its *shelter* —
 * both are injected, because the flow behind the app is tuned for a wash and
 * the one behind the landing page is tuned to be the page.
 *
 * **Density before darkness.** A trail that does not read as a stroke is almost
 * always emitting too sparsely rather than too faintly: at five pixels between
 * particles with radii of three to nine, neighbouring blobs overlap by most of
 * their area and merge into a continuous smear. Turning opacity up instead
 * gives a line of countable dots, which reads as dirt.
 *
 * **A fast sweep looks different from a slow one**, and that is the momentum:
 * each particle is born with a fraction of the pointer's own velocity and then
 * bleeds it off, so a flick throws its particles a long way down the stroke and
 * a crawl leaves them where they fell.
 */

import { between, drawAll, envelope, type Drawable, type Random } from "./particles";

/** One particle per this much cursor travel, in CSS pixels. */
export const EMIT_DISTANCE = 5;

/** Beyond this, the oldest is dropped to make room. */
export const MAX_PARTICLES = 300;

/**
 * Peak opacity of one particle, before the trail buffer composites several of
 * them on top of each other.
 *
 * This is the loudest number in the whole interface and it is deliberate: the
 * trail is the thing a person is meant to notice. What makes it survivable is
 * not that it is quiet but that it is *positional* — see `mask.ts`. Over a
 * reading column this is multiplied by zero.
 */
export const PEAK_ALPHA = 0.55;
export const MIN_ALPHA = 0.34;

/** Radius bounds, in CSS pixels. */
export const SIZE_MIN = 3;
export const SIZE_MAX = 9;

/** Lifetime bounds, in milliseconds. A sweep should still be dissolving at ~2s. */
export const LIFE_MIN = 1600;
export const LIFE_MAX = 2000;

/** How much of the pointer's own velocity a new particle is born with. */
const INHERIT = 0.32;

/** Momentum half-life, in milliseconds. Higher throws a longer tail. */
const MOMENTUM_TAU = 230;

/** Drift speed once the flow field has taken over, in px/ms. */
const FLOW_SPEED = 0.022;

/** How fast a particle is handed to the field, per millisecond. */
const FLOW_RATE = 0.006;

/** Ignore a jump this large: a tab switch is not cursor travel. */
const TELEPORT = 400;

/** The flow direction at a point and a time, in radians. Injected per surface. */
export type FlowField = (x: number, y: number, timeMs: number) => number;

/** How much ink is permitted at a point, in `0..1`. Injected per surface. */
export type Mask = (x: number, y: number) => number;

interface Trail extends Drawable {
  vx: number;
  vy: number;
  age: number;
  life: number;
  peak: number;
  born: number;
  spread: number;
}

export class CursorLayer {
  readonly particles: Trail[] = [];

  private pending = 0;
  private last: { x: number; y: number; at: number } | null = null;
  private readonly random: Random;
  private readonly flow: FlowField;
  private mask: Mask = () => 1;

  constructor(flow: FlowField, random: Random = Math.random) {
    this.flow = flow;
    this.random = random;
  }

  /** Replace the shelter mask. Cheap, and the caller re-measures as layout moves. */
  setMask(mask: Mask): void {
    this.mask = mask;
  }

  get active(): boolean {
    return this.particles.length > 0;
  }

  /**
   * Record a pointer position, emitting along the segment since the last one.
   *
   * Emission walks the segment rather than dropping everything at the new
   * point: at 60fps a fast sweep covers eighty pixels between two events, and
   * sixteen particles stacked at the far end of that is a clump where the whole
   * point was a continuous stroke.
   */
  push(x: number, y: number, at: number): void {
    const previous = this.last;
    this.last = { x, y, at };
    if (!previous) return;

    const dx = x - previous.x;
    const dy = y - previous.y;
    const distance = Math.hypot(dx, dy);
    if (distance === 0 || distance > TELEPORT) return;

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
    // Scatter across the stroke, so it has a soft edge rather than being a
    // ruled line one sprite wide.
    const offset = between(random, -3, 3);
    const inherited = speed * INHERIT;
    const born = between(random, SIZE_MIN, SIZE_MAX);

    this.particles.push({
      x: x - uy * offset,
      y: y + ux * offset,
      vx: ux * inherited + between(random, -0.012, 0.012),
      vy: uy * inherited + between(random, -0.012, 0.012),
      radius: born,
      born,
      spread: between(random, 0.5, 1.1),
      age: 0,
      life: between(random, LIFE_MIN, LIFE_MAX),
      peak: between(random, MIN_ALPHA, PEAK_ALPHA),
      alpha: 0,
    });

    // Oldest first: it is the one already closest to invisible.
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

      particle.vx *= drag;
      particle.vy *= drag;

      // The flow field takes over as the particle fades, so the trail dissolves
      // into the drift rather than switching off. Squared, so the handover
      // happens in the tail and does not erase the thrown-tail effect.
      const handover = t * t * (1 - Math.exp(-FLOW_RATE * dt));
      const angle = this.flow(particle.x, particle.y, now);
      particle.vx += (Math.cos(angle) * FLOW_SPEED - particle.vx) * handover;
      particle.vy += (Math.sin(angle) * FLOW_SPEED - particle.vy) * handover;

      particle.x += particle.vx * dt;
      particle.y += particle.vy * dt;
      particle.radius = particle.born * (1 + t * particle.spread);

      // The shelter is applied here rather than at emission, so a particle that
      // drifts over a reading column dims as it arrives instead of carrying the
      // brightness it was born with into the text.
      particle.alpha =
        particle.peak * envelope(t, 0.05, 1.3) * this.mask(particle.x, particle.y);
    }
  }

  draw(context: CanvasRenderingContext2D, sprite: CanvasImageSource): void {
    drawAll(context, sprite, this.particles);
  }
}
