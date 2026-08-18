/**
 * The ambient layer: eight hundred specks on a flow field, and no cursor in it.
 *
 * This is the weather. It is running before you touch the mouse and it carries
 * on after you stop, and nothing it does is a response to anything you did —
 * which is precisely why the cursor trail reads as *yours*. The two layers were
 * split so that this one could be made fainter or denser without touching the
 * trail's tuning, and every number in this file is one of those knobs.
 *
 * **Everything here is faint on purpose and that is not the same as invisible.**
 * A single speck at 0.08 is nothing; eight hundred of them, each smearing over
 * the trail buffer, is a texture you can see. The visible result is the sum, so
 * the per-particle number has to be much smaller than the one you want to read.
 *
 * **They respawn.** A flow field is full of attractors, and eight hundred
 * immortal particles find them within about ten seconds — after which the drift
 * is not a drift, it is a handful of permanent streaks with empty paper between
 * them. Each particle gets a few seconds of life and comes back somewhere else,
 * fading at both ends so the swap is never a pop.
 */

import { between, drawAll, envelope, type Drawable, type Random } from "./field";
import { flowAngle } from "./noise";

/** The whole layer, as a fixed-size pool. Never grows, never shrinks. */
export const AMBIENT_COUNT = 800;

/** The ceiling on one speck. The layer's *visible* darkness is well above it. */
export const AMBIENT_ALPHA_MAX = 0.08;
export const AMBIENT_ALPHA_MIN = 0.025;

/** Radius in CSS pixels: specks, not motes. */
export const AMBIENT_SIZE_MIN = 0.5;
export const AMBIENT_SIZE_MAX = 2;

/** Lifetime bounds in milliseconds, before a particle is respawned elsewhere. */
export const AMBIENT_LIFE_MIN = 6000;
export const AMBIENT_LIFE_MAX = 14000;

/** Drift speed in CSS pixels per millisecond — about 12 to 30 px a second. */
const SPEED_MIN = 0.012;
const SPEED_MAX = 0.03;

/**
 * How quickly a particle turns to face the field.
 *
 * Per millisecond, applied as an exponential approach. Turning instantly would
 * make every particle a perfect field line and the whole layer a contour map;
 * this lag is what gives it the slight overshoot that reads as drift.
 */
const TURN_RATE = 0.004;

interface Speck extends Drawable {
  vx: number;
  vy: number;
  speed: number;
  age: number;
  life: number;
  peak: number;
}

export class AmbientLayer {
  readonly particles: Speck[] = [];
  private readonly random: Random;
  private width = 0;
  private height = 0;

  constructor(random: Random = Math.random) {
    this.random = random;
  }

  /**
   * Size the field, filling it the first time and only reseeding positions.
   *
   * A resize must not empty the pool: dropping eight hundred particles because
   * somebody dragged a window edge would blank the layer and refill it over the
   * next ten seconds, which is far more visible than the resize was.
   */
  resize(width: number, height: number): void {
    const first = this.particles.length === 0;
    this.width = width;
    this.height = height;
    if (first) {
      for (let i = 0; i < AMBIENT_COUNT; i += 1) this.particles.push(this.spawn(true));
      return;
    }
    for (const speck of this.particles) {
      if (speck.x > width || speck.y > height) {
        speck.x = this.random() * width;
        speck.y = this.random() * height;
      }
    }
  }

  private spawn(stagger: boolean): Speck {
    const random = this.random;
    const speed = between(random, SPEED_MIN, SPEED_MAX);
    const angle = random() * Math.PI * 2;
    const life = between(random, AMBIENT_LIFE_MIN, AMBIENT_LIFE_MAX);
    return {
      x: random() * this.width,
      y: random() * this.height,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      speed,
      radius: between(random, AMBIENT_SIZE_MIN, AMBIENT_SIZE_MAX),
      // Staggered on the first fill only, so the pool does not respawn in one
      // wave every ten seconds for the life of the page.
      age: stagger ? random() * life : 0,
      life,
      peak: between(random, AMBIENT_ALPHA_MIN, AMBIENT_ALPHA_MAX),
      alpha: 0,
    };
  }

  private recycle(speck: Speck): void {
    Object.assign(speck, this.spawn(false));
  }

  /** Advance the whole pool by `dt` milliseconds at absolute time `now`. */
  step(dt: number, now: number): void {
    const turn = 1 - Math.exp(-TURN_RATE * dt);

    for (const speck of this.particles) {
      speck.age += dt;
      if (speck.age >= speck.life) {
        this.recycle(speck);
        continue;
      }

      const angle = flowAngle(speck.x, speck.y, now);
      // Approach the field direction rather than snapping to it.
      speck.vx += (Math.cos(angle) * speck.speed - speck.vx) * turn;
      speck.vy += (Math.sin(angle) * speck.speed - speck.vy) * turn;
      speck.x += speck.vx * dt;
      speck.y += speck.vy * dt;

      // Wrapped, not bounced. A bounce puts a visible hard edge on all four
      // sides of the screen; a wrap has no edge at all.
      if (speck.x < -4) speck.x += this.width + 8;
      else if (speck.x > this.width + 4) speck.x -= this.width + 8;
      if (speck.y < -4) speck.y += this.height + 8;
      else if (speck.y > this.height + 4) speck.y -= this.height + 8;

      // Flatter than the trail's envelope: this layer should sit at its own
      // level almost all the time and only taper at the very ends of a life.
      speck.alpha = speck.peak * envelope(speck.age / speck.life, 0.08, 0.35);
    }
  }

  draw(context: CanvasRenderingContext2D, sprite: CanvasImageSource): void {
    drawAll(context, sprite, this.particles);
  }
}
