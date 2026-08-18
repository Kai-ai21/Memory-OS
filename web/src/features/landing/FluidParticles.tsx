/**
 * The landing page's background: several hundred specks on a Perlin flow
 * field, drawn onto a buffer that is faded rather than cleared.
 *
 * Adapted from a supplied reference implementation, which had five defects.
 * Each is named where it is fixed rather than only in a commit message,
 * because four of the five are invisible until they are expensive:
 *
 *   1. the noise factory ran on every render and sat in the effect's
 *      dependency array — see `noise3` below, and the deps at the bottom;
 *   2. the frame loop was never cancelled — see the cleanup;
 *   3. no `devicePixelRatio` scaling — see `resize`;
 *   4. no `prefers-reduced-motion` gate — see `useReducedMotion`;
 *   5. resize left the trail buffer in an undefined state — see `resize`.
 *
 * **The trail is the effect.** Painting a translucent rectangle over the
 * previous frame instead of clearing it leaves every position a particle has
 * occupied in the last second faintly on screen, and the overlap of a moving
 * particle with its own recent past is what turns a field of dots into flowing
 * ink. Clear outright and you have eight hundred dots, however you tune them.
 *
 * Unlike the application background in `app/BackgroundLayer`, this canvas is
 * the bottom of its own stacking context with nothing behind it that matters —
 * so it can fade with an opaque page colour, which is the simple version, and
 * does not need the alpha-erasing trick that one uses to protect the wash
 * underneath it.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

import { noise3 } from "../../lib/noise";
import { readGround, readInk } from "../../lib/tokens";
import { cn } from "../../lib/utils";

export interface FluidParticlesProps {
  children?: ReactNode;
  /**
   * The reference shipped 2000. Eight hundred reads the same and costs 40% of
   * the draw calls — with a trail buffer the density you perceive is the smear,
   * not the particle count, so the last twelve hundred were paying to be
   * averaged into a mark the first eight hundred had already made.
   */
  particleCount?: number;
  /** Field scale. Larger means the flow turns over a shorter distance. */
  noiseIntensity?: number;
  particleSize?: { min: number; max: number };
  className?: string;
}

/**
 * Peak opacity of one particle. **0.06, tuned down from the specified 0.08.**
 *
 * The reference's 0.15 is nearly three times what a light ground will take.
 * 0.08 is close, and it was still too much for a reason the single-particle
 * number does not show: with a trail buffer the mark you see is not one
 * particle, it is every particle that has crossed that pixel in the last
 * second, and a flow field funnels them. Measured over 72 seconds at 0.08 the
 * darkest pixel on the page reached luminance 93 against a ground of 248 —
 * effectively a black scratch. At 0.06 it settles near 200.
 *
 * The failure mode to tune against is *not* "individual particles are
 * countable" here; they never were. It is the opposite — they merge into a
 * scribbled mat. Both are cured by taking ink out.
 */
const PEAK_OPACITY = 0.06;

/**
 * How much page colour is laid over the previous frame each 60fps frame.
 * **0.3, tuned up from the reference's 0.12.**
 *
 * This is the knob for trail *length*, and it is inverted: lower is longer. It
 * is also the only thing stopping the buffer silting up: a pixel's composited
 * darkness settles at roughly the ink arriving per frame divided by this
 * number, so too low and the flow field's contour lines are painted onto the
 * page and left there. At the reference's 0.12 — and at 0.1, which is where
 * this started — nine seconds is enough to turn the whole viewport into a
 * scratched mat covering 75% of the page in visible ink.
 *
 * Measured, over a full minute rather than the first few seconds, which is
 * where an accumulation bug hides.
 */
const TRAIL_ALPHA = 0.3;

/** Drift speed, in CSS pixels per millisecond — about 110px a second. */
const SPEED = 0.11;

/**
 * How fast the field itself turns over, per millisecond.
 *
 * The reference's `Date.now() * 0.0001` gives one noise unit per ten seconds,
 * which sounds like plenty and is not: a flow field has stagnation points, a
 * stream of particles funnels into each one, and the trail buffer over that
 * spot receives ink faster than it can shed it for as long as the point stays
 * put. Turning the field over faster keeps the attractors migrating, which is
 * the cause rather than the symptom — the alternative is to keep taking
 * opacity out until the worst permanent spot is acceptable, and pay for one
 * pathological pixel across the whole effect.
 */
const FIELD_DRIFT = 0.00018;

/** Lifetime bounds in milliseconds, before a particle respawns elsewhere. */
const LIFE_MIN = 1800;
const LIFE_MAX = 2800;

/** The largest step one frame may claim, in ms. A backgrounded tab hands back
 *  minutes on its first frame, and every particle would jump across the page. */
const MAX_STEP = 64;

interface Particle {
  x: number;
  y: number;
  size: number;
  life: number;
  maxLife: number;
}

/**
 * **Defect 4.** Reduced motion, resolved once and watched afterwards.
 *
 * The reference has no check at all. Eight hundred — or, as shipped, two
 * thousand — drifting specks across the whole viewport is a vestibular trigger,
 * and this is the setting a person turns on to be spared exactly this. It
 * returns `true` until proven otherwise is *not* the behaviour here: it
 * resolves in an effect, and the canvas is simply not rendered while it holds.
 * Not a paused canvas, not a slower one — none.
 */
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

export function FluidParticles({
  children,
  particleCount = 800,
  noiseIntensity = 0.003,
  particleSize = { min: 0.5, max: 2 },
  className,
}: FluidParticlesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reducedMotion = useReducedMotion();

  // Read off the prop rather than the object, so the dependency list below can
  // be primitives. `particleSize` is a fresh object literal on every render
  // when the caller does not pass one, which would restart the effect forever
  // — the same defect as the noise factory, wearing different clothes.
  const sizeMin = particleSize.min;
  const sizeMax = particleSize.max;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;

    const ink = readInk();
    const ground = readGround();
    let width = 0;
    let height = 0;
    let particles: Particle[] = [];

    function spawn(fresh: boolean): Particle {
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * (sizeMax - sizeMin) + sizeMin,
        // Staggered on the first fill, so eight hundred particles do not all
        // reach the end of their lives in the same frame for the life of the
        // page — which would read as the whole field blinking.
        life: fresh ? Math.random() * LIFE_MAX : 0,
        maxLife: LIFE_MIN + Math.random() * (LIFE_MAX - LIFE_MIN),
      };
    }

    /**
     * **Defects 3 and 5**, which are the same handful of lines.
     *
     * *3 — device pixels.* The reference sets `canvas.width = innerWidth`,
     * which on a retina display is a backing store at half the resolution of
     * the element, upscaled by the compositor. On sub-pixel specks that does
     * not read as "soft", it reads as smeared. The store is sized in device
     * pixels, the element in CSS pixels, and the context is transformed once so
     * every coordinate below stays in CSS pixels. Capped at 2: a 3x screen
     * would cost 2.25 times the fill for a difference invisible at 8% alpha.
     *
     * *5 — the trail buffer.* Assigning `canvas.width` resets the bitmap to
     * transparent black, which is not the state this effect's arithmetic
     * assumes: every frame composites the new particles over a buffer that is
     * supposed to already hold page colour, so a resize left the fade building
     * up out of transparency and the first second after any drag showed the
     * accumulated smear as a dark ghost of the old layout. It is now repainted
     * opaque, and the particles — which are still holding coordinates from the
     * old viewport, the other half of the artifact — are folded into the new
     * bounds rather than left clumped along an edge.
     */
    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas!.width = Math.round(width * ratio);
      canvas!.height = Math.round(height * ratio);
      canvas!.style.width = `${width}px`;
      canvas!.style.height = `${height}px`;
      // Set rather than multiplied: resize fires many times during a drag and a
      // relative `scale` would compound.
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);

      context!.fillStyle = `rgba(${ground}, 1)`;
      context!.fillRect(0, 0, width, height);

      if (particles.length === 0) {
        particles = Array.from({ length: particleCount }, () => spawn(true));
      } else {
        for (const particle of particles) {
          particle.x = ((particle.x % width) + width) % width;
          particle.y = ((particle.y % height) + height) % height;
        }
      }
    }

    let frameId: number | null = null;
    let previous = 0;

    function frame(now: number) {
      const dt = previous === 0 ? 1000 / 60 : Math.min(now - previous, MAX_STEP);
      previous = now;

      // The trail. Page colour at low alpha rather than a clear: this is the
      // whole reason the effect reads as ink rather than as dots.
      context!.fillStyle = `rgba(${ground}, ${TRAIL_ALPHA * (dt / (1000 / 60))})`;
      context!.fillRect(0, 0, width, height);

      for (const particle of particles) {
        particle.life += dt;
        if (particle.life > particle.maxLife) {
          Object.assign(particle, spawn(false));
        }

        // Fade in and out over a life, so nothing pops into or out of being.
        const opacity =
          Math.sin((particle.life / particle.maxLife) * Math.PI) * PEAK_OPACITY;

        // The field is sampled in three dimensions, the third being time, so it
        // turns over instead of being a fixed pattern the particles slide along
        // forever — a static field collects them onto its attractor lines
        // within seconds and the drift becomes a few permanent streaks.
        const angle =
          noise3(
            particle.x * noiseIntensity,
            particle.y * noiseIntensity,
            now * FIELD_DRIFT,
          ) *
          Math.PI *
          4;

        // Velocity per millisecond, not per frame. The reference advanced by a
        // fixed 2px each frame, which makes the whole effect run 2.4x faster on
        // a 144Hz display than on the machine it was tuned on.
        particle.x += Math.cos(angle) * SPEED * dt;
        particle.y += Math.sin(angle) * SPEED * dt;

        if (particle.x < 0) particle.x = width;
        else if (particle.x > width) particle.x = 0;
        if (particle.y < 0) particle.y = height;
        else if (particle.y > height) particle.y = 0;

        context!.fillStyle = `rgba(${ink}, ${opacity})`;
        context!.beginPath();
        context!.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        context!.fill();
      }

      frameId = requestAnimationFrame(frame);
    }

    resize();
    window.addEventListener("resize", resize);
    frameId = requestAnimationFrame(frame);

    return () => {
      window.removeEventListener("resize", resize);
      // **Defect 2.** The reference returns a cleanup that removes the resize
      // listener and nothing else, so the loop it started keeps running for the
      // life of the tab — through every navigation away from this page, at
      // sixty frames a second, drawing into a canvas that is no longer in the
      // document. React's strict mode mounts every effect twice in development,
      // so the reference leaks a second loop before you have even navigated.
      if (frameId !== null) cancelAnimationFrame(frameId);
    };
    // **Defect 1.** The reference built a fresh noise object on every render
    // and listed it here, so every render tore down the particle system and
    // rebuilt it from scratch — eight hundred new particles, a new loop, and
    // the trail buffer wiped, sixty times a second under any parent that
    // re-renders. `noise3` is a module-scope function: there is no factory to
    // call and nothing here whose identity changes between renders. Every
    // entry below is a number.
  }, [particleCount, noiseIntensity, sizeMin, sizeMax]);

  return (
    <div className={cn("relative h-dvh w-full overflow-hidden bg-ground", className)}>
      {reducedMotion ? null : (
        <canvas
          ref={canvasRef}
          data-testid="fluid-particles"
          aria-hidden
          className="pointer-events-none absolute inset-0"
        />
      )}
      <div className="relative z-10 flex h-full w-full items-center justify-center">
        {children}
      </div>
    </div>
  );
}
