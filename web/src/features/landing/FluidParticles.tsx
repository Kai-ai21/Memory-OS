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

import { BrushLayer } from "../../components/BrushLayer";
import { makeMask, readShelters, type Mask } from "../../lib/mask";
import { noise3 } from "../../lib/noise";
import { readGround, readInk } from "../../lib/tokens";
import { cn } from "../../lib/utils";

export interface FluidParticlesProps {
  children?: ReactNode;
  /**
   * The reference shipped 2000 and M9.4 cut it to 800; M9.5 put it back to
   * 1200. With a trail buffer the density you perceive is the smear rather than
   * the count, so this is the least effective of the three levers — the ones
   * that changed the picture were opacity and, far more, the clear alpha.
   */
  particleCount?: number;
  /** Field scale. Larger means the flow turns over a shorter distance. */
  noiseIntensity?: number;
  particleSize?: { min: number; max: number };
  className?: string;
}

/**
 * Peak opacity of one ambient particle. **0.35, from 0.22.**
 *
 * Nearly six times what M9.4 shipped. It is affordable here for a reason that
 * does not hold inside the application: this page is a wordmark, a card and a
 * link on an otherwise empty screen, and everything that has to be read sits
 * inside a shelter the field is not drawn over at all. The particles are the
 * page.
 */
const PEAK_OPACITY = 0.35;

/**
 * How much page colour is laid over the previous frame each 60fps frame.
 * **0.03, as specified.**
 *
 * This is the knob for trail length, inverted: lower is longer. It is also the
 * only thing removing ink, so a pixel's composited darkness settles at roughly
 * the ink arriving per frame divided by this number — at 0.35 over 0.03 that
 * ratio is eleven, which means every pixel a particle crosses saturates to
 * solid ink and stays there for the two or three seconds it takes 0.97^n to
 * reach zero.
 *
 * **That is the intended effect here, not a bug, and it is why the shelter had
 * to widen to 340px in the same change.** Outside the shelter the page is meant
 * to be dark drifting structure rather than grey fog; inside it there is no ink
 * at all. What the number costs is measured in the milestone report — the
 * unmasked area is very dark, and that is what was asked for.
 */
const TRAIL_ALPHA = 0.03;

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

/**
 * How far outside the centred content the field climbs back to full strength.
 *
 * 340px, and it is wide because the field behind it got much heavier: at a
 * clear alpha of 0.03 the unmasked page saturates, so the distance from the
 * last legible pixel of type to the first fully dark one has to be enough that
 * the ramp between them is not itself a distraction. There is nothing else on
 * this page to protect — the wordmark, the card and the link are one box — so
 * one shelter covers everything and the rest of the screen is free.
 */
const SHELTER_FEATHER = 340;

/** How often the shelter is re-measured, in frames. Four times a second. */
const REMEASURE_EVERY = 15;

/**
 * The drift and the stroke now share one shelter, at full width.
 *
 * M9.5 gave the drift 45% of the stroke's feather, because at 0.22 it needed
 * less room and a full-width shelter emptied the page. M9.6 asks for
 * "particles and stroke fade to zero within 340px", which settles it — and at
 * 0.35 with a 0.03 clear the drift needs every pixel of that anyway. A filament
 * at these values crossing a line of 11px type does not make it harder to read,
 * it removes it.
 */
const AMBIENT_SHELTER_SCALE = 1;

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
  particleCount = 1200,
  noiseIntensity = 0.003,
  particleSize = { min: 2, max: 5 },
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
    let sinceMeasure = REMEASURE_EVERY;
    /** See `AMBIENT_SHELTER_SCALE`. The stroke keeps its own, in `BrushLayer`. */
    let driftMask: Mask = () => 1;

    function frame(now: number) {
      const dt = previous === 0 ? 1000 / 60 : Math.min(now - previous, MAX_STEP);
      previous = now;

      // The shelter is read off the DOM a few times a second rather than
      // configured, so it follows the layout instead of duplicating it.
      sinceMeasure += 1;
      if (sinceMeasure >= REMEASURE_EVERY) {
        sinceMeasure = 0;
        const shelters = readShelters(SHELTER_FEATHER);
        driftMask = makeMask(
          shelters.map((shelter) => ({
            ...shelter,
            feather: shelter.feather * AMBIENT_SHELTER_SCALE,
          })),
        );
      }

      // The trail. Page colour at low alpha rather than a clear: this is the
      // whole reason the effect reads as ink rather than as dots.
      context!.fillStyle = `rgba(${ground}, ${TRAIL_ALPHA * (dt / (1000 / 60))})`;
      context!.fillRect(0, 0, width, height);

      for (const particle of particles) {
        particle.life += dt;
        if (particle.life > particle.maxLife) {
          Object.assign(particle, spawn(false));
        }

        // Fade in and out over a life, so nothing pops into or out of being,
        // and again by the drift's own shelter, so the small type on this page
        // is not crossed by a filament.
        const opacity =
          Math.sin((particle.life / particle.maxLife) * Math.PI) *
          PEAK_OPACITY *
          driftMask(particle.x, particle.y);

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

        if (opacity > 0.002) {
          context!.fillStyle = `rgba(${ink}, ${opacity})`;
          context!.beginPath();
          context!.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
          context!.fill();
        }
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
      {reducedMotion ? null : (
        <BrushLayer feather={SHELTER_FEATHER} className="absolute" />
      )}
      <div className="relative z-10 flex h-full w-full items-center justify-center">
        {children}
      </div>
    </div>
  );
}
