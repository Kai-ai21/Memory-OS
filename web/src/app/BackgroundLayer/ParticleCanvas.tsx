/**
 * The canvas, the frame loop, and the rule that the loop is usually not running.
 *
 * **Canvas rather than elements.** A hundred and fifty absolutely-positioned
 * divs with a transform and an opacity each is a hundred and fifty composited
 * layers for an effect nobody is supposed to consciously see. One canvas, one
 * paint, one `drawImage` per particle.
 *
 * **The loop stops.** This is the part worth reading. A `requestAnimationFrame`
 * loop that runs forever is a wakeup sixty times a second, in perpetuity,
 * behind a page that is not moving — on a laptop that is measurable battery for
 * literally nothing. So: no frame is scheduled until the mouse moves, and the
 * loop cancels itself two seconds after the last movement *and* the last
 * particle. The next `mousemove` starts it again. On a page being read rather
 * than pointed at, this component costs one idle event listener.
 *
 * Nothing here is reachable by the pointer or by a screen reader, and the
 * canvas sits at the same negative z-index as the wash it draws over — behind
 * every element in the application, including the ones it appears to pass
 * beneath.
 */

import { useEffect, useRef } from "react";

import { ParticleField, buildSprite } from "./field";

/** Silence for this long, with nothing still alive, and the loop shuts down. */
export const IDLE_MS = 2000;

/**
 * The largest step a single frame may claim, in milliseconds.
 *
 * A backgrounded tab hands back a delta of minutes on its first frame. Without
 * a clamp, every live particle ages out at once and the field blinks empty the
 * moment you switch back to the window.
 */
const MAX_STEP = 64;

export function ParticleCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const sprite = buildSprite();
    if (!context || !sprite) return;

    const field = new ParticleField();
    let frameId: number | null = null;
    let lastFrameAt = 0;
    let lastMoveAt = 0;
    let width = 0;
    let height = 0;

    /**
     * Size the backing store to device pixels and the element to CSS pixels.
     *
     * Without the `devicePixelRatio` multiply the whole field is drawn at half
     * resolution on a retina display and upscaled — which on a soft gradient
     * sprite does not read as "blurry", it reads as banding.
     */
    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas!.width = Math.round(width * ratio);
      canvas!.height = Math.round(height * ratio);
      canvas!.style.width = `${width}px`;
      canvas!.style.height = `${height}px`;
      // Set rather than scaled: `resize` runs many times during a drag, and a
      // relative `scale` would compound.
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function frame(now: number) {
      frameId = null;
      const dt = Math.min(now - lastFrameAt, MAX_STEP);
      lastFrameAt = now;

      field.step(dt);
      context!.clearRect(0, 0, width, height);
      field.draw(context!, sprite!);

      // The whole idle rule, in one condition. Both halves are needed: quitting
      // on silence alone would delete a field mid-fade, and quitting on an
      // empty field alone would quit between two particles.
      if (now - lastMoveAt > IDLE_MS && field.particles.length === 0) {
        field.breakTrail();
        return;
      }
      frameId = requestAnimationFrame(frame);
    }

    function start(now: number) {
      if (frameId !== null) return;
      // Seeded from the clock the loop is about to be handed, so the first
      // frame after an idle stretch is a step of ~16ms and not of ten seconds.
      lastFrameAt = now;
      frameId = requestAnimationFrame(frame);
    }

    function onMove(event: MouseEvent) {
      lastMoveAt = performance.now();
      field.push(event.clientX, event.clientY);
      start(lastMoveAt);
    }

    /**
     * Leaving the window breaks the trail rather than stopping the loop.
     *
     * The cursor's next appearance is somewhere unrelated, and a segment drawn
     * between where it left and where it returned would emit a line of
     * particles across a screen the pointer never crossed.
     */
    function onLeave() {
      field.breakTrail();
    }

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", onLeave);
    window.addEventListener("blur", onLeave);

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("blur", onLeave);
      if (frameId !== null) cancelAnimationFrame(frameId);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      data-testid="particle-canvas"
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10"
    />
  );
}
