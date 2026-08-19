/**
 * The brush stroke's own canvas, its own frame loop, and its own idle rule.
 *
 * **A separate layer from the ambient field, deliberately.** The two want
 * opposite compositing: the drift accumulates onto a buffer that is faded
 * rather than cleared, and the stroke is redrawn from scratch every frame from
 * a list of positions. Sharing a canvas would mean either the stroke smearing
 * into the drift's trail buffer or the drift losing its buffer. Separating them
 * also means the blur applies to the stroke alone, which is the only thing that
 * wants one.
 *
 * **And its own loop, which is usually not running.** The drift is always
 * moving and its loop cannot stop; a stroke exists only after you move the
 * mouse and is gone 1.5 seconds later. Two seconds of stillness with nothing
 * left to draw and this cancels its frame request; the next `mousemove` starts
 * it again. On a page being read rather than pointed at, this component costs
 * one idle event listener — which is what a blurred full-viewport redraw every
 * frame is worth when there is nothing on it.
 */

import { useEffect, useRef, useState } from "react";

import { BLUR_RADIUS, BrushStroke, LIFETIME } from "../lib/brush";
import { makeMask, readShelters, type Mask } from "../lib/mask";

/** Silence for this long, with an empty stroke, and the loop shuts down. */
export const IDLE_MS = 2000;

/** How often the shelter is re-measured, in frames. Four times a second. */
const REMEASURE_EVERY = 15;

export interface BrushLayerProps {
  /** Distance over which the stroke climbs back to full strength outside a shelter. */
  feather: number;
  /** Extra classes for the canvas — stacking is the caller's business. */
  className?: string;
}

/**
 * Reduced motion, resolved in an effect and watched afterwards.
 *
 * A dark stroke chasing the pointer across the whole viewport is the most
 * motion anything in this application produces. `reduce` means this component
 * renders nothing at all — no canvas, no listeners, no loop.
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

export function BrushLayer({ feather, className = "" }: BrushLayerProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const stroke = new BrushStroke();
    let frameId: number | null = null;
    let lastMoveAt = 0;
    let sinceMeasure = REMEASURE_EVERY;
    let mask: Mask = () => 1;
    let width = 0;
    let height = 0;
    let ratio = 1;

    function resize() {
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas!.width = Math.round(width * ratio);
      canvas!.height = Math.round(height * ratio);
      canvas!.style.width = `${width}px`;
      canvas!.style.height = `${height}px`;
    }

    function frame(now: number) {
      frameId = null;

      sinceMeasure += 1;
      if (sinceMeasure >= REMEASURE_EVERY) {
        sinceMeasure = 0;
        mask = makeMask(readShelters(feather));
      }

      stroke.prune(now);

      // Cleared outright rather than faded. The taper *is* the fade — every
      // point's opacity is a function of its own age — so a trail buffer here
      // would apply a second, slower decay on top of it and the stroke would
      // leave a permanent smear behind the one that is supposed to be fading.
      context!.setTransform(1, 0, 0, 1, 0, 0);
      context!.clearRect(0, 0, canvas!.width, canvas!.height);
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);
      stroke.draw(context!, now, mask);

      // Both halves are needed. Quitting on silence alone would delete a stroke
      // mid-fade; quitting on an empty stroke alone would quit between two
      // samples of a slow movement.
      if (now - lastMoveAt > IDLE_MS && !stroke.active) return;
      frameId = requestAnimationFrame(frame);
    }

    function start() {
      if (frameId !== null || document.hidden) return;
      frameId = requestAnimationFrame(frame);
    }

    function onMove(event: MouseEvent) {
      lastMoveAt = performance.now();
      stroke.push(event.clientX, event.clientY, lastMoveAt);
      start();
    }

    /** The pointer's next appearance is somewhere unrelated; do not join them. */
    function onLeave() {
      stroke.clear();
    }

    function onVisibility() {
      if (!document.hidden) return;
      if (frameId !== null) cancelAnimationFrame(frameId);
      frameId = null;
      stroke.clear();
      context!.setTransform(1, 0, 0, 1, 0, 0);
      context!.clearRect(0, 0, canvas!.width, canvas!.height);
    }

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", onLeave);
    window.addEventListener("blur", onLeave);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("blur", onLeave);
      document.removeEventListener("visibilitychange", onVisibility);
      if (frameId !== null) cancelAnimationFrame(frameId);
    };
    // Everything else is a module constant; only the feather can move.
  }, [feather]);

  if (reducedMotion) return null;

  return (
    <canvas
      ref={ref}
      data-testid="brush-layer"
      aria-hidden
      /* **The blur is a CSS filter on the element, not `ctx.filter`.** Both were
         measured; the numbers are in the milestone report. `ctx.filter` re-runs
         a Gaussian over the drawn region on the CPU for every one of the sixty
         stroke calls that make up the taper, and it is the dominant cost of the
         frame by an order of magnitude. As a CSS filter it is one GPU pass on a
         layer that is already being composited, and it costs approximately
         nothing — which is available here precisely because the stroke has a
         canvas to itself and nothing else on it needs blurring. */
      style={{ filter: `blur(${BLUR_RADIUS}px)` }}
      className={`pointer-events-none fixed inset-0 ${className}`}
    />
  );
}

/** Re-exported so a caller can size its idle expectations off one number. */
export { LIFETIME };
