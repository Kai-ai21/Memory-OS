/**
 * One canvas, one frame loop, two layers, and a buffer that does not fully
 * clear.
 *
 * **The trail buffer is the whole reason the cursor layer reads as a stroke.**
 * Clearing outright and redrawing gives you whatever is alive this instant —
 * two hundred discrete blobs — and no matter how you tune them they stay two
 * hundred discrete blobs. Fading the previous frame instead leaves every
 * position a particle has occupied in the last second still faintly on screen,
 * and the overlap of a moving particle with its own recent past is what turns a
 * row of dots into a smear.
 *
 * **It is erased, not painted over, and that is a deviation worth reading.**
 * The obvious implementation of a fading buffer is to fill each frame with the
 * page colour at low alpha. That cannot be done here. This canvas sits *over*
 * the two wash radials — same stacking level, later sibling — and a repeated
 * translucent fill converges on a fully opaque rectangle in about forty frames,
 * so the wash would disappear underneath it. The wash exists for exactly one
 * element, the NEW CONVERSATION button, which is the only frosted glass in the
 * application and stops reading as glass the moment there is nothing behind it
 * to frost; `theme.test.ts` has a test whose entire job is to prevent that.
 * `destination-out` gets the identical decay by removing alpha instead of
 * adding colour, and leaves the canvas genuinely transparent between the marks.
 *
 * **The decay is per millisecond, not per frame.** A fixed per-frame alpha
 * makes the trail half as long on a 120Hz display as on a 60Hz one, for the
 * same gesture — the kind of bug that only ever reproduces on somebody else's
 * machine.
 *
 * **The idle stop is gone, and it had to be.** Step 1a could cancel the loop
 * two seconds after the last movement because there was nothing on screen that
 * was not a response to the cursor. An ambient layer is by definition always
 * moving, so the loop now runs whenever the document is visible and stops when
 * it is not. `visibilitychange` is belt and braces — browsers already stop
 * serving frames to a hidden tab — but it also releases the trail buffer, so a
 * backgrounded tab is not holding a viewport of pixels it will never show.
 */

import { useEffect, useRef } from "react";

import { AmbientLayer } from "./ambient";
import { buildSprite } from "./field";

/**
 * How much of the buffer is erased per 60fps frame. **Settled at 0.14.**
 *
 * The brief's starting point was 0.08 and it is too slow on a light ground —
 * measurably, not as a matter of taste. The ambient layer lays ink down every
 * frame and this number is the only thing taking it away, so the layer's
 * *composited* darkness is roughly its per-particle alpha divided by this
 * value. At 0.08 the field reaches a steady state covering **41% of the
 * viewport** in visible ink: eight hundred particles tracing the flow field's
 * contour lines onto the page and leaving them there, which reads as scratched
 * paper rather than as weather. At 0.14 the same layer settles at **1.1%**.
 *
 * The trail survives the change, which is the other half of the test. A 2px/ms
 * sweep still peaks at a composited 0.47 — a smear you can plainly see — and is
 * back to the ambient baseline about 1.2 seconds later. Below roughly 0.10 the
 * page silts up; above roughly 0.20 the tail is gone before the eye follows it.
 *
 * Read the other way: this is the knob for trail *length*. Lower is longer.
 */
const CLEAR_ALPHA = 0.14;

/** The largest step one frame may claim, in milliseconds. */
const MAX_STEP = 64;

/**
 * How far outside the reading column the brush stroke climbs back to full
 * strength, in CSS pixels.
 *
 * Owned here rather than in `BrushLayer` because it is a fact about *this*
 * layout: the app's margins are narrow, and a feather as wide as the landing
 * page's would mean the stroke never reaches full strength anywhere on a
 * laptop — the same failure as turning it down. `Shell` marks its content
 * wrapper as a shelter; inside it the stroke is multiplied by zero, and it is
 * back to full 160px out, which lands in the gutter or behind the sidebar.
 *
 * The ambient drift is deliberately *not* masked in the application. It runs at
 * 0.08 — an eighth of the stroke — and sheltering it as well would leave a
 * visible rectangular hole in the wash across the middle of every screen.
 */
export const APP_SHELTER_FEATHER = 160;

export function ParticleCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const sprite = buildSprite();
    if (!context || !sprite) return;

    const ambient = new AmbientLayer();
    let frameId: number | null = null;
    let lastFrameAt = 0;
    let width = 0;
    let height = 0;

    /**
     * Size the backing store to device pixels and the element to CSS pixels.
     *
     * Without the `devicePixelRatio` multiply the field is drawn at half
     * resolution on a retina display and upscaled, which on a soft gradient
     * sprite does not read as "blurry" — it reads as banding. Capped at 2:
     * a 3x phone would triple the fill cost of the trail buffer for a
     * difference nobody can see on a 0.08 alpha smear.
     */
    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas!.width = Math.round(width * ratio);
      canvas!.height = Math.round(height * ratio);
      canvas!.style.width = `${width}px`;
      canvas!.style.height = `${height}px`;
      // Set rather than scaled: `resize` fires many times during a drag, and a
      // relative `scale` would compound.
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);
      ambient.resize(width, height);
    }

    function frame(now: number) {
      frameId = null;
      const dt = Math.min(now - lastFrameAt, MAX_STEP);
      lastFrameAt = now;

      // Fade the previous frame. `destination-out` scales the alpha already in
      // the buffer; the colour of the fill is irrelevant, only its alpha counts.
      const decay = 1 - Math.pow(1 - CLEAR_ALPHA, dt / (1000 / 60));
      context!.globalCompositeOperation = "destination-out";
      context!.fillStyle = `rgba(0, 0, 0, ${decay})`;
      context!.fillRect(0, 0, width, height);
      context!.globalCompositeOperation = "source-over";

      ambient.step(dt, now);
      ambient.draw(context!, sprite!);

      frameId = requestAnimationFrame(frame);
    }

    function start() {
      if (frameId !== null || document.hidden) return;
      // Seeded from the clock the loop is about to be handed, so the first
      // frame after a hidden stretch is a step of ~16ms and not of ten minutes.
      lastFrameAt = performance.now();
      frameId = requestAnimationFrame(frame);
    }

    function stop() {
      if (frameId === null) return;
      cancelAnimationFrame(frameId);
      frameId = null;
    }

    function onVisibility() {
      if (document.hidden) {
        stop();
        // Drop the buffer as well as the loop. A hidden tab holding a viewport
        // of pixels it cannot show is the memory half of the same waste.
        context!.setTransform(1, 0, 0, 1, 0, 0);
        context!.clearRect(0, 0, canvas!.width, canvas!.height);
        resize();
      } else {
        start();
      }
    }

    resize();
    start();
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
      stop();
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
