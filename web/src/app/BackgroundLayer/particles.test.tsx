/**
 * The properties of the background field that a person could not check by
 * looking at it.
 *
 * Everything else about this effect is a judgement about how faint is faint
 * enough and how long a tail should be, and a test pinning those would pin
 * numbers that are meant to be tuned by eye. What is here is the set that fails
 * silently and expensively: a canvas that ignores an accessibility switch, an
 * array with no ceiling, a loop that never stops, a trail that stops reading as
 * a trail, and the contrast the ink costs the text under it.
 *
 * **No real frame loop runs here.** `requestAnimationFrame` is replaced with a
 * queue of one and driven by hand off a virtual clock, so "the loop stops" is
 * asserted on whether a frame was *scheduled* rather than on a timer expiring.
 */

/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { BackgroundLayer } from ".";
import { AMBIENT_ALPHA_MAX, AMBIENT_COUNT, AmbientLayer } from "./ambient";
import { HEAD_ALPHA as STROKE_ALPHA } from "../../lib/brush";
import { makeMask } from "../../lib/mask";
import { INK_FALLBACK } from "./field";
import { noise3 } from "../../lib/noise";

/* --- The environment the component reads ---------------------------------- */

/** Which media queries answer true. jsdom ships no `matchMedia` at all. */
function stubMedia(truthy: string[] = []) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: truthy.some((needle) => query.includes(needle)),
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  );
}

/**
 * A 2D context that records nothing and draws nothing.
 *
 * jsdom's `getContext` returns null without the `canvas` package, which would
 * make the component bail out before it ever wired a listener — so the loop
 * test would pass for the wrong reason.
 */
function stubCanvas() {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    createRadialGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
    globalAlpha: 1,
    globalCompositeOperation: "source-over",
    fillStyle: "",
  } as unknown as CanvasRenderingContext2D);
}

/** A hand-cranked clock and frame queue. */
function stubFrames() {
  const state = { now: 0, pending: null as FrameRequestCallback | null };

  vi.spyOn(performance, "now").mockImplementation(() => state.now);
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((callback: FrameRequestCallback) => {
      state.pending = callback;
      return 1;
    }),
  );
  vi.stubGlobal(
    "cancelAnimationFrame",
    vi.fn(() => {
      state.pending = null;
    }),
  );

  return {
    get scheduled() {
      return state.pending !== null;
    },
    move(x: number, y: number) {
      fireEvent.mouseMove(window, { clientX: x, clientY: y });
    },
    /** Run frames at ~60fps until `ms` of virtual time has passed. */
    run(ms: number) {
      const until = state.now + ms;
      while (state.now < until) {
        state.now += 16;
        const callback = state.pending;
        if (!callback) continue;
        state.pending = null;
        callback(state.now);
      }
    },
  };
}

/** jsdom's `document.hidden` is read-only; this is the only way to move it. */
function setHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, value: hidden });
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: hidden ? "hidden" : "visible",
  });
  fireEvent(document, new Event("visibilitychange"));
}

beforeEach(() => {
  window.localStorage.clear();
  stubCanvas();
});

afterEach(() => {
  setHidden(false);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/* --- The gates ------------------------------------------------------------ */

describe("reduced motion", () => {
  it("creates no canvas at all, and still paints the wash", () => {
    // Not "creates a paused canvas" and not "creates one with a shorter
    // animation". This is a vestibular-trigger effect, and the person who set
    // that switch set it for exactly this; the only correct amount is none.
    // It matters more now than it did with one layer: the ambient drift never
    // stops on its own, so there is no moment at which it is not moving.
    stubMedia(["prefers-reduced-motion"]);
    stubFrames();

    render(<BackgroundLayer />);

    expect(screen.queryByTestId("particle-canvas")).not.toBeInTheDocument();
    expect(screen.getByTestId("background-layer")).toBeInTheDocument();
    // And no switch either: a control offering to turn on something that is
    // forbidden is a control that lies.
    expect(screen.queryByTestId("particle-toggle")).not.toBeInTheDocument();
  });

  it("creates no canvas on a touch-primary pointer", () => {
    // Half the effect follows a cursor that does not exist, and the other half
    // would drift on eight hundred particles for as long as the battery lasted.
    stubMedia(["pointer: coarse"]);
    stubFrames();

    render(<BackgroundLayer />);

    expect(screen.queryByTestId("particle-canvas")).not.toBeInTheDocument();
  });
});

/* --- The two layers, separately -------------------------------------------- */

/* The cursor layer's tests went with the cursor layer. M9.6 replaced the
   emitter with a stroke, which is `lib/brush` and is tested beside it in
   `components/BrushLayer.test.tsx`. What stays here is the drift, the loop it
   runs in, and what the two of them cost the text. */

describe("the ambient layer", () => {
  it("is a fixed pool: exactly AMBIENT_COUNT, and it never grows", () => {
    // A pool rather than a spawner is what makes the cost of this layer a
    // constant. Particles are recycled in place when their life runs out, so a
    // leak here would show up as an array that creeps upward over minutes.
    const layer = new AmbientLayer(() => 0.5);
    layer.resize(1280, 800);
    expect(layer.particles).toHaveLength(AMBIENT_COUNT);

    // Well past the longest ambient lifetime, so every particle has recycled.
    for (let i = 0; i < 1500; i += 1) layer.step(16, i * 16);
    expect(layer.particles).toHaveLength(AMBIENT_COUNT);
  });

  it("keeps its particles on screen, and keeps the pool through a resize", () => {
    // Wrapped rather than bounced, and a window drag must not blank the layer
    // and refill it over the next ten seconds — which is far more visible than
    // the resize was.
    const layer = new AmbientLayer(Math.random);
    layer.resize(1280, 800);
    for (let i = 0; i < 600; i += 1) layer.step(16, i * 16);

    layer.resize(640, 480);
    expect(layer.particles).toHaveLength(AMBIENT_COUNT);
    for (const speck of layer.particles) {
      expect(speck.x).toBeGreaterThanOrEqual(-8);
      expect(speck.x).toBeLessThanOrEqual(640 + 8);
      expect(speck.y).toBeGreaterThanOrEqual(-8);
      expect(speck.y).toBeLessThanOrEqual(480 + 8);
    }
  });

  it("never draws a particle above its own ceiling", () => {
    const layer = new AmbientLayer(Math.random);
    layer.resize(1280, 800);
    for (let i = 0; i < 900; i += 1) {
      layer.step(16, i * 16);
      for (const speck of layer.particles) {
        expect(speck.alpha).toBeLessThanOrEqual(AMBIENT_ALPHA_MAX);
      }
    }
  });
});

describe("the flow field both layers ride", () => {
  it("is deterministic and stays in range", () => {
    // The noise moved to `lib` in M9.4 so the landing page could sample it
    // without importing from this directory. Asserted from here as well as
    // there: this is the caller whose look depends on it being reproducible.
    // Seeded from a constant so "why does it look like that" has an answer. A
    // random seed would make the field unreproducible between two loads of the
    // same page.
    for (let i = 0; i < 500; i += 1) {
      const value = noise3(i * 0.31, i * 0.17, i * 0.05);
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(noise3(3.7, 1.2, 0.4)).toBe(noise3(3.7, 1.2, 0.4));
  });
});

/* --- The loop -------------------------------------------------------------- */

describe("the loop", () => {
  it("runs while the document is visible and stops while it is not", () => {
    // **This replaces step 1a's idle stop, which no longer applies.** That rule
    // could cancel the loop two seconds after the last movement because nothing
    // on screen was independent of the cursor. The ambient layer is by
    // definition always moving, so the honest contract is now visibility: a
    // backgrounded tab schedules nothing and holds no buffer.
    stubMedia();
    const frames = stubFrames();

    render(<BackgroundLayer />);
    expect(screen.getByTestId("particle-canvas")).toBeInTheDocument();

    // Running from mount, with no cursor input at all — the drift does not
    // wait to be asked.
    expect(frames.scheduled).toBe(true);
    frames.run(500);
    expect(frames.scheduled).toBe(true);

    setHidden(true);
    expect(frames.scheduled).toBe(false);
    frames.run(2000);
    expect(frames.scheduled).toBe(false);

    setHidden(false);
    expect(frames.scheduled).toBe(true);
  });

  it("tears the loop down with the component", () => {
    // An unmount that left a frame scheduled would keep a whole canvas, two
    // layers and a thousand particles alive behind a page that no longer has a
    // background at all.
    stubMedia();
    const frames = stubFrames();

    const view = render(<BackgroundLayer />);
    expect(frames.scheduled).toBe(true);

    view.unmount();
    expect(frames.scheduled).toBe(false);
  });
});

/* --- The switch ------------------------------------------------------------ */

describe("the preference", () => {
  it("turns the canvas off and remembers it across a remount", () => {
    stubMedia();
    stubFrames();

    const first = render(<BackgroundLayer />);
    expect(screen.getByTestId("particle-canvas")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("particle-toggle"));
    expect(screen.queryByTestId("particle-canvas")).not.toBeInTheDocument();

    first.unmount();
    render(<BackgroundLayer />);
    expect(screen.queryByTestId("particle-canvas")).not.toBeInTheDocument();
    expect(screen.getByTestId("particle-toggle")).toHaveAttribute("aria-pressed", "false");
  });
});

/* --- What a particle costs the text under it ------------------------------- */

/**
 * **Read this before changing an opacity, and read the model before the
 * numbers — step 1a got the model wrong.**
 *
 * A particle does not veil the text. The canvas is at `-z-10`, *behind* every
 * glyph in the document, so ink laid down by the field darkens the paper
 * between the letters and leaves the letters themselves untouched. Step 1a
 * computed contrast as though the veil covered both, which is the arithmetic
 * for a scrim drawn on top; compositing the same alpha into the background
 * only makes contrast fall roughly twice as fast. Every figure below uses the
 * correct model — token colour against veiled ground — and they are worse than
 * the ones 1a reported for the layer it shipped.
 *
 * Where each role crosses WCAG AA on this palette, as background veil alpha:
 *
 * | role                  | AA clean | crosses 4.5:1 at |
 * |-----------------------|----------|------------------|
 * | `ink` (body, headings)|   16.80  |      α 0.536     |
 * | `ink-2`               |    7.13  |      α 0.216     |
 * | `ink-3` (micro-labels)|    5.09  |      α 0.062     |
 * | `accent` (links)      |    4.86  |      α 0.040     |
 *
 * The two lightest roles have almost no headroom: they clear AA by 0.59 and
 * 0.36 respectively with nothing on top of them, so *any* visible veil takes
 * them under. That is a fact about the palette, not about this effect.
 *
 * What is asserted here is therefore the one structural invariant: **a single
 * particle from either layer leaves body text above AA.** The rest is
 * measured, recorded, and deliberately not asserted away — the cursor layer's
 * 0.35 is an instruction, and the trail buffer composites several particles
 * on top of each other besides. The measured consequences, at the shipped
 * clear alpha of 0.14 on an 800x600 viewport, are in the report and in
 * `ParticleCanvas.tsx`.
 */
describe("what a particle costs the text under it", () => {
  const TOKENS = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");

  function channels(name: string): [number, number, number] {
    const match = TOKENS.match(new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6})\\s*;`));
    if (!match) throw new Error(`--color-${name} is not declared as a hex`);
    const value = match[1];
    return [1, 3, 5].map((at) => parseInt(value.slice(at, at + 2), 16)) as [
      number,
      number,
      number,
    ];
  }

  const INK = INK_FALLBACK.split(",").map((part) => Number(part.trim())) as [
    number,
    number,
    number,
  ];

  function luminance([r, g, b]: [number, number, number]): number {
    const linear = [r, g, b].map((value) => {
      const unit = value / 255;
      return unit <= 0.03928 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  }

  /**
   * Contrast of an *unveiled* glyph against ground with `alpha` of ink
   * composited into it. This is the layering the component actually produces.
   */
  function ratio(role: string, alpha: number): number {
    const ground = channels("ground").map((c, i) => c * (1 - alpha) + INK[i] * alpha) as [
      number,
      number,
      number,
    ];
    const a = luminance(channels(role));
    const b = luminance(ground);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  }

  it("declares an ink that is the ink token, not black", () => {
    // `#000` on a warm white ground reads as a hole rather than as smoke, and
    // the sprite is built from whatever the token says at runtime — so the
    // fallback baked into `field.ts` has to be the colour the palette declares.
    expect(INK).toEqual(channels("ink"));
  });

  it("leaves body text above AA under one ambient particle", () => {
    // The ambient layer is permanent and unmasked, so its ceiling still has to
    // clear the bar on its own. It does, with room: 0.08 against a crossing
    // point of 0.536.
    expect(ratio("ink", AMBIENT_ALPHA_MAX)).toBeGreaterThanOrEqual(4.5);
  });

  it("records that the stroke alone would not, and why that is survivable", () => {
    // **The cursor mark is far past the point where opacity alone is safe.** At
    // its 0.85 head it is not "harder to read through", it is opaque; `ink`
    // crosses AA against darkened paper at α 0.536 and every lighter role went
    // under long before that.
    //
    // What replaces the opacity ceiling is positional: the reading column is a
    // shelter and the stroke is multiplied by zero inside it. That is asserted
    // below rather than here, because it is the thing that is actually true —
    // "the mark is never dark over text" is a fact about *where* it is drawn,
    // not about how dark it is.
    expect(STROKE_ALPHA).toBe(0.85);
    expect(ratio("ink", STROKE_ALPHA)).toBeLessThan(4.5);
  });

  it("draws no stroke ink at all inside a shelter", () => {
    // The replacement gate, and the reason 0.55 can ship. A reading column is
    // a shelter; inside one the multiplier is exactly zero, at the centre and
    // hard against every edge, so no peak opacity anywhere above can put ink
    // behind a paragraph.
    const column = { left: 300, top: 100, right: 900, bottom: 700, feather: 160 };
    const mask = makeMask([column]);

    expect(mask(600, 400)).toBe(0);
    expect(mask(300, 100)).toBe(0);
    expect(mask(900, 700)).toBe(0);
    // And it is full strength once clear of the feather, which is what keeps
    // the margins dramatic rather than merely less timid.
    expect(mask(900 + 160, 400)).toBe(1);
    expect(mask(60, 400)).toBe(1);
    // Monotonic in between, with no crease: a linear ramp is visible as one.
    const ramp = [40, 80, 120].map((d) => mask(900 + d, 400));
    expect(ramp[0]).toBeLessThan(ramp[1]);
    expect(ramp[1]).toBeLessThan(ramp[2]);
    expect(ramp[2]).toBeLessThan(1);
  });

  it("pins where each role crosses AA, so a palette change cannot move it quietly", () => {
    // These four numbers are what every opacity in the effect is tuned
    // against. If the palette is revised and `accent` gets lighter, this row
    // goes red before anybody has to notice by eye.
    const crossing = (role: string) => {
      for (let alpha = 0; alpha <= 1; alpha += 0.002) {
        if (ratio(role, alpha) < 4.5) return alpha;
      }
      return 1;
    };

    expect(crossing("ink")).toBeGreaterThan(0.5);
    expect(crossing("ink-2")).toBeGreaterThan(0.2);
    expect(crossing("ink-3")).toBeGreaterThan(0.055);
    expect(crossing("accent")).toBeGreaterThan(0.035);
  });

  it("records what each layer's ceiling costs the two lightest roles", () => {
    // Not a pass/fail on the design. The ambient ceiling is 0.08 and the
    // stroke's head is 0.85, both by instruction, and both are above the point
    // at which micro-labels and links stop clearing AA against darkened paper.
    // Recorded so the cost stays a known quantity rather than a surprise.
    expect(AMBIENT_ALPHA_MAX).toBe(0.08);
    expect(STROKE_ALPHA).toBe(0.85);

    expect(ratio("ink-3", AMBIENT_ALPHA_MAX)).toBeLessThan(4.5);
    expect(ratio("accent", AMBIENT_ALPHA_MAX)).toBeLessThan(4.5);
    expect(ratio("ink-2", STROKE_ALPHA)).toBeLessThan(4.5);
  });
});
