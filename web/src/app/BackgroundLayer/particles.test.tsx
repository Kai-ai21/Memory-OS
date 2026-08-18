/**
 * The three properties of the background field that a person could not check by
 * looking at it.
 *
 * Everything else about this effect is a judgement call about how faint is
 * faint enough, and a test asserting `peak <= 0.12` would pin a number that is
 * meant to be tuned by eye. These three are different: each one is a promise
 * that fails silently and expensively.
 *
 * **No real frame loop runs here.** `requestAnimationFrame` is replaced with a
 * queue of one and driven by hand off a virtual clock, so "the loop stops" is
 * asserted on whether a frame was *scheduled* rather than on a timer expiring
 * — which is the actual claim, and which a `setTimeout` in a test suite could
 * only approximate.
 */

/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { BackgroundLayer } from ".";
import { ALPHA_MAX, INK_FALLBACK, MAX_PARTICLES, ParticleField } from "./field";

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
    now: () => state.now,
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

beforeEach(() => {
  window.localStorage.clear();
  stubCanvas();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/* --- The three ------------------------------------------------------------ */

describe("reduced motion", () => {
  it("creates no canvas at all, and still paints the wash", () => {
    // Not "creates a paused canvas" and not "creates one with a shorter
    // animation". This is a vestibular-trigger effect, and the person who set
    // that switch set it for exactly this; the only correct amount is none.
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
    // There is no cursor to follow, so the loop would burn battery drawing
    // particles at wherever a finger last lifted.
    stubMedia(["pointer: coarse"]);
    stubFrames();

    render(<BackgroundLayer />);

    expect(screen.queryByTestId("particle-canvas")).not.toBeInTheDocument();
  });
});

describe("the cap", () => {
  it("never holds more than MAX_PARTICLES, however hard it is driven", () => {
    // The cap is the only thing standing between this and an unbounded array
    // that grows for as long as the tab is open. Driven far past it — ten
    // thousand pixels of travel is a few seconds of an energetic cursor.
    const field = new ParticleField(() => 0.5);

    for (let i = 0; i < 1000; i += 1) {
      field.push(i * 10, 0);
      expect(field.particles.length).toBeLessThanOrEqual(MAX_PARTICLES);
    }

    expect(field.particles.length).toBe(MAX_PARTICLES);

    // And direct emission, which is the path the cap actually guards.
    for (let i = 0; i < 500; i += 1) {
      field.emit(i, i);
      expect(field.particles.length).toBeLessThanOrEqual(MAX_PARTICLES);
    }
  });

  it("emits on distance travelled rather than per frame", () => {
    // A still cursor must cost nothing. Two events at the same point is not
    // movement, whatever the frame rate is.
    const field = new ParticleField(() => 0.5);

    field.push(10, 10);
    field.push(10, 10);
    expect(field.particles).toHaveLength(0);

    // 200px of travel at one per 40px.
    field.push(210, 10);
    expect(field.particles).toHaveLength(5);
  });
});

describe("the idle loop", () => {
  it("stops once movement and particles have both ceased, and restarts on the next move", () => {
    stubMedia();
    const frames = stubFrames();

    render(<BackgroundLayer />);
    expect(screen.getByTestId("particle-canvas")).toBeInTheDocument();

    // Nothing has moved yet, so nothing is scheduled: the resting cost of this
    // component on a page being read is one event listener.
    expect(frames.scheduled).toBe(false);

    frames.move(0, 400);
    frames.move(600, 400);
    expect(frames.scheduled).toBe(true);

    // Still running while the field is alive and the idle window is open.
    frames.run(1000);
    expect(frames.scheduled).toBe(true);

    // Past the longest particle lifetime and past the idle window, with no
    // further movement. A loop still scheduled here is one that would keep
    // waking the machine sixty times a second, forever, behind a static page.
    frames.run(4000);
    expect(frames.scheduled).toBe(false);

    // And it comes back.
    frames.move(610, 380);
    expect(frames.scheduled).toBe(true);
  });
});

/* --- The switch ----------------------------------------------------------- */

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

/* --- The cost in contrast -------------------------------------------------- */

/**
 * The fourth property, and the one that made the effect fainter than specified.
 *
 * A particle is ink laid over both the glyph and the paper under it, so it is
 * not only decoration — it spends contrast. `contrast.test.ts` proves the
 * palette clears WCAG AA; this proves it still clears it with the field's
 * darkest possible particle sitting on top, which is the only version of that
 * guarantee a reader actually experiences.
 *
 * Reads the tokens off disk for the same reason that file does: a palette
 * pasted in here would prove the numbers in this test are consistent with each
 * other, which is not the question.
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

  /** Source-over compositing of the particle's ink at `alpha`. */
  function veiled(colour: [number, number, number], alpha: number) {
    return colour.map((c, i) => c * (1 - alpha) + INK[i] * alpha) as [
      number,
      number,
      number,
    ];
  }

  function luminance([r, g, b]: [number, number, number]): number {
    const linear = [r, g, b].map((value) => {
      const unit = value / 255;
      return unit <= 0.03928 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  }

  function contrast(fg: [number, number, number], bg: [number, number, number]): number {
    const a = luminance(fg);
    const b = luminance(bg);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  }

  it("declares an ink that is the ink token, not black", () => {
    // `#000` on a warm white ground reads as a hole rather than as smoke, and
    // the sprite is built from whatever the token says at runtime — so the
    // fallback baked in here has to be the same colour the palette declares.
    expect(INK).toEqual(channels("ink"));
  });

  it.each([["ink"], ["ink-2"], ["ink-3"], ["accent"]])(
    "leaves %s above WCAG AA at the darkest a particle can be",
    (role) => {
      // `accent` is the binding one: 4.86:1 on the ground, so it clears AA by
      // 0.36 and an ink veil crosses the bar at α = 0.105. That measurement is
      // why ALPHA_MAX is 0.09 and not the 0.12 the effect was specified at.
      const alpha = ALPHA_MAX;
      const text = veiled(channels(role), alpha);
      const ground = veiled(channels("ground"), alpha);

      expect(contrast(text, ground)).toBeGreaterThanOrEqual(4.5);
    },
  );

  it("would fail at the opacity originally specified", () => {
    // Named so the reason lives in the suite rather than only in a commit
    // message. If somebody raises the ceiling back to 0.12, the row above goes
    // red and this one says what it cost.
    const text = veiled(channels("accent"), 0.12);
    const ground = veiled(channels("ground"), 0.12);

    expect(contrast(text, ground)).toBeLessThan(4.5);
    expect(ALPHA_MAX).toBeLessThan(0.105);
  });
});
