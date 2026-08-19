/**
 * The three things about the cursor mark that looking at it will not tell you.
 *
 * Each is a promise that fails silently and expensively: a canvas that ignores
 * the accessibility switch it exists to respect, a buffer with no ceiling, and
 * a blurred full-viewport redraw that keeps running behind a page nobody is
 * pointing at.
 *
 * **No real frame loop runs here.** `requestAnimationFrame` is a hand-cranked
 * queue driven off a virtual clock, so "the loop stops" is asserted on whether
 * a frame was *scheduled* — which is the actual claim, and which a timer in a
 * test suite could only approximate.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { BrushLayer, IDLE_MS } from "./BrushLayer";
import { BrushStroke, LIFETIME, MAX_POINTS } from "../lib/brush";

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

/** jsdom's `getContext` returns null, and the component correctly bails on that. */
function stubCanvas() {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    quadraticCurveTo: vi.fn(),
    stroke: vi.fn(),
    lineCap: "butt",
    lineJoin: "miter",
    lineWidth: 1,
    strokeStyle: "",
  } as unknown as CanvasRenderingContext2D);
}

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

beforeEach(() => stubCanvas());
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("reduced motion", () => {
  it("creates no canvas at all", () => {
    // Not a paused canvas and not a slower one. A dark stroke chasing the
    // pointer across the whole viewport is the most motion this application
    // produces, and this is the setting somebody turns on to be spared it.
    stubMedia(["prefers-reduced-motion"]);
    stubFrames();

    render(<BrushLayer feather={200} />);

    expect(screen.queryByTestId("brush-layer")).not.toBeInTheDocument();
  });

  it("renders the canvas when motion is allowed", () => {
    stubMedia();
    stubFrames();

    render(<BrushLayer feather={200} />);

    expect(screen.getByTestId("brush-layer")).toBeInTheDocument();
  });
});

describe("the position buffer", () => {
  it("never exceeds MAX_POINTS, however fast the mouse reports", () => {
    // A 1000Hz mouse produces fifteen hundred samples inside the 1.5s window.
    // Without the cap that is fifteen hundred stroke calls a frame for a mark
    // that is sixty segments long to look at.
    const stroke = new BrushStroke();

    for (let i = 0; i < 5000; i += 1) {
      // Small steps, so nothing trips the teleport guard and clears the buffer.
      stroke.push(100 + (i % 50), 100 + (i % 37), i);
      expect(stroke.points.length).toBeLessThanOrEqual(MAX_POINTS);
    }
    expect(stroke.points.length).toBe(MAX_POINTS);
  });

  it("drops points once they are older than the window", () => {
    // The other half of "bounded": the cap limits a fast mouse, and this limits
    // a slow one. A stroke that stops being pruned stops fading.
    const stroke = new BrushStroke();
    for (let i = 0; i < 10; i += 1) stroke.push(i * 10, 0, i * 100);
    expect(stroke.points.length).toBe(10);

    stroke.prune(900 + LIFETIME + 1);
    expect(stroke.points).toHaveLength(0);
    expect(stroke.active).toBe(false);
  });

  it("starts a new stroke rather than joining across a teleport", () => {
    // A tab switch is not a gesture, and a line drawn between where the pointer
    // left and where it came back crosses a screen it never touched.
    const stroke = new BrushStroke();
    stroke.push(10, 10, 0);
    stroke.push(20, 20, 16);
    expect(stroke.points).toHaveLength(2);

    stroke.push(1400, 900, 32);
    expect(stroke.points).toHaveLength(1);
  });
});

describe("the loop", () => {
  it("stops after the idle timeout and restarts on movement", () => {
    stubMedia();
    const frames = stubFrames();

    render(<BrushLayer feather={200} />);
    expect(screen.getByTestId("brush-layer")).toBeInTheDocument();

    // Nothing has moved, so nothing is scheduled: a blurred full-viewport
    // redraw every frame with no stroke on it is the cost this avoids.
    expect(frames.scheduled).toBe(false);

    frames.move(100, 100);
    frames.move(400, 300);
    expect(frames.scheduled).toBe(true);

    // Still running while the stroke is alive and the idle window is open.
    frames.run(LIFETIME / 2);
    expect(frames.scheduled).toBe(true);

    // Past both the fade and the idle window, with no further movement.
    frames.run(IDLE_MS + LIFETIME);
    expect(frames.scheduled).toBe(false);

    frames.move(420, 320);
    expect(frames.scheduled).toBe(true);
  });

  it("tears the loop down with the component", () => {
    stubMedia();
    const frames = stubFrames();

    const view = render(<BrushLayer feather={200} />);
    frames.move(100, 100);
    expect(frames.scheduled).toBe(true);

    view.unmount();
    expect(frames.scheduled).toBe(false);
  });
});
