/**
 * Reduced motion, and the two places CSS cannot reach.
 *
 * The blanket rule in `index.css` covers every keyframe and transition, so the
 * risk in this milestone is not the CSS — it is the two animations driven from
 * JavaScript, which no stylesheet can shorten: the staggered arrival's inline
 * `animation-delay`, and the counter's `requestAnimationFrame` loop. Both have
 * to ask, and both are asserted here.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { Arrive } from "../components/Arrive";
import { Count } from "../components/Count";
import { COUNT_LIMIT, STAGGER_CAP, staggerDelay } from "./motion";

/** Answer `prefers-reduced-motion` however the test needs. */
function stubReducedMotion(reduced: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: reduced && query.includes("prefers-reduced-motion"),
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      onchange: null,
      dispatchEvent: () => false,
    })),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("with reduced motion", () => {
  it("renders arriving content with no animation at all", () => {
    stubReducedMotion(true);
    render(
      <Arrive index={5}>
        <span>a result</span>
      </Arrive>,
    );

    /* Not merely a shortened duration. The stylesheet would collapse that, but
       not the `backwards` fill — which starts an item at `opacity: 0` and would
       leave it there for the 0.01ms the guard allows, flashing on every list.
       So the class is not applied and there is no delay to apply it with. */
    expect(screen.queryByTestId("arrive")).not.toBeInTheDocument();
    const rendered = screen.getByText("a result").parentElement!;
    expect(rendered.className).not.toContain("arrive");
    expect(rendered.style.animationDelay).toBe("");
  });

  it("keeps the animation when motion is allowed", () => {
    stubReducedMotion(false);
    render(
      <Arrive index={2}>
        <span>a result</span>
      </Arrive>,
    );

    const node = screen.getByTestId("arrive");
    expect(node.className).toContain("arrive");
    expect(node.style.animationDelay).toBe("60ms");
  });

  it("shows a number at its final value without counting to it", () => {
    stubReducedMotion(true);
    const frame = vi.spyOn(window, "requestAnimationFrame");

    const { rerender } = render(<Count value={10} />);
    rerender(<Count value={30} />);

    // Straight to the answer, and no animation loop was ever started.
    expect(screen.getByTestId("count")).toHaveTextContent("30");
    expect(frame).not.toHaveBeenCalled();
  });
});

describe("the stagger cap", () => {
  it("stops adding delay past the eighth item", () => {
    expect(staggerDelay(0)).toBe(0);
    expect(staggerDelay(3)).toBe(90);
    expect(staggerDelay(STAGGER_CAP)).toBe(240);

    // The row that matters. At 30ms uncapped, item 49 would arrive 1.47s late
    // — slower than no animation, on exactly the row somebody scrolled to find.
    expect(staggerDelay(49)).toBe(staggerDelay(STAGGER_CAP));
    expect(staggerDelay(49)).toBeLessThan(250);
  });
});

describe("count thresholds", () => {
  it("does not animate a jump of a hundred or more", () => {
    stubReducedMotion(false);
    const frame = vi.spyOn(window, "requestAnimationFrame");

    const { rerender } = render(<Count value={3833} />);
    frame.mockClear();
    rerender(<Count value={3833 + COUNT_LIMIT} />);

    // A blur that reads as a slot machine, and the reader waits for it to
    // settle before they can trust the figure.
    expect(frame).not.toHaveBeenCalled();
    expect(screen.getByTestId("count")).toHaveTextContent("3,933");
  });

  it("does not animate on first render", () => {
    stubReducedMotion(false);
    const frame = vi.spyOn(window, "requestAnimationFrame");

    render(<Count value={42} />);

    // Counting up from zero on load delays the first honest reading of every
    // figure on the screen.
    expect(frame).not.toHaveBeenCalled();
    expect(screen.getByTestId("count")).toHaveTextContent("42");
  });
});
