/**
 * The auto-scroll rule: follow the bottom, unless the reader left it.
 *
 * This is the assertion the feature exists for. An unconditional
 * `scrollIntoView` passes every test anybody would think to write — the
 * transcript does end up at the bottom — and is precisely the bug: scrolling up
 * to re-read an earlier answer is undone by the next token of the one still
 * streaming.
 *
 * jsdom has no layout, so `scrollHeight` and `clientHeight` are always 0 and
 * "near the bottom" is trivially true. The container's metrics are therefore
 * defined explicitly, which is the only way to express "the reader has scrolled
 * up" in this environment at all.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { useRef } from "react";

import { isNearBottom, STICK_THRESHOLD_PX, useStickToBottom } from "./scroll";

/** Give an element the layout jsdom will not. */
function withMetrics(
  element: HTMLElement,
  metrics: { scrollTop: number; scrollHeight: number; clientHeight: number },
) {
  Object.defineProperty(element, "scrollHeight", {
    value: metrics.scrollHeight,
    configurable: true,
  });
  Object.defineProperty(element, "clientHeight", {
    value: metrics.clientHeight,
    configurable: true,
  });
  Object.defineProperty(element, "scrollTop", {
    value: metrics.scrollTop,
    writable: true,
    configurable: true,
  });
}

/** A transcript in its own scroll container, with a growing body. */
function Transcript({ text }: { text: string }) {
  const bottom = useRef<HTMLDivElement>(null);
  useStickToBottom(bottom, [text]);
  return (
    <div data-testid="scroller" style={{ overflowY: "auto" }}>
      <p>{text}</p>
      <div ref={bottom} data-testid="anchor" />
    </div>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("isNearBottom", () => {
  it("tolerates the last pixel or two, and nothing more", () => {
    // Flush. Sub-pixel layout means this is never exactly zero in a browser.
    expect(isNearBottom({ scrollTop: 900, scrollHeight: 1000, clientHeight: 100 })).toBe(true);
    // Within the threshold: still following.
    expect(
      isNearBottom({ scrollTop: 900 - STICK_THRESHOLD_PX, scrollHeight: 1000, clientHeight: 100 }),
    ).toBe(true);
    // Past it: the reader has moved, and moved on purpose.
    expect(
      isNearBottom({ scrollTop: 900 - STICK_THRESHOLD_PX - 1, scrollHeight: 1000, clientHeight: 100 }),
    ).toBe(false);
  });
});

describe("auto-scroll", () => {
  it("follows the bottom while the reader is there", () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    const { rerender, getByTestId } = render(<Transcript text="one" />);
    const scroller = getByTestId("scroller");
    withMetrics(scroller, { scrollTop: 900, scrollHeight: 1000, clientHeight: 100 });

    act(() => {
      scroller.dispatchEvent(new Event("scroll"));
    });

    scrollIntoView.mockClear();
    rerender(<Transcript text="one two" />);

    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("does NOT fire once the reader has scrolled up", () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    const { rerender, getByTestId } = render(<Transcript text="one" />);
    const scroller = getByTestId("scroller");

    // The reader scrolls back to re-read something well above the bottom.
    withMetrics(scroller, { scrollTop: 100, scrollHeight: 1000, clientHeight: 100 });
    act(() => {
      scroller.dispatchEvent(new Event("scroll"));
    });

    scrollIntoView.mockClear();

    // More tokens arrive. The transcript must stay exactly where it was put.
    rerender(<Transcript text="one two" />);
    rerender(<Transcript text="one two three" />);

    expect(scrollIntoView).not.toHaveBeenCalled();
  });

  it("re-follows when the reader scrolls back down", () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    const { rerender, getByTestId } = render(<Transcript text="one" />);
    const scroller = getByTestId("scroller");

    withMetrics(scroller, { scrollTop: 100, scrollHeight: 1000, clientHeight: 100 });
    act(() => scroller.dispatchEvent(new Event("scroll")));
    scrollIntoView.mockClear();
    rerender(<Transcript text="one two" />);
    expect(scrollIntoView).not.toHaveBeenCalled();

    // Back to the bottom under their own steam: following resumes. Without
    // this the feature is a one-way latch and the transcript never catches up
    // again for the rest of the session.
    withMetrics(scroller, { scrollTop: 900, scrollHeight: 1000, clientHeight: 100 });
    act(() => scroller.dispatchEvent(new Event("scroll")));
    rerender(<Transcript text="one two three" />);

    expect(scrollIntoView).toHaveBeenCalled();
  });
});
