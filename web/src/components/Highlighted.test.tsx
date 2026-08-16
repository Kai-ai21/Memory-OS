/**
 * The highlight, as it renders.
 *
 * `lib/highlight.test.ts` pins the arithmetic; this pins what reaches the DOM,
 * and they fail for different reasons. The arithmetic can be perfect while the
 * component marks the wrong element, drops the lead-in, or truncates through
 * the match — and the borrowed-prefix handling from M1.4a is the piece most
 * likely to be lost in a redesign, because it looks like styling. It is not:
 * 28.1% of stored chunk text in this corpus is borrowed from the previous
 * chunk, and rendering it as part of the match asserts something false about
 * what matched.
 */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { Highlighted } from "./Highlighted";

const BORROWED = "borrowed context from the previous chunk. ";
const OWN = "a worker claims a job and holds a lease on it";

describe("marking the chunk's own span", () => {
  it("marks the tail and mutes the borrowed lead-in", () => {
    // Offsets as the database stores them: the span covers only the tail, and
    // the stored text is longer by however much was borrowed.
    render(
      <Highlighted
        text={BORROWED + OWN}
        charStart={1000}
        charEnd={1000 + OWN.length}
      />,
    );

    expect(screen.getByTestId("mark")).toHaveTextContent(OWN);
    expect(screen.getByTestId("lead-in")).toHaveTextContent("borrowed context");
  });

  it("marks the whole text when nothing was borrowed", () => {
    // Ordinal 0 borrows nothing, so its stored text is exactly its span.
    render(<Highlighted text={OWN} charStart={0} charEnd={OWN.length} />);

    expect(screen.getByTestId("mark")).toHaveTextContent(OWN);
    expect(screen.queryByTestId("lead-in")).not.toBeInTheDocument();
  });

  it("marks nothing when the offsets and the text disagree", () => {
    // A span longer than the text it describes cannot be positioned. Showing
    // the text unmarked is honest; guessing would put the highlight on the
    // wrong words and look entirely plausible.
    render(<Highlighted text="short" charStart={0} charEnd={9999} />);

    expect(screen.queryByTestId("mark")).not.toBeInTheDocument();
    expect(screen.getByTestId("chunk-text")).toHaveTextContent("short");
  });

  it("never drops characters from the text it was given", () => {
    // The invariant that matters most: the highlight is an aid to reading the
    // chunk, never a filter on it.
    const text = BORROWED + OWN;
    render(<Highlighted text={text} charStart={1000} charEnd={1000 + OWN.length} full />);

    expect(screen.getByTestId("chunk-text").textContent).toBe(text);
  });

  it("takes offsets into the excerpt when they are absolute", () => {
    // A citation's offsets are already relative to the window the API built,
    // and redoing the borrowed-prefix arithmetic on them is exactly the bug
    // M1.4a fixed. `absolute` is the UI declining to recompute.
    render(<Highlighted text="alpha beta gamma" charStart={6} charEnd={10} absolute />);

    expect(screen.getByTestId("mark")).toHaveTextContent("beta");
  });
});

describe("clamping", () => {
  it("clamps tightly in a list and not at all when expanded", () => {
    // The density fix in M9.0. Collapsed, results are compared against each
    // other and must fit together; expanded is the request to read one.
    const { rerender } = render(
      <Highlighted text={OWN} charStart={0} charEnd={OWN.length} tight />,
    );
    expect(screen.getByTestId("chunk-text").className).toContain("clamped-tight");

    rerender(<Highlighted text={OWN} charStart={0} charEnd={OWN.length} tight full />);
    expect(screen.getByTestId("chunk-text").className).not.toContain("clamped");
  });

  it("truncates the lead-in from the left, so the eye lands on the mark", () => {
    // A lead-in can be longer than the match. Keeping its head would push the
    // highlight past the clamp, and the most important element on the screen
    // would never be visible.
    const long = "x".repeat(400);
    render(
      <Highlighted text={long + OWN} charStart={1000} charEnd={1000 + OWN.length} />,
    );

    expect(screen.getByTestId("chunk-text")).toHaveTextContent(/^…/);
    expect(screen.getByTestId("mark")).toHaveTextContent(OWN);
  });
});
