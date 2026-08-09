/**
 * The highlight, at its boundaries.
 *
 * This is the most important rendering in the interface and the easiest to get
 * subtly wrong: an off-by-one at the start of a chunk marks the wrong word and
 * looks completely plausible. So the ranges that break naive implementations —
 * zero, the very end, the whole string, inverted, out of range — are each pinned.
 */

import { describe, expect, it } from "vitest";

import { rebase, segment } from "./highlight";

const TEXT = "the worker claims a job and holds a lease";

describe("segment", () => {
  it("marks a span in the middle", () => {
    expect(segment(TEXT, 4, 10)).toEqual([
      { text: "the ", marked: false },
      { text: "worker", marked: true },
      { text: " claims a job and holds a lease", marked: false },
    ]);
  });

  it("marks a span that starts at the very beginning", () => {
    // No empty leading segment: a range at 0 produces two parts, not three.
    expect(segment(TEXT, 0, 3)).toEqual([
      { text: "the", marked: true },
      { text: " worker claims a job and holds a lease", marked: false },
    ]);
  });

  it("marks a span that ends at the very end", () => {
    const start = TEXT.length - 5;
    expect(segment(TEXT, start, TEXT.length)).toEqual([
      { text: TEXT.slice(0, start), marked: false },
      { text: "lease", marked: true },
    ]);
  });

  it("marks the whole string as a single segment", () => {
    expect(segment(TEXT, 0, TEXT.length)).toEqual([{ text: TEXT, marked: true }]);
  });

  it("clamps a range that runs past the end", () => {
    // The offsets come from the database and the text from an API response; a
    // mismatch should degrade to marking what exists, not throw.
    expect(segment(TEXT, 36, 9999)).toEqual([
      { text: TEXT.slice(0, 36), marked: false },
      { text: "lease", marked: true },
    ]);
  });

  it("marks nothing for an inverted range", () => {
    expect(segment(TEXT, 10, 4)).toEqual([{ text: TEXT, marked: false }]);
  });

  it("marks nothing for an empty range", () => {
    expect(segment(TEXT, 5, 5)).toEqual([{ text: TEXT, marked: false }]);
  });

  it("handles negative offsets", () => {
    expect(segment(TEXT, -10, 3)).toEqual([
      { text: "the", marked: true },
      { text: " worker claims a job and holds a lease", marked: false },
    ]);
  });

  it("returns nothing for empty text", () => {
    expect(segment("", 0, 5)).toEqual([]);
  });

  it("loses no characters, whatever the range", () => {
    // The invariant that matters most: the reader must see the whole chunk. A
    // highlight is an aid to reading it, never a filter on it.
    for (const [start, end] of [
      [0, 0],
      [0, 1],
      [3, 9],
      [0, TEXT.length],
      [TEXT.length - 1, TEXT.length],
      [-5, 500],
      [20, 4],
    ]) {
      const joined = segment(TEXT, start, end)
        .map((part) => part.text)
        .join("");
      expect(joined).toBe(TEXT);
    }
  });
});

describe("rebase", () => {
  const document = `# Heading\n\n${TEXT}\n\nand some trailing prose`;
  const offset = document.indexOf(TEXT);

  it("translates document offsets onto the chunk", () => {
    // The document says [offset+4, offset+10]; within the chunk that is [4, 10].
    expect(rebase(document, TEXT, offset + 4, offset + 10)).toEqual({ start: 4, end: 10 });
  });

  it("finds the chunk even when the recorded offset has drifted", () => {
    // Normalization can shift offsets. The text is located by search first, so a
    // stale offset still resolves rather than marking the wrong span.
    expect(rebase(document, TEXT, offset + 4 + 3, offset + 10 + 3)).not.toBeNull();
  });

  it("clamps a range that extends past the chunk", () => {
    const result = rebase(document, TEXT, offset - 5, offset + 6);
    expect(result).toEqual({ start: 0, end: 6 });
  });

  it("returns null when the chunk is not in the document", () => {
    // Better than a plausible-looking wrong range.
    expect(rebase(document, "text that is not there", 0, 5)).toBeNull();
  });

  it("returns null when the range falls entirely outside the chunk", () => {
    expect(rebase(document, TEXT, 0, 3)).toBeNull();
  });

  it("returns null for empty chunk text", () => {
    expect(rebase(document, "", 0, 5)).toBeNull();
  });
});
