/**
 * The highlight, at its boundaries and against the real offset semantics.
 *
 * Two things are pinned here. The segmentation, whose edge cases — zero, the very
 * end, the whole string, inverted, out of range — are where an off-by-one marks
 * the wrong word and looks completely plausible. And the arithmetic that recovers
 * a chunk's own text from the text stored for it, which is the part that had to be
 * measured against the corpus rather than assumed: `char_start`..`char_end` tiles
 * the document contiguously, while `content` carries an extra borrowed prefix that
 * nothing records the length of.
 */

import { describe, expect, it } from "vitest";

import { overlapPrefixLength, ownSpan, segment } from "./highlight";

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

describe("recovering the chunk's own text", () => {
  it("finds no prefix on the first chunk of a memory", () => {
    // Chunk 0 borrows nothing, so its stored text is exactly its span. Measured:
    // true for every ordinal-0 chunk in the corpus.
    const text = "a".repeat(754);
    expect(overlapPrefixLength(text, 0, 754)).toBe(0);
    expect(ownSpan(text, 0, 754)).toEqual({ start: 0, end: 754 });
  });

  it("recovers the borrowed prefix on a later chunk", () => {
    // The real numbers from tests/integration/test_job_queue.py, chunk #1:
    // span 364 characters, 792 characters stored, so 428 are borrowed.
    const text = "x".repeat(792);
    expect(overlapPrefixLength(text, 754, 1118)).toBe(428);
    expect(ownSpan(text, 754, 1118)).toEqual({ start: 428, end: 792 });
  });

  it("puts the match at the end of the stored text, not the start", () => {
    // The consequence that shapes the whole component: what a reader's eye lands
    // on first is context borrowed from the previous chunk.
    const span = ownSpan("y".repeat(792), 754, 1118);
    expect(span?.start).toBeGreaterThan(0);
    expect(span?.end).toBe(792);
  });

  it("marks nothing when the offsets and the text disagree", () => {
    // A span longer than the text it describes cannot be positioned. Marking
    // nothing is honest; guessing would assert something false about what matched.
    expect(overlapPrefixLength("short", 0, 9999)).toBeNull();
    expect(ownSpan("short", 0, 9999)).toBeNull();
  });

  it("marks nothing for an empty or inverted span", () => {
    expect(overlapPrefixLength("some text", 100, 100)).toBeNull();
    expect(overlapPrefixLength("some text", 100, 50)).toBeNull();
  });

  it("composes with segment to mark exactly the chunk's own text", () => {
    const borrowed = "borrowed context from the previous chunk. ";
    const own = "the text that actually matched.";
    const text = borrowed + own;
    // Offsets as the database stores them: the span covers only `own`.
    const charStart = 1000;
    const charEnd = charStart + own.length;

    const span = ownSpan(text, charStart, charEnd);
    expect(span).not.toBeNull();
    const segments = segment(text, span!.start, span!.end);

    expect(segments).toEqual([
      { text: borrowed, marked: false },
      { text: own, marked: true },
    ]);
  });
});
