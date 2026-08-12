/**
 * Locating a chunk's own text inside the text that is stored for it.
 *
 * This is the most important piece of rendering in the interface, and getting it
 * right required first working out what the stored offsets actually mean. They do
 * not mean what M1.4's docstring claims ("`char_start`/`char_end` index exactly
 * into the stored text"). Measured across all 934 chunks of this repository:
 *
 *   chunk #1  char_start=754  char_end=1118  span=364  len(content)=792
 *
 * `char_start`..`char_end` is a *contiguous, non-overlapping tiling* of the
 * document — chunk N ends exactly where N+1 begins, for every chunk in the
 * corpus. But `content` is longer than that span, because the chunker prepends an
 * overlap prefix borrowed from the previous chunk so that a concept crossing a
 * boundary appears whole in one chunk. Nothing records how long that prefix is.
 *
 * It is recoverable, exactly:
 *
 *   prefix = len(content) - (char_end - char_start)
 *
 * and `content === document[char_start - prefix : char_end]`, verified for 934 of
 * 934 chunks with zero failures. So the chunk's authoritative text is the *tail*
 * of what is stored, and the borrowed context is the head.
 *
 * Two consequences the interface is built around. The highlight needs no document
 * fetch — it is arithmetic on values the search response already carries, so a
 * result is legible the instant it arrives. And the borrowed prefix is rendered as
 * de-emphasised lead-in rather than as part of the match, because reading it as
 * the match is exactly the misattribution these offsets invite: 28.1% of all
 * stored chunk text is borrowed from a neighbour.
 */

export interface Segment {
  text: string;
  marked: boolean;
}

/**
 * Split `text` into marked and unmarked segments at `[start, end)`.
 *
 * Offsets are clamped into range, and an inverted or empty range yields one
 * unmarked segment. Rendering nothing, or throwing, would both be worse than
 * showing the text unhighlighted: the text is what the reader came for, and the
 * highlight is an aid to reading it.
 */
export function segment(text: string, start: number, end: number): Segment[] {
  if (text.length === 0) return [];

  const from = clamp(start, 0, text.length);
  const to = clamp(end, 0, text.length);
  if (to <= from) return [{ text, marked: false }];

  const segments: Segment[] = [];
  // Empty leading and trailing segments are omitted rather than emitted, so a
  // range covering the whole text produces exactly one marked segment.
  if (from > 0) segments.push({ text: text.slice(0, from), marked: false });
  segments.push({ text: text.slice(from, to), marked: true });
  if (to < text.length) segments.push({ text: text.slice(to), marked: false });
  return segments;
}

/**
 * Split `text` at several ranges at once, for a diff rather than a match.
 *
 * `segment` above marks one span because a search result has one matched span.
 * A version diff has many, and running `segment` per span and concatenating
 * would double-count every unmarked stretch between two of them.
 *
 * Ranges are sorted and clipped, and overlapping ones are merged rather than
 * rejected. Two marks touching the same character would nest `<mark>` inside
 * `<mark>`, which renders as a darker patch that means nothing — merging says
 * the same thing without inventing a third emphasis level.
 */
export function segmentAll(
  text: string,
  ranges: { start: number; end: number }[],
): Segment[] {
  if (text.length === 0) return [];

  const clipped = ranges
    .map((range) => ({
      start: clamp(range.start, 0, text.length),
      end: clamp(range.end, 0, text.length),
    }))
    .filter((range) => range.end > range.start)
    .sort((a, b) => a.start - b.start);

  if (clipped.length === 0) return [{ text, marked: false }];

  const merged: { start: number; end: number }[] = [];
  for (const range of clipped) {
    const last = merged[merged.length - 1];
    if (last && range.start <= last.end) last.end = Math.max(last.end, range.end);
    else merged.push({ ...range });
  }

  const segments: Segment[] = [];
  let cursor = 0;
  for (const range of merged) {
    if (range.start > cursor) {
      segments.push({ text: text.slice(cursor, range.start), marked: false });
    }
    segments.push({ text: text.slice(range.start, range.end), marked: true });
    cursor = range.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), marked: false });
  return segments;
}

/**
 * How much of a chunk's stored text is borrowed from the previous chunk.
 *
 * Returns null when the arithmetic cannot hold — a span longer than the text
 * means the two disagree, and marking nothing is the honest answer. Never
 * guesses: a highlight in the wrong place is worse than no highlight, because it
 * asserts something false about which text matched.
 */
export function overlapPrefixLength(
  text: string,
  charStart: number,
  charEnd: number,
): number | null {
  const span = charEnd - charStart;
  if (span <= 0) return null;
  const prefix = text.length - span;
  if (prefix < 0 || prefix > text.length) return null;
  return prefix;
}

/**
 * The chunk's own span within its stored text: everything after the borrowed
 * prefix.
 */
export function ownSpan(
  text: string,
  charStart: number,
  charEnd: number,
): { start: number; end: number } | null {
  const prefix = overlapPrefixLength(text, charStart, charEnd);
  if (prefix === null) return null;
  return { start: prefix, end: text.length };
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
