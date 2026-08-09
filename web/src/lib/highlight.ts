/**
 * Splitting text at a character range, for the matched-span highlight.
 *
 * This is the most important piece of rendering in the interface — the highlight
 * is what makes a result legible without reading it — and it is also the easiest
 * to get subtly wrong at the boundaries. So it is a pure function over strings
 * with no React in it, and the edge cases are tested rather than eyeballed.
 *
 * A note on what the offsets mean. `char_start`/`char_end` on a chunk index into
 * the *parent memory's* normalized text, not into the chunk's own text. When the
 * chunk text is the exact slice, the whole chunk is the match. When it is not —
 * the chunker adds an overlap prefix from the previous chunk — the offsets have to
 * be rebased before they mean anything locally, and `rebase` does that.
 */

export interface Segment {
  text: string;
  marked: boolean;
}

/**
 * Split `text` into marked and unmarked segments at `[start, end)`.
 *
 * Offsets are clamped into range and an inverted or empty range yields one
 * unmarked segment. Silently rendering nothing, or throwing, would both be worse
 * than showing the text unhighlighted: the text is the thing the user came for,
 * and the highlight is an aid to reading it.
 */
export function segment(text: string, start: number, end: number): Segment[] {
  if (text.length === 0) return [];

  const from = clamp(start, 0, text.length);
  const to = clamp(end, 0, text.length);
  if (to <= from) return [{ text, marked: false }];

  const segments: Segment[] = [];
  // Leading and trailing segments are omitted when empty rather than emitted as
  // empty strings — a range covering the whole text should produce exactly one
  // marked segment, which is what makes the tests here readable.
  if (from > 0) segments.push({ text: text.slice(0, from), marked: false });
  segments.push({ text: text.slice(from, to), marked: true });
  if (to < text.length) segments.push({ text: text.slice(to), marked: false });
  return segments;
}

/**
 * Translate document offsets into offsets within `chunkText`.
 *
 * The chunk's text usually appears verbatim in the document at `char_start`, but
 * not always: normalization and the chunker's overlap prefix mean the two can
 * disagree. So the text is located by search first and only then by arithmetic,
 * and if neither works the caller is told there is nothing to mark rather than
 * being handed a plausible-looking wrong range.
 */
export function rebase(
  documentText: string,
  chunkText: string,
  charStart: number,
  charEnd: number,
): { start: number; end: number } | null {
  if (!chunkText) return null;

  // Where does this chunk actually sit in the document? Searching from a little
  // before the recorded offset finds it even when normalization shifted things,
  // without matching an identical passage elsewhere in the file.
  const searchFrom = Math.max(0, charStart - chunkText.length);
  let offset = documentText.indexOf(chunkText, searchFrom);
  if (offset < 0) offset = documentText.indexOf(chunkText);
  if (offset < 0) return null;

  const start = charStart - offset;
  const end = charEnd - offset;
  if (end <= 0 || start >= chunkText.length) return null;
  return { start: Math.max(0, start), end: Math.min(chunkText.length, end) };
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
