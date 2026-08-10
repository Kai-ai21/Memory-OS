/**
 * Chunk text, with the chunk's own span marked and its borrowed lead-in muted.
 *
 * The whole point of the interface: a result you can judge at a glance instead of
 * reading. See `lib/highlight.ts` for why the mark is the *tail* of the stored
 * text — the head is context borrowed from the previous chunk, which is 28.1% of
 * all stored chunk text in this corpus and would otherwise read as the match.
 *
 * The lead-in is truncated from the left, because it can be longer than the match
 * itself and the reader's eye should land on the highlight rather than travel to
 * it.
 */

import { ownSpan, segment } from "../lib/highlight";

/**
 * How much borrowed context to show before the match.
 *
 * Small on purpose. Measured at 180 characters the lead-in filled the clamped
 * height on its own for code chunks — short lines, many of them — so the fade cut
 * off before reaching the highlight and the most important element on the screen
 * was never visible. Enough to orient, not to read.
 */
const MAX_LEAD_IN = 90;

interface Props {
  text: string;
  charStart: number;
  charEnd: number;
  code?: boolean;
  /** Show the whole lead-in and the full height rather than clamping. */
  full?: boolean;
  /**
   * `charStart`/`charEnd` already index into `text`.
   *
   * True for a citation excerpt, where the API located the span inside the
   * window it built and there is no borrowed prefix to recover — the whole
   * point of sending those offsets is that the UI does not redo the arithmetic
   * that M1.4a got wrong.
   */
  absolute?: boolean;
}

export function Highlighted({ text, charStart, charEnd, code, full, absolute }: Props) {
  const span = absolute
    ? { start: charStart, end: charEnd }
    : ownSpan(text, charStart, charEnd);
  const className = `${code ? "code-content" : "prose-content"}${full ? "" : " clamped"}`;

  if (!span) {
    // The offsets and the text disagree. Show the text, mark nothing.
    return (
      <div className={className} data-testid="chunk-text">
        {text}
      </div>
    );
  }

  const trimmed = !full && span.start > MAX_LEAD_IN;
  const from = trimmed ? span.start - MAX_LEAD_IN : 0;
  const visible = text.slice(from);
  const segments = segment(visible, span.start - from, span.end - from);

  return (
    <div className={className} data-testid="chunk-text">
      {trimmed ? <span className="text-faint">… </span> : null}
      {segments.map((part, index) =>
        part.marked ? (
          <mark key={index} className="mark" data-testid="mark">
            {part.text}
          </mark>
        ) : (
          // The borrowed lead-in, de-emphasised: present for orientation, and
          // visibly not the thing that matched.
          <span key={index} className="text-faint" data-testid="lead-in">
            {part.text}
          </span>
        ),
      )}
    </div>
  );
}
