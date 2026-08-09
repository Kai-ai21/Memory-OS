/**
 * Chunk text with the matched span marked.
 *
 * The whole point of the interface: a result you can judge at a glance instead of
 * reading. Everything else on the screen is metadata about this.
 */

import { rebase, segment } from "../lib/highlight";

interface Props {
  /** The chunk's own text. */
  text: string;
  /** Offsets into the parent memory, as stored on the chunk. */
  charStart: number;
  charEnd: number;
  /**
   * The parent memory's normalized text, when it has been fetched. Without it the
   * offsets cannot be rebased onto the chunk, so nothing is marked — which is
   * correct: a highlight in the wrong place is worse than none.
   */
  documentText?: string | null;
  code?: boolean;
}

export function Highlighted({ text, charStart, charEnd, documentText, code }: Props) {
  const local = documentText ? rebase(documentText, text, charStart, charEnd) : null;
  const segments = local ? segment(text, local.start, local.end) : [{ text, marked: false }];

  return (
    <div className={code ? "code-content" : "prose-content"} data-testid="chunk-text">
      {segments.map((part, index) =>
        part.marked ? (
          <mark key={index} className="mark" data-testid="mark">
            {part.text}
          </mark>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </div>
  );
}
