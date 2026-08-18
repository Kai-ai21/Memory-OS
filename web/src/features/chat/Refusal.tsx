/**
 * "The stored memories don't cover this", said as the most confident thing on
 * the screen.
 *
 * **This is the product's most distinctive behaviour and it should look
 * deliberate.** Every other system in this category answers anyway. The earlier
 * draft of this screen rendered a refusal in grey italic body text, half-hidden
 * behind the composer — the visual language of an apology, or of a loading
 * state, which is precisely wrong. A refusal is not a failure to answer. It is
 * the system reporting a measured fact about the corpus: nothing here supports a
 * claim, so no claim is made.
 *
 * So it gets the loudest treatment available short of colour-as-alarm: a warn
 * rule (this palette's colour for what the system does not have), a mono label
 * above it naming the finding rather than the mood, and the sentence itself at
 * full body size in white — the same size as an answer, because it *is* the
 * answer.
 *
 * The label is `NO SUPPORTING MEMORIES` and not "declined" or "sorry". "Declined"
 * describes the system's behaviour; this names what was actually found, which is
 * the part a reader can do something about — add a source, or ask a different
 * question.
 *
 * Shared by the streamed path and the stored-transcript path deliberately. Two
 * copies of a refusal treatment is two places for one of them to soften.
 */

import type { ReactNode } from "react";

export function Refusal({
  children,
  footnote,
}: {
  /** The refusal itself, in the API's own words. Never reworded here. */
  children: ReactNode;
  /** Anything after it — the citation note, the model id. */
  footnote?: ReactNode;
}) {
  return (
    <div className="relative pl-6" data-testid="refusal">
      {/* A solid rule, full height. The dark theme faded it downward and lit it
          with a glow; on light a fading rule just looks like a rendering error
          and there is no glow to spend, so it is a plain 2px bar in `warn` —
          which is more emphatic here, not less. */}
      <span className="absolute inset-y-0 left-0 w-0.5 bg-warn" aria-hidden />
      <p className="meta-label mb-2 font-bold text-warn" data-testid="refusal-label">
        No supporting memories
      </p>
      <p className="prose-content text-base text-ink" data-testid="refusal-text">
        {children}
      </p>
      {footnote ? <div className="mt-2">{footnote}</div> : null}
    </div>
  );
}
