/**
 * A date, as the distance from now, with the exact value on hover.
 *
 * The counterpart to `DateStamp`. That component exists because a date with a
 * *provenance* claim attached has to render the claim as well; this one is for
 * the many places where the date is simply a date — when a session was last
 * active, when a source last synced, when a suggestion arrived — and the only
 * question being asked of it is how long ago that was.
 *
 * **The `title` is not optional and is the reason this is safe.** Rendering
 * "3 days ago" and nothing else destroys information: a sync time that has to
 * be matched against a log line needs the timestamp, and the reader has no way
 * to recover it. With the absolute value on the element, the relative form is
 * a summary of something still present rather than a replacement for it. See
 * the note above `relativeTime` in `lib/format`.
 *
 * A `<time>` element rather than a `<span>`, with `dateTime` — the machine
 * readable value belongs in the markup, and this is the element the platform
 * provides for exactly this.
 */

import { relativeTime, timestamp } from "../lib/format";

export function RelativeTime({
  value,
  className = "",
}: {
  value: string | null | undefined;
  className?: string;
}) {
  // Nothing to be relative to. Rendered as a plain span: a `<time>` with no
  // `datetime` is invalid, and an em dash is not a time.
  if (!value) {
    return (
      <span className={className} data-testid="relative-time">
        —
      </span>
    );
  }

  return (
    <time
      dateTime={value}
      title={timestamp(value)}
      className={className}
      data-testid="relative-time"
    >
      {relativeTime(value)}
    </time>
  );
}
