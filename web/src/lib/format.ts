/**
 * Formatting for an instrument rather than a consumer app.
 *
 * The rule throughout: never round away information the reader might be
 * comparing. Scores keep four decimals because two results can differ in the
 * third, and "0.78" twice in a column when the values are 0.7799 and 0.7812 is a
 * lie the interface is telling.
 */

/** A similarity, at the precision differences actually show up in. */
export function score(value: number): string {
  return value.toFixed(4);
}

/** A percentage for coverage figures, one decimal. */
export function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Thousands separators, because six-figure chunk counts are unreadable without. */
export function count(value: number): string {
  return value.toLocaleString("en-US");
}

/**
 * A timestamp, in the reader's zone, without the noise.
 *
 * Absolute rather than relative: "3 days ago" is friendlier and useless for
 * correlating an `ingested_at` against a log line.
 */
export function timestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * A timestamp in UTC, for anywhere it is being read against a UTC boundary.
 *
 * The timeline needs this and the search results do not, which is the whole
 * distinction. `date_trunc` runs in UTC — deliberately, so the same corpus
 * buckets the same way on every machine — and the local-zone renderer above
 * then prints a memory in the `2026-08-07` bucket as "08 Aug 2026, 03:58"
 * anywhere east of Greenwich. Both are correct and together they look like a
 * bug, so the view that shows bucket boundaries shows their contents in the
 * same zone the boundaries were computed in.
 */
export function timestampUtc(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    hour12: false,
  });
}

/* --- Relative time --------------------------------------------------------
 *
 * **This is a reversal, and the reason it is safe is the `title`.** The note on
 * `timestamp` above argued for absolute times on the grounds that "3 days ago"
 * is useless for correlating an `ingested_at` against a log line, and that
 * argument was and is correct. What it got wrong was treating the two as a
 * choice: the absolute value does not have to be *displayed* to be available,
 * and every relative stamp this application renders carries the full timestamp
 * in its `title`. So the glanceable form is on screen and the correlatable one
 * is one hover away, which is strictly more than showing only the absolute.
 *
 * What that buys is the common case. Almost every date in this interface is
 * read as "how stale is this" — the last sync, the last time a source was
 * seen, when a decision was recorded — and answering it from an absolute
 * timestamp is arithmetic the reader does against today's date, every time,
 * for every row.
 *
 * `Intl.RelativeTimeFormat` rather than a date library. It is in every browser
 * this targets, it localises, and the alternative is 12KB to render eight
 * strings.
 */

/** The thresholds, largest unit first. Days stop at a week — see `relativeTime`. */
const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["day", 86_400],
  ["hour", 3_600],
  ["minute", 60],
];

const RELATIVE = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

/**
 * "2 hours ago", "yesterday", "3 days ago" — and a date once it is past a week.
 *
 * **A week is the cut-off and it is not arbitrary.** Relative time is worth
 * something while the reader can still hold the reference point: "3 days ago"
 * lands, "23 days ago" is a number they have to convert back into a date to do
 * anything with. Past seven days the absolute date is both shorter and more
 * useful, so that is what it returns.
 *
 * `numeric: "auto"` is what produces "yesterday" instead of "1 day ago", which
 * is the whole reason to use the platform formatter rather than a template.
 *
 * Future dates work by the same arithmetic and read as "in 2 hours". They are
 * rare here but not impossible — a review due date is one — and a formatter
 * that silently rendered them as past would be worse than one that never saw
 * them.
 */
export function relativeTime(value: string | null | undefined, now: Date = new Date()): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;

  const seconds = (parsed.getTime() - now.getTime()) / 1000;
  const magnitude = Math.abs(seconds);

  // Beyond a week in either direction, the date itself. No time of day: at this
  // distance the hour is noise, and the `title` still carries it.
  if (magnitude >= 7 * 86_400) {
    return parsed.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
    });
  }

  // Under a minute is "now" rather than "in 3 seconds" or "3 seconds ago". The
  // precision is real and nobody wants it.
  if (magnitude < 60) return "just now";

  for (const [unit, size] of UNITS) {
    if (magnitude >= size) {
      // Truncated towards zero, so 90 minutes is "1 hour ago" rather than
      // "2 hours ago". Rounding up reports a thing as older than it is, and
      // for a freshness read that is the direction that misleads.
      return RELATIVE.format(Math.trunc(seconds / size), unit);
    }
  }

  return "just now";
}

/** A hash, shortened for display but never for comparison. */
export function shortHash(value: string | null | undefined, length = 12): string {
  if (!value) return "—";
  return value.slice(0, length);
}

/** `[1204, 1560]` — a character range, as the schema thinks of it. */
export function range(start: number, end: number): string {
  return `[${start}, ${end}]`;
}

/**
 * Whether a memory's content should be rendered as code or as prose.
 *
 * Drives which typeface the chunk text gets, which is the difference between a
 * readable document and a wall of monospace. Kind comes from the parser that read
 * the bytes, so it is more trustworthy than the file extension.
 */
export function isCode(kind: string | null | undefined): boolean {
  return kind === "code";
}


/**
 * A file size somebody can read.
 *
 * Binary units, because that is what a file manager shows — a number that
 * disagrees with the one beside it in Finder is a number that gets questioned.
 * One decimal below ten and none above, so the column stays narrow without
 * rounding 1.4MB to 1MB.
 */
export function fileSize(size: number): string {
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB"];
  let value = size / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}
