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
