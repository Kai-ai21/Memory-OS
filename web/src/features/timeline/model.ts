/**
 * The chart's non-visual decisions, out of the components that render them.
 *
 * Split out for the reason `searchParams.ts` is: a module that exports both a
 * component and a helper breaks fast refresh, and — more usefully — these are
 * the parts worth testing without a DOM. The bucket label is the axis, and an
 * axis that disagrees with the range a click fetches is the bug this file exists
 * to make checkable.
 */

import type { Bucket, Period } from "../../api/client";

/** The order kinds stack in, bottom first. Stable, so bars are comparable. */
export const KIND_ORDER = [
  "code",
  "note",
  "document",
  "email",
  "commit",
  "meeting",
  "bookmark",
  "other",
];

const KIND_COLOR: Record<string, string> = {
  code: "var(--color-kind-code)",
  note: "var(--color-kind-note)",
  document: "var(--color-kind-document)",
  email: "var(--color-kind-email)",
  commit: "var(--color-kind-commit)",
  meeting: "var(--color-kind-meeting)",
  bookmark: "var(--color-kind-bookmark)",
  other: "var(--color-kind-other)",
};

export function kindColor(kind: string): string {
  return KIND_COLOR[kind] ?? KIND_COLOR.other;
}

/**
 * Kinds present anywhere in the window, in stack order, for the legend.
 *
 * Anywhere rather than per bucket: a legend that changed as you changed the
 * grain would make two charts of the same corpus unreadable against each other.
 */
export function kindsPresent(buckets: Bucket[]): string[] {
  const seen = new Set<string>();
  for (const bucket of buckets) {
    for (const kind of Object.keys(bucket.by_kind ?? {})) seen.add(kind);
  }
  return [
    ...KIND_ORDER.filter((kind) => seen.has(kind)),
    ...[...seen].filter((kind) => !KIND_ORDER.includes(kind)).sort(),
  ];
}

/**
 * `2026-08` for a month, `2026-08-07` for a day or a week.
 *
 * Sliced off the ISO string rather than passed through `Date`, deliberately. The
 * API sends bucket boundaries already truncated in UTC, and rendering them in
 * the reader's zone would shift a month bar labelled `2026-08` to July for
 * anybody west of Greenwich — the label would then disagree with the range the
 * click sends back, which is the same instant either way.
 */
export function bucketLabel(bucket: Bucket, period: Period): string {
  const iso = bucket.start.slice(0, 10);
  return period === "month" ? iso.slice(0, 7) : iso;
}

/** The date half of an ISO instant. */
export function isoDate(value: string): string {
  return value.slice(0, 10);
}

/** How many days a bucket spans. 31 for January, 28 for February, 1 for a day. */
export function bucketDays(bucket: Bucket): number {
  return (Date.parse(bucket.end) - Date.parse(bucket.start)) / 86_400_000;
}
