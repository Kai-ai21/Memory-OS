/**
 * How much a date is worth, and how to say so on screen.
 *
 * M1.1 stored `occurred_at_source` beside every timestamp precisely so this
 * question could be asked later. **A filesystem mtime and a Date header on an
 * email are not the same claim**, and an interface that renders them in the same
 * grey monospace is asserting that they are. Whatever the timeline shows is only
 * as good as the provenance underneath it, so the provenance travels with every
 * date rather than living in a legend somewhere.
 *
 * Three tiers, and the boundary between them is who is making the claim:
 *
 *   stated    `declared`, `parsed` — the source said when this happened, either
 *             in a field meant for it or in text we read. Wrong only if the
 *             source is wrong.
 *   inferred  `filesystem`, `inferred` — nobody said. We took it from metadata
 *             about the container or guessed it from context. An mtime is a fact
 *             about a file on a disk, not about the work in it: a fresh clone
 *             makes every date today.
 *   none      `unknown` — no date at all, which the corpus records honestly
 *             rather than defaulting. Never plotted, never in a bucket.
 *
 * The marker is a glyph rather than a colour alone. The one accent in this
 * interface is spent on the matched-span highlight, and a second colour scale
 * for confidence would compete with it — but more to the point, a colour-only
 * signal is invisible to a reader who cannot see the colour, and this one
 * changes what a number means.
 */

export type Confidence = "stated" | "inferred" | "none";

const TIERS: Record<string, Confidence> = {
  declared: "stated",
  parsed: "stated",
  filesystem: "inferred",
  inferred: "inferred",
  unknown: "none",
};

/** What tier a provenance value falls in. Unrecognised values are not trusted. */
export function confidence(provenance: string | null | undefined): Confidence {
  if (!provenance) return "none";
  // An unknown string is treated as inferred rather than stated. A new
  // provenance value should have to earn the higher tier by being added here.
  return TIERS[provenance] ?? "inferred";
}

/**
 * The glyph shown beside a date.
 *
 * `~` for approximate is the convention this borrows, and it reads as
 * "thereabouts" without a legend. Stated dates get nothing at all: the absence
 * of a mark is the strongest way to say "this one is not qualified", and marking
 * every date would make the mark invisible.
 */
export function marker(provenance: string | null | undefined): string {
  const tier = confidence(provenance);
  if (tier === "stated") return "";
  return tier === "none" ? "?" : "~";
}

/** The class that mutes a date the corpus is not sure about. */
export function dateClass(provenance: string | null | undefined): string {
  return confidence(provenance) === "stated" ? "text-ink" : "text-ink-2";
}

/** A sentence for the `title` attribute, so the mark is self-explaining. */
export function explain(provenance: string | null | undefined): string {
  switch (provenance) {
    case "declared":
      return "declared: the source stated this date in a field meant for it";
    case "parsed":
      return "parsed: read out of the content's own text";
    case "filesystem":
      return "filesystem: the file's mtime — when the bytes were last written to this disk, which is not when the work happened";
    case "inferred":
      return "inferred: guessed from context, not stated anywhere";
    case "unknown":
      return "unknown: no date at all. Not placed on the timeline, because an unknown date is not evidence of any date";
    default:
      return `${provenance ?? "unknown"}: unrecognised provenance, treated as low confidence`;
  }
}
