/**
 * A timestamp that says how much it is worth.
 *
 * Used everywhere a date appears from here on. The mark is the point: an mtime
 * and a date an email declared are different claims, and rendering them
 * identically is the interface asserting they are the same. See
 * `lib/provenance.ts` for the tiers.
 *
 * **M9.10 makes the displayed form relative and keeps the absolute one in the
 * `title`.** "3 days ago" is what the reader is almost always actually asking
 * for, and the exact timestamp is still there on hover for the times they are
 * not — see the note above `relativeTime`. The two facts the title carries are
 * joined with an em dash: the timestamp first, because it is the one being
 * looked up, and the provenance explanation after it.
 *
 * `utc` is the exception and stays absolute. It is set only where a date is
 * being read against a UTC bucket boundary, and a relative stamp cannot express
 * a boundary at all.
 */

import { relativeTime, timestamp, timestampUtc } from "../lib/format";
import { dateClass, explain, marker } from "../lib/provenance";

export function DateStamp({
  value,
  provenance,
  showProvenance = false,
  utc = false,
}: {
  value: string | null | undefined;
  provenance: string | null | undefined;
  /** Spell the provenance out in full, for views with room for it. */
  showProvenance?: boolean;
  /**
   * Render in UTC rather than the reader's zone.
   *
   * Set wherever the date is being read against a UTC bucket boundary. See
   * `timestampUtc` — the two zones are each right on their own and look like an
   * off-by-one when a bar labelled `2026-08-07` lists a memory dated the 8th.
   */
  utc?: boolean;
}) {
  const mark = marker(provenance);
  const provenanceNote = explain(provenance);

  /* Both facts, or whichever one there is. A title of "— mtime of the file"
     with a leading dash is what naive concatenation gives when a date is
     missing, and it reads as a bug. */
  const exact = value ? (utc ? timestampUtc(value) : timestamp(value)) : "";
  const title = [exact, provenanceNote].filter(Boolean).join(" — ");

  // No date at all. Says so as a word rather than as an em dash, because "—"
  // in a column of timestamps reads as a formatting artefact and this is a
  // fact about the corpus: the date is not known and was not invented.
  if (!value) {
    return (
      <span
        className="meta text-ink-3"
        title={title}
        data-testid="date-stamp"
        data-provenance={provenance ?? "unknown"}
      >
        no date{mark}
      </span>
    );
  }

  return (
    <span
      className={`meta ${dateClass(provenance)}`}
      title={title}
      data-testid="date-stamp"
      data-provenance={provenance ?? "unknown"}
    >
      {/* UTC stays absolute — see the header. */}
      {utc ? timestampUtc(value) : relativeTime(value)}
      {mark ? <span className="text-accent">{mark}</span> : null}
      {showProvenance ? <span className="ml-1.5 text-ink-3">{provenance}</span> : null}
    </span>
  );
}
