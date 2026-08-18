/**
 * One item's versions, on a line, with both clocks and what actually changed.
 *
 * The list underneath already says which versions exist. What it cannot say is
 * *how they are spaced* — five revisions in an hour and five over a year are the
 * same five rows — and spacing is the whole reason to draw a line instead of a
 * table.
 *
 * **Two markers per version, because there are two clocks.** A hollow tick where
 * the version occurred, a filled one where it was ingested, joined by a rule.
 * That rule is the lag M4.0's `out_of_order` measures, made visible per item: a
 * long one means the content existed well before this system learned about it.
 *
 * **What changed** is read off the two hashes, which the corpus already stores
 * for exactly this. Bytes differing with normalized text identical is a real and
 * frequent case — a trailing newline, a line ending — and calling that "changed"
 * without qualification would send somebody looking for an edit that is not
 * there.
 */

import type { MemoryVersion } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { shortHash } from "../../lib/format";

export function VersionTimeline({ versions }: { versions: MemoryVersion[] }) {
  // Oldest first: a timeline reads left to right, and the API returns newest
  // first because the list below it wants the current version at the top.
  const ordered = [...versions].sort((a, b) => a.version - b.version);

  const stamps = ordered.flatMap((version) =>
    [version.occurred_at, version.ingested_at].filter((value): value is string => !!value),
  );
  const times = stamps.map((value) => Date.parse(value));
  const first = Math.min(...times);
  const last = Math.max(...times);
  // A single version, or several that all landed in the same instant, has no
  // extent to scale against. Everything sits at the left rather than dividing
  // by zero and landing at NaN%.
  const span = last > first ? last - first : 0;
  const at = (value: string | null | undefined): number =>
    !value || span === 0 ? 0 : ((Date.parse(value) - first) / span) * 100;

  if (ordered.length === 0) return null;

  return (
    <div className="flex flex-col gap-2" data-testid="version-timeline">
      {/* The rail. One row per version so lines never overlap — an item with
          forty versions is a scroll, not a pile. */}
      <div className="flex flex-col gap-1 border border-rule bg-raised px-3 py-2">
        {ordered.map((version) => {
          const occurred = at(version.occurred_at);
          const ingested = at(version.ingested_at);
          const left = Math.min(occurred, ingested);
          const width = Math.abs(ingested - occurred);
          return (
            <div key={version.id} className="flex items-center gap-2">
              <span className="meta w-8 shrink-0 text-faint">v{version.version}</span>
              <div className="relative h-4 flex-1" data-testid="version-rail">
                {/* The lag, as a length. */}
                <span
                  className="absolute top-2 h-px bg-rule-strong"
                  style={{ left: `${left}%`, width: `${width}%` }}
                  aria-hidden="true"
                />
                <span
                  className="absolute top-1 -ml-1 h-2 w-2 rounded-full border border-edge bg-void"
                  style={{ left: `${occurred}%` }}
                  title={`occurred ${version.occurred_at ?? "unknown"} (${version.occurred_at_source})`}
                  data-testid="occurred-tick"
                />
                <span
                  className="absolute top-1 -ml-1 h-2 w-2 rounded-full bg-edge"
                  style={{ left: `${ingested}%` }}
                  title={`ingested ${version.ingested_at}`}
                  data-testid="ingested-tick"
                />
              </div>
              <span className="meta w-24 shrink-0 text-right">
                {changed(version, ordered)}
              </span>
            </div>
          );
        })}
      </div>

      <div className="meta flex flex-wrap items-baseline gap-x-4 text-faint">
        <span className="inline-flex items-baseline gap-1">
          <span className="inline-block h-2 w-2 rounded-full border border-edge bg-void" />
          occurred
        </span>
        <span className="inline-flex items-baseline gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-edge" />
          ingested
        </span>
        <span>the rule between them is the lag</span>
      </div>

      {/* The same versions as rows, since a tick cannot carry a hash. */}
      <ul>
        {ordered.map((version) => (
          <li
            key={version.id}
            className="flex flex-wrap items-baseline gap-x-4 border-b border-rule/60 py-1"
          >
            <span className="meta w-8 text-ink">v{version.version}</span>
            <span className="w-44 shrink-0">
              <DateStamp
                value={version.occurred_at}
                provenance={version.occurred_at_source}
              />
            </span>
            <span className="meta w-44 shrink-0 text-faint">
              ingested <DateStamp value={version.ingested_at} provenance="declared" />
            </span>
            <span className="meta text-faint" title={version.content_hash}>
              {shortHash(version.content_hash, 8)}
            </span>
            <span className="meta text-muted">{changed(version, ordered)}</span>
            {version.is_current ? <span className="meta text-accent">current</span> : null}
            {version.deleted_at ? <span className="meta text-deny">tombstoned</span> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * What this version changed, relative to the one before it.
 *
 * The three answers are genuinely different and the middle one is the one worth
 * having a phrase for: the file was rewritten and the corpus did not change,
 * which is what `normalized_hash` exists to detect and why re-chunking was
 * skipped.
 */
function changed(version: MemoryVersion, ordered: MemoryVersion[]): string {
  const index = ordered.findIndex((candidate) => candidate.id === version.id);
  if (index <= 0) return "first version";
  const previous = ordered[index - 1];
  if (version.normalized_hash && version.normalized_hash !== previous.normalized_hash) {
    return "text changed";
  }
  if (version.content_hash !== previous.content_hash) return "bytes only";
  return "no change";
}
