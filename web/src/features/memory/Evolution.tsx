/**
 * What changed between two versions, side by side.
 *
 * **Span by span, not file by file.** The obvious rendering is two columns each
 * holding a whole version with the changes marked, and it does not survive
 * contact with this corpus: the README's two versions are 54,266 and 56,772
 * characters, so the reader gets two screens of identical prose to scroll
 * through looking for the amber. A diff is a list of changes, and the useful
 * view is that list — each row showing what the old version said on the left and
 * what the new one says on the right.
 *
 * The marks are the existing highlight rendering: `.mark` for what arrived,
 * `.mark-removed` for what left. `segmentAll` rather than `segment`, because a
 * span here can contain several changed runs and marking them one at a time
 * would double-count the unchanged text between them.
 *
 * **Summaries are opt-in and say so.** Each one is a model call, and a page that
 * spent money on mount would be a page nobody could leave open. The button is
 * the request; the cache means the second press is free.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  api,
  type DiffSpan,
  type Evolution as EvolutionData,
  type EvolutionVersion,
  type VersionDiff,
} from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { count, isCode } from "../../lib/format";
import { segmentAll } from "../../lib/highlight";

export function Evolution({ memoryId, kind }: { memoryId: string; kind: string }) {
  const [pair, setPair] = useState<{ from: string; to: string } | null>(null);
  const [summarize, setSummarize] = useState(false);

  const evolution = useQuery({
    queryKey: ["evolution", memoryId, pair?.from, pair?.to, summarize],
    queryFn: () =>
      api.evolution({ id: memoryId, from: pair?.from, to: pair?.to, summarize }),
  });

  if (evolution.isLoading) return <Loading rows={3} />;
  if (evolution.isError) return <Failure error={evolution.error} />;
  if (!evolution.data) return null;

  const data = evolution.data;
  if (data.versions.length < 2) {
    return (
      <p className="meta text-faint">
        One version. Nothing has changed since this was first ingested, so there is no
        diff to show.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <VersionSelector
        versions={data.versions}
        pair={pair}
        onSelect={setPair}
        summarize={summarize}
        onSummarize={setSummarize}
      />
      {data.diffs.map((diff) => (
        <DiffPanel
          key={`${diff.from_id}-${diff.to_id}`}
          diff={diff}
          versions={data.versions}
          code={isCode(kind)}
        />
      ))}
    </div>
  );
}

/**
 * Which pair to diff.
 *
 * Defaults to every consecutive pair, which is the history. Picking two
 * non-adjacent versions asks a different question — "what changed between v1 and
 * v4" is not three diffs read together, because material added in v2 and removed
 * in v3 appears in neither.
 */
function VersionSelector({
  versions,
  pair,
  onSelect,
  summarize,
  onSummarize,
}: {
  versions: EvolutionVersion[];
  pair: { from: string; to: string } | null;
  onSelect: (pair: { from: string; to: string } | null) => void;
  summarize: boolean;
  onSummarize: (on: boolean) => void;
}) {
  const from = pair?.from ?? versions[0].id;
  const to = pair?.to ?? versions[versions.length - 1].id;

  return (
    <div className="flex flex-wrap items-baseline gap-x-5 gap-y-2 border-b border-rule pb-2">
      <span className="inline-flex items-baseline gap-2">
        <span className="meta-label">compare</span>
        <select
          className="field meta"
          value={from}
          onChange={(event) => onSelect({ from: event.target.value, to })}
          aria-label="older version"
        >
          {versions.map((version) => (
            <option key={version.id} value={version.id}>
              v{version.version}
            </option>
          ))}
        </select>
        <span className="meta text-faint">→</span>
        <select
          className="field meta"
          value={to}
          onChange={(event) => onSelect({ from, to: event.target.value })}
          aria-label="newer version"
        >
          {versions.map((version) => (
            <option key={version.id} value={version.id}>
              v{version.version}
            </option>
          ))}
        </select>
      </span>

      <button
        type="button"
        className={`btn ${pair === null ? "btn-on" : ""}`}
        onClick={() => onSelect(null)}
      >
        every consecutive pair
      </button>

      <button
        type="button"
        className={`btn ml-auto ${summarize ? "btn-on" : ""}`}
        onClick={() => onSummarize(!summarize)}
        title="Each summary is one language-model call. Cached on the version pair afterwards."
      >
        {summarize ? "summaries on" : "describe changes"}
      </button>
    </div>
  );
}

function DiffPanel({
  diff,
  versions,
  code,
}: {
  diff: VersionDiff;
  versions: EvolutionVersion[];
  code: boolean;
}) {
  const after = versions.find((version) => version.id === diff.to_id);

  return (
    <section className="flex flex-col gap-2" data-testid="diff-panel">
      <SectionHeading
        right={
          <span className="flex items-baseline gap-3">
            <span className="text-affirm">+{count(diff.added_chars)}</span>
            <span className="text-deny">−{count(diff.removed_chars)}</span>
            <span className="text-faint">
              {diff.span_count} span{diff.span_count === 1 ? "" : "s"}
            </span>
            <span className="text-faint">
              {/* Null rather than zero, and rendered as a reason rather than a
                  dash: superseded versions have no chunks to count. */}
              chunk delta{" "}
              {diff.chunk_delta === null ? (
                <span title="superseded versions hold no chunks — M1.4 deletes them when the next version is written">
                  n/a
                </span>
              ) : (
                diff.chunk_delta
              )}
            </span>
          </span>
        }
      >
        v{diff.from_version} → v{diff.to_version}
      </SectionHeading>

      {after ? (
        <div className="flex flex-wrap items-baseline gap-x-4">
          <Tag>{after.change}</Tag>
          <span className="meta text-faint">occurred</span>
          <DateStamp value={after.occurred_at} provenance={after.occurred_at_source} />
          <span className="meta text-faint">
            {count(after.characters)} chars
            {after.holds_chunks ? `, ${after.chunks} chunks` : ", chunks discarded"}
          </span>
        </div>
      ) : null}

      {diff.summary ? <SummaryLine summary={diff.summary} /> : null}

      {diff.is_empty ? (
        <p className="meta text-muted" data-testid="empty-diff">
          The normalized text is identical. The bytes changed — a new artifact and a new
          version were recorded — but nothing a reader would call a change did, so the
          chunks were adopted rather than rebuilt.
        </p>
      ) : (
        <>
          <SpanTable spans={diff.spans} code={code} />
          {diff.span_count > diff.spans.length ? (
            <p className="meta text-faint">
              showing {diff.spans.length} of {count(diff.span_count)} spans
            </p>
          ) : null}
          {diff.affected_chunks.length > 0 ? (
            <p className="meta text-muted" data-testid="affected-chunks">
              touches {diff.affected_chunks.length} chunk
              {diff.affected_chunks.length === 1 ? "" : "s"}:{" "}
              {diff.affected_chunks
                .slice(0, 10)
                .map(
                  (chunk) =>
                    `#${chunk.ordinal}${chunk.definition ? ` (${chunk.definition})` : ""}`,
                )
                .join(", ")}
              {diff.affected_chunks.length > 10 ? " …" : ""}
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}

/**
 * The generated description, with its grounding verdict attached.
 *
 * The verdict is never hidden and the ungrounded terms are named. M2.6's rule
 * reaching a new place: a summary that named something the diff does not contain
 * is reported as such rather than quietly presented as fact.
 */
function SummaryLine({ summary }: { summary: NonNullable<VersionDiff["summary"]> }) {
  return (
    <div
      className={`border-l-2 pl-3 ${summary.grounded ? "border-rule-strong" : "border-deny"}`}
      data-testid="change-summary"
    >
      <p className="prose-content text-ink">{summary.text}</p>
      <p className="meta mt-1 text-faint">
        {summary.trivial ? "decided from the diff, not generated" : summary.model_id}
        {summary.cached ? " · cached" : ""}
        {summary.grounded ? "" : " · ungrounded"}
      </p>
      {summary.unsupported.length > 0 ? (
        <p className="meta text-deny">
          named but not in the diff: {summary.unsupported.join(", ")}
        </p>
      ) : null}
      {summary.context_only.length > 0 ? (
        <p className="meta text-muted" title="present only on unchanged context lines">
          from context, not from the change: {summary.context_only.join(", ")}
        </p>
      ) : null}
    </div>
  );
}

function SpanTable({ spans, code }: { spans: DiffSpan[]; code: boolean }) {
  if (spans.length === 0) return null;
  const body = code ? "code-content" : "prose-content";

  return (
    <div className="flex flex-col border border-rule" data-testid="span-table">
      {spans.map((span, index) => (
        <div
          key={`${span.b_start}-${index}`}
          className={`grid grid-cols-1 gap-px sm:grid-cols-2 ${
            index > 0 ? "border-t border-rule" : ""
          }`}
          data-testid="diff-row"
          data-kind={span.kind}
        >
          <Side
            text={span.a_text}
            body={body}
            removed
            label={`before · ${span.a_start}`}
            empty="nothing here before"
          />
          <Side
            text={span.b_text}
            body={body}
            label={`after · ${span.b_start}`}
            empty="removed, nothing replaced it"
          />
        </div>
      ))}
    </div>
  );
}

function Side({
  text,
  body,
  removed,
  label,
  empty,
}: {
  text: string;
  body: string;
  removed?: boolean;
  label: string;
  empty: string;
}) {
  // The whole side is the changed run, so the mark covers all of it. `segmentAll`
  // rather than a bare span so the two sides go through the same code path the
  // search highlight does, and a future partial mark needs no new rendering.
  const segments = segmentAll(text, [{ start: 0, end: text.length }]);

  return (
    <div className={`px-3 py-2 ${removed ? "bg-paper" : "bg-raised"}`}>
      <p className="meta-label mb-1">{label}</p>
      {text.length === 0 ? (
        <p className="meta text-faint">{empty}</p>
      ) : (
        <div className={body}>
          {segments.map((segment, index) =>
            segment.marked ? (
              <mark
                key={index}
                className={removed ? "mark-removed" : "mark"}
                data-testid={removed ? "removed" : "added"}
              >
                {segment.text}
              </mark>
            ) : (
              <span key={index}>{segment.text}</span>
            ),
          )}
        </div>
      )}
    </div>
  );
}

export type { EvolutionData };
