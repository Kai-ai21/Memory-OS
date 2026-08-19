/**
 * The temporal layer, looked at rather than printed.
 *
 * M4.0 produces numbers and a text histogram. What this adds is *shape* — that a
 * period was all one kind, that two clusters are the same size but one is spread
 * and one is a spike, that the hollow between them is a real gap with named
 * memories on both sides. None of that is legible in a column of counts, and
 * Phase 5 depends on being able to look at a period and recognise what was
 * happening in it.
 *
 * The provenance strip sits above the chart and is not collapsible. Everything
 * below it is only as good as it: a timeline of filesystem mtimes describes when
 * files were written to a disk, and on a fresh clone that is one afternoon
 * regardless of when the work happened.
 */

import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api, type Bucket, type Period } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Empty, Failure, Loading, Meta, SectionHeading, Tag } from "../../components/primitives";
import { count } from "../../lib/format";
import { confidence, explain, marker } from "../../lib/provenance";
import { ActivityChart } from "./ActivityChart";
import { bucketDays, bucketLabel, isoDate, kindColor, kindsPresent } from "./model";

const PERIODS: Period[] = ["day", "week", "month"];

export function TimelinePage() {
  // The window and grain live in the URL, so a period worth talking about can
  // be sent to somebody. Same reason the search page keeps its query there.
  const [params, setParams] = useSearchParams();
  const period = (params.get("period") ?? "month") as Period;
  const source = params.get("source") ?? undefined;
  const minDays = Number(params.get("min_days") ?? 30);
  const [selected, setSelected] = useState<Bucket | null>(null);

  const timeline = useQuery({
    queryKey: ["timeline", period, source],
    queryFn: () => api.timeline({ period, source }),
  });
  const gaps = useQuery({
    queryKey: ["gaps", minDays, source],
    queryFn: () => api.gaps({ minDays, source }),
  });
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });

  function setParam(key: string, value: string | null) {
    const next = new URLSearchParams(params);
    if (value === null) next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true });
    // The selection is a range in the old grain. Keeping it across a regrain
    // would show January's memories under a bar labelled Q1.
    setSelected(null);
  }

  if (timeline.isLoading) return <Loading rows={5} />;
  if (timeline.isError) return <Failure error={timeline.error} />;
  if (!timeline.data) return null;

  const data = timeline.data;
  const kinds = kindsPresent(data.buckets);
  // The longest run of empty buckets, counted from the response rather than
  // requested. `/gaps` answers "is there a silence of at least N days" and says
  // nothing when the answer is no; this is the follow-up question a reader
  // immediately has, and the buckets already on screen contain it. Derived, not
  // invented — it moves when the data does.
  const longestQuiet = longestEmptyRun(data.buckets);
  const dated = data.provenance.filter((band) => band.provenance !== "unknown");
  const undated = data.provenance.find((band) => band.provenance === "unknown");
  const total = data.provenance.reduce((sum, band) => sum + band.count, 0);

  return (
    <div className="flex flex-col gap-5">
      {/* --- What the dates are worth ---------------------------------- */}
      <section className="flex flex-col gap-2">
        <SectionHeading right={`${count(total)} current memories`}>
          date provenance
        </SectionHeading>
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
          {dated.map((band) => (
            <span key={band.provenance} className="inline-flex items-baseline gap-1.5">
              <span
                className={`meta ${confidence(band.provenance) === "stated" ? "text-ink" : "text-ink-2"}`}
                title={explain(band.provenance)}
              >
                {band.provenance}
                <span className="text-accent">{marker(band.provenance)}</span>
              </span>
              <span className="meta text-ink-3">
                {count(band.count)} ({total ? ((band.count / total) * 100).toFixed(1) : "0.0"}%)
              </span>
            </span>
          ))}
          {undated ? (
            <span className="inline-flex items-baseline gap-1.5">
              <span className="meta text-ink-3" title={explain("unknown")}>
                undated<span className="text-accent">?</span>
              </span>
              <span className={`meta ${undated.count ? "text-deny" : "text-ink-3"}`}>
                {count(undated.count)}
              </span>
            </span>
          ) : null}
        </div>
        {dated.every((band) => confidence(band.provenance) !== "stated") && total > 0 ? (
          // The honest caption for this corpus, and it is not decoration: it
          // changes what every bar below means.
          <p className="meta max-w-prose text-ink-2">
            Every date here is inferred rather than stated. Nothing in this corpus declared
            when it happened, so the chart below shows when files were last written to this
            disk — which is not the same as when the work happened.
          </p>
        ) : null}
      </section>

      {/* --- Controls --------------------------------------------------- */}
      <section className="flex flex-wrap items-baseline gap-x-6 gap-y-2 border-b border-rule pb-2">
        <span className="inline-flex items-baseline gap-2">
          <span className="meta-label">period</span>
          {PERIODS.map((value) => (
            <button
              key={value}
              type="button"
              className={`btn ${value === period ? "btn-on" : ""}`}
              onClick={() => setParam("period", value)}
            >
              {value}
            </button>
          ))}
        </span>

        <span className="inline-flex items-baseline gap-2">
          <span className="meta-label">source</span>
          <select
            className="field meta"
            value={source ?? ""}
            onChange={(event) => setParam("source", event.target.value || null)}
          >
            <option value="">all</option>
            {(sources.data ?? []).map((row) => (
              <option key={row.id} value={row.name}>
                {row.name}
              </option>
            ))}
          </select>
        </span>

        <span className="inline-flex items-baseline gap-2">
          <span className="meta-label">gaps over</span>
          <input
            type="number"
            min={0.1}
            step={0.1}
            className="field meta w-20"
            value={minDays}
            onChange={(event) => setParam("min_days", event.target.value)}
          />
          <span className="meta text-ink-3">days</span>
        </span>

        <span className="ml-auto inline-flex items-baseline gap-3">
          {kinds.map((kind) => (
            <span key={kind} className="inline-flex items-baseline gap-1">
              <span
                className="inline-block h-2 w-2"
                style={{ backgroundColor: kindColor(kind) }}
                aria-hidden="true"
              />
              <span className="meta text-ink-3">{kind}</span>
            </span>
          ))}
        </span>
      </section>

      {/* --- The chart --------------------------------------------------- */}
      <section className="flex flex-col gap-1">
        <div className="flex flex-wrap items-baseline gap-x-5">
          {/* Named as UTC because every date on this page is, and a chart
              whose axis is in one zone and whose rows are in another is the
              off-by-one nobody finds. */}
          <Meta label="window (utc)">
            {isoDate(data.start)} → {isoDate(data.end)}
          </Meta>
          <Meta label="periods">{data.buckets.length}</Meta>
          <Meta label="plotted">{count(data.total)}</Meta>
          <Meta label="gaps">{gaps.data?.length ?? 0}</Meta>
        </div>
        <div className="mt-4">
          <ActivityChart
            buckets={data.buckets}
            gaps={gaps.data ?? []}
            selected={selected?.start ?? null}
            onSelect={setSelected}
            labelFor={(bucket) => bucketLabel(bucket, period)}
          />
        </div>
        <p className="meta mt-1 text-ink-3">
          <span className="hatch mr-1 inline-block h-2 w-4 align-middle opacity-45" /> a period
          with nothing in it. <span className="ml-2">Arrow keys walk the bars; enter opens one.</span>
        </p>
      </section>

      {/* --- The clicked period ------------------------------------------ */}
      <PeriodDetail bucket={selected} period={period} source={source} />
      {/* --- Gaps, listed ------------------------------------------------ */}
      <section className="flex flex-col gap-2">
        <SectionHeading right={`${minDays}d or longer`}>gaps</SectionHeading>
        {gaps.isError ? <Failure error={gaps.error} /> : null}
        {gaps.data && gaps.data.length === 0 ? (
          /* Not a blank section, and not "no data". Which threshold produced
             this answer is the whole of what the reader needs, because it is
             adjustable and sitting a few inches above — and the second sentence
             is the rule people get wrong: trailing silence is not a gap. */
          <div className="panel flex flex-col gap-2 border-dashed p-5" data-testid="no-gaps">
            <p className="meta-label font-bold text-warn">No silence this long</p>
            <p className="prose-content max-w-prose text-sm text-ink-2">
              Nothing in this corpus is quiet for {minDays} days or more. Lower the
              threshold above to see shorter silences
              {longestQuiet !== null && longestQuiet > 0 ? (
                <>
                  {" "}
                  — the longest run of empty {period}
                  {longestQuiet === 1 ? "" : "s"} in the window is{" "}
                  <span className="text-ink">
                    {longestQuiet} {period}
                    {longestQuiet === 1 ? "" : "s"}
                  </span>
                </>
              ) : null}
              . A gap also needs activity on both sides: the silence since the newest
              memory is not one, however long it has run.
            </p>
          </div>
        ) : null}
        <ul>
          {(gaps.data ?? []).map((gap) => (
            <li
              key={`${gap.source_name}-${gap.start}`}
              className="flex flex-wrap items-baseline gap-x-4 border-b border-rule/60 py-1.5"
            >
              <span className="meta w-16 text-ink">{gap.days.toFixed(1)}d</span>
              <span className="meta text-ink-3">{gap.source_name}</span>
              <span className="meta text-ink-2">
                <DateStamp
                  value={gap.start}
                  provenance={gap.before.occurred_at_source}
                  utc
                />
                {" → "}
                <DateStamp
                  value={gap.end}
                  provenance={gap.after.occurred_at_source}
                  utc
                />
              </span>
              <span className="meta flex-1 truncate text-ink-3">
                after{" "}
                <Link to={`/memory/${gap.before.id}`} className="text-accent underline">
                  {gap.before.external_key}
                </Link>
                , next{" "}
                <Link to={`/memory/${gap.after.id}`} className="text-accent underline">
                  {gap.after.external_key}
                </Link>
              </span>
            </li>
          ))}
        </ul>
      </section>

    </div>
  );
}

/**
 * The longest consecutive run of empty buckets in a window.
 *
 * Counts periods rather than days on purpose: it is reported beside the grain
 * the reader chose, so "3 days" and "3 months" are both answerable from the
 * same number without the page claiming a precision the buckets do not have.
 */
function longestEmptyRun(buckets: Bucket[]): number {
  let longest = 0;
  let run = 0;
  for (const bucket of buckets) {
    run = bucket.count === 0 ? run + 1 : 0;
    if (run > longest) longest = run;
  }
  return longest;
}

/**
 * The reference's provenance chip: DECLARED, FILESYSTEM, PARSED.
 *
 * Cyan when the source actually said when this happened, neutral when nobody
 * did and the date came from metadata about the container. Never colour alone —
 * the word is the label, and `DateStamp` beside it carries the glyph too, so a
 * reader who cannot see the difference in hue still gets the difference in
 * claim.
 */
function ProvenanceChip({ provenance }: { provenance: string | null | undefined }) {
  const tier = confidence(provenance);
  return (
    <span
      className={`chip shrink-0 uppercase ${tier === "stated" ? "chip-accent" : ""}`}
      title={explain(provenance)}
      data-testid="provenance-chip"
      data-provenance={provenance ?? "unknown"}
    >
      {provenance ?? "unknown"}
    </span>
  );
}

/**
 * The memories in the clicked period.
 *
 * Fetched by the bucket's own boundaries rather than by a date and a guessed
 * width: `window_days` is the bucket's real length, which is 31 for January and
 * 28 for February, so the list is exactly the bar.
 */
function PeriodDetail({
  bucket,
  period,
  source,
}: {
  bucket: Bucket | null;
  period: Period;
  source: string | undefined;
}) {
  const windowDays = bucket ? bucketDays(bucket) : 0;
  const detail = useQuery({
    queryKey: ["memories-at", bucket?.start, windowDays, source],
    queryFn: () => api.memoriesAt({ date: bucket!.start, windowDays, source }),
    enabled: bucket !== null,
  });

  if (!bucket) {
    return (
      <Empty title="no period selected">
        Click a bar, or focus one and press enter, to list what happened in it.
      </Empty>
    );
  }

  return (
    <section className="flex flex-col gap-2">
      <SectionHeading right={`${count(bucket.count)} in this ${period}`}>
        {bucketLabel(bucket, period)} — {isoDate(bucket.start)} to {isoDate(bucket.end)}
      </SectionHeading>

      {detail.isLoading ? <Loading rows={3} /> : null}
      {detail.isError ? <Failure error={detail.error} /> : null}
      {detail.data && detail.data.total === 0 ? (
        <p className="meta text-ink-3">
          Nothing occurred in this period. That is the gap, seen from inside it.
        </p>
      ) : null}
      {detail.data && detail.data.memories.length < detail.data.total ? (
        <p className="meta text-ink-2">
          showing {detail.data.memories.length} of {count(detail.data.total)}
        </p>
      ) : null}

      <ul data-testid="period-memories">
        {(detail.data?.memories ?? []).map((memory) => (
          <li
            key={memory.id}
            className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-rule py-3 last:border-b-0"
          >
            <div className="flex min-w-0 flex-1 basis-64 flex-col gap-1">
              <Link
                to={`/memory/${memory.id}`}
                className="meta truncate text-accent hover:underline"
              >
                {memory.external_key}
              </Link>
              {memory.title ? (
                <span className="prose-content truncate text-sm">{memory.title}</span>
              ) : null}
            </div>
            <Tag>{memory.kind}</Tag>
            {/* The reference's DECLARED / FILESYSTEM chip, and it is the real
                `occurred_at_source` rather than a label. A stated date takes the
                accent chip; an inferred one stays neutral. That difference is the
                whole reason M1.1 stored the column: an mtime is a fact about a
                file on a disk, not about when the work happened, and two dates
                rendered identically assert they are the same kind of claim. */}
            <ProvenanceChip provenance={memory.occurred_at_source} />
            <span className="shrink-0">
              <DateStamp
                value={memory.occurred_at}
                provenance={memory.occurred_at_source}
                utc
              />
            </span>
            <span className="meta shrink-0 text-ink-3" title="when this system learned about it">
              ingested {memory.ingested_at ? isoDate(memory.ingested_at) : "—"}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
