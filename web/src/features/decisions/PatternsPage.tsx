/**
 * Behavioural patterns, with both evidence lists side by side.
 *
 * **Counter-evidence sits at the same visual weight as supporting evidence, in
 * the same grid, at the same width, always expanded.** Putting it below a fold,
 * behind a toggle, or in a smaller type size is how a tool becomes a flatterer:
 * the reader sees five decisions that agree, skims past the three that do not,
 * and comes away believing a claim the corpus half contradicts.
 *
 * The empty state is the important one, because on any young corpus it is the
 * one that shows. "No patterns" alone reads like a broken detector, so this page
 * shows the calibration table instead — every confidence band with the interval
 * its sample supports — which is the same result stated in a way somebody can
 * act on. A band whose stated confidence falls *inside* its interval is not a
 * near miss; it is the data saying you were as reliable as you claimed.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  api,
  type CalibrationBand,
  type Pattern,
  type PatternEvidence,
} from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Empty, Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { percent } from "../../lib/format";

export function PatternsPage() {
  const [includeDismissed, setIncludeDismissed] = useState(false);
  const client = useQueryClient();
  const patterns = useQuery({
    queryKey: ["patterns", includeDismissed],
    queryFn: () => api.patterns(includeDismissed),
  });
  const calibration = useQuery({
    queryKey: ["calibration"],
    queryFn: api.calibration,
  });

  const dismiss = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      api.dismissPattern(id, reason),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["patterns"] });
    },
  });

  if (patterns.isLoading) return <Loading rows={5} />;
  if (patterns.isError) return <Failure error={patterns.error} />;

  const rows = patterns.data ?? [];

  return (
    <div className="flex flex-col gap-5">
      <SectionHeading
        right={
          <span className="flex items-baseline gap-3">
            <button
              type="button"
              className={
                includeDismissed ? "text-accent underline" : "text-ink-2 hover:text-ink"
              }
              onClick={() => setIncludeDismissed((current) => !current)}
            >
              show dismissed
            </button>
            <Link className="text-accent underline" to="/decisions/reflections">
              reflections
            </Link>
            <Link className="text-accent underline" to="/decisions">
              decisions
            </Link>
          </span>
        }
      >
        patterns
      </SectionHeading>

      {dismiss.isError ? <Failure error={dismiss.error} /> : null}

      {rows.length === 0 ? (
        <Empty title="no patterns with sufficient support">
          A pattern needs at least three distinct decisions agreeing, more agreeing than
          disagreeing, and — for anything arithmetic — a gap its sample size can actually
          resolve. Nothing here clears that yet, which is a result rather than a failure.
          Run <code className="kbd">memoryos patterns discover</code> to see each
          candidate and the reason it was refused.
        </Empty>
      ) : (
        <ul className="flex flex-col gap-6">
          {rows.map((pattern) => (
            <PatternRow
              key={pattern.id}
              pattern={pattern}
              busy={dismiss.isPending}
              onDismiss={(reason) => dismiss.mutate({ id: pattern.id, reason })}
            />
          ))}
        </ul>
      )}

      {calibration.data ? (
        <section className="flex flex-col gap-3">
          <SectionHeading>calibration — stated confidence against what happened</SectionHeading>
          <p className="meta max-w-prose leading-relaxed text-ink-2">
            A band is a finding only when the stated confidence falls <em>outside</em> the
            interval its sample supports. Anything inside is consistent with having been
            exactly as reliable as claimed — fourteen assumptions that all held cannot
            distinguish &quot;right 85% of the time&quot; from &quot;right always&quot;.
          </p>
          {calibration.data.excluded_decisions + calibration.data.excluded_assumptions >
          0 ? (
            /* Above the tables, never below them. An empty table on its own reads
               as "nothing recorded yet"; the same table under this line reads as
               what it is, which is a measurement this corpus cannot support. */
            <p className="meta max-w-prose leading-relaxed text-deny">
              {calibration.data.excluded_decisions} decision(s) and{" "}
              {calibration.data.excluded_assumptions} assumption(s) are excluded from every
              band below: their confidence was reconstructed after the outcome was known,
              and hindsight cannot measure foresight at any weight.
            </p>
          ) : null}
          <CalibrationTable title="decisions" bands={calibration.data.decisions} />
          <CalibrationTable title="assumptions" bands={calibration.data.assumptions} />
          <p className="meta max-w-prose leading-relaxed text-ink-3">
            Calibration is only meaningful when the confidence was written down before the
            outcome was known. <code className="kbd">confidence_horizon</code> records
            which, derived from whether the decision&apos;s own date was asserted or
            recovered.
          </p>
        </section>
      ) : null}
    </div>
  );
}

function PatternRow({
  pattern,
  busy,
  onDismiss,
}: {
  pattern: Pattern;
  busy: boolean;
  onDismiss: (reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  const span =
    pattern.span_days === null ? null : `${Math.round(pattern.span_days)} days`;

  return (
    <li className="border-b border-rule-strong pb-5">
      <div className="meta mb-1 flex flex-wrap items-baseline gap-3 text-ink-3">
        <Tag>{pattern.kind}</Tag>
        <span className="text-ink">
          confidence {pattern.confidence === null ? "—" : pattern.confidence.toFixed(2)}
        </span>
        <span className="text-affirm">{pattern.support_count} supporting</span>
        <span className="text-deny">{pattern.contradiction_count} contradicting</span>
        {span ? <span>spanning {span}</span> : null}
        {pattern.dismissed_at ? (
          <span className="text-deny">dismissed: {pattern.dismissed_reason}</span>
        ) : null}
      </div>
      <p className="prose-content max-w-prose text-sm text-ink">{pattern.statement}</p>

      {/* Equal columns, equal type, both always open. */}
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <EvidenceColumn
          title="supports"
          tone="text-affirm"
          items={pattern.supporting}
        />
        <EvidenceColumn
          title="contradicts"
          tone="text-deny"
          items={pattern.contradicting}
        />
      </div>

      {pattern.dismissed_at === null ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <input
            className="field flex-1"
            placeholder="why this is not a real pattern"
            aria-label={`dismiss reason for ${pattern.id}`}
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          />
          <button
            type="button"
            className="meta-label border border-rule px-3 py-(--row-py) text-deny disabled:opacity-40"
            // A reason is required by the API and by a CHECK constraint, so the
            // button refuses rather than sending something the server rejects.
            disabled={busy || reason.trim().length === 0}
            onClick={() => onDismiss(reason.trim())}
          >
            dismiss
          </button>
        </div>
      ) : null}
    </li>
  );
}

function EvidenceColumn({
  title,
  tone,
  items,
}: {
  title: string;
  tone: string;
  items: PatternEvidence[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <p className={`meta-label ${tone}`}>
        {title} ({items.length})
      </p>
      {items.length === 0 ? (
        <p className="meta text-ink-3">none found</p>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {items.map((item) => (
            <li key={`${item.decision_id}-${item.note ?? ""}`}>
              <Link
                className="prose-content text-sm text-accent"
                to={`/decisions/${item.decision_id}`}
              >
                {item.decision_question}
              </Link>
              <div className="meta text-ink-3">
                <DateStamp value={item.decided_at} provenance="declared" />
                {item.note ? <span className="ml-2">{item.note}</span> : null}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function CalibrationTable({
  title,
  bands,
}: {
  title: string;
  bands: CalibrationBand[];
}) {
  if (bands.length === 0) {
    return (
      <p className="meta text-ink-3">
        {title}: nothing recorded before its outcome.
      </p>
    );
  }
  return (
    <div>
      <p className="meta-label mb-1 text-ink-2">{title} by stated confidence</p>
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-rule-strong">
            {["band", "n", "stated", "actual", "95% CI", ""].map((heading) => (
              <th key={heading} className="meta-label py-(--row-py) pr-4 font-normal">
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bands.map((band) => (
            <tr key={`${band.low}`} className="border-b border-rule/60">
              <td className="meta py-(--row-py) pr-4 text-ink">
                {band.low.toFixed(2)}–{band.high.toFixed(2)}
              </td>
              <td className="meta py-(--row-py) pr-4 text-ink-3">{band.n}</td>
              <td className="meta py-(--row-py) pr-4 text-ink">{band.stated.toFixed(2)}</td>
              <td className="meta py-(--row-py) pr-4 text-ink">{percent(band.observed)}</td>
              <td className="meta py-(--row-py) pr-4 text-ink-3">
                {percent(band.interval_low)}–{percent(band.interval_high)}
              </td>
              <td className="meta py-(--row-py)">
                {band.miscalibrated ? (
                  <span className="text-deny">outside the interval</span>
                ) : (
                  <span className="text-ink-3">within what the sample supports</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
