/**
 * Candidate outcomes: the decision on the left, the memory that followed it on
 * the right, and the gap between them stated in the middle.
 *
 * **The gap is the claim.** Everything in this queue is here because one thing
 * occurred after another, which is not evidence of anything on its own — two
 * documents in the same repository are related by default. So the number the
 * claim rests on is on screen, in days, beside the window that admitted it,
 * rather than folded into a confidence score the reviewer would have to trust.
 *
 * **`entity_filter` is shown even though it will usually say `applied`.** When
 * it says `unavailable` the candidate was found by time alone, because nothing
 * in the decision's evidence has been extracted — and a candidate found by time
 * alone is much weaker evidence than one sharing a resolved entity. A queue
 * that hid the difference would silently change meaning depending on whether
 * anybody had run extraction lately, which is exactly what happened to this
 * corpus between M5.0 and M5.1.
 *
 * Accepting writes an `inferred` outcome, never a declared one, and the button
 * says so. Somebody who actually watched the outcome should record it directly.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type OutcomeSuggestion } from "../../api/client";
import { Empty, Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { timestamp } from "../../lib/format";

interface Draft {
  description?: string | null;
  verdict?: string | null;
  rationale?: string | null;
}

/**
 * The gap, in a unit a reader can judge. Mirrors `describe_gap` in
 * `application/outcome_suggest.py`, which the CLI uses.
 *
 * "0.0 days" is what this corpus produces constantly, and it is the least
 * useful thing the interface could say: every mtime here falls inside a
 * 2-day-18-hour window and files written in one batch are seconds apart. A gap
 * that rounds to zero is the temporal signal saying it has nothing to offer,
 * not a very tight correlation, and it should read that way.
 */
function describeGap(days: number): string {
  if (days >= 1) return `${days.toFixed(1)} days`;
  const hours = days * 24;
  if (hours >= 1) return `${hours.toFixed(1)} hours`;
  return `${Math.round(hours * 60)} minutes`;
}

export function OutcomeQueue() {
  const client = useQueryClient();
  const [status, setStatus] = useState<OutcomeSuggestion["status"]>("pending");
  const suggestions = useQuery({
    queryKey: ["outcome-suggestions", status],
    queryFn: () => api.outcomeSuggestions(status),
  });
  const rate = useQuery({ queryKey: ["success-rate"], queryFn: api.successRate });

  const review = useMutation({
    mutationFn: ({ id, verdict }: { id: string; verdict: "accept" | "reject" }) =>
      verdict === "accept"
        ? api.acceptOutcomeSuggestion(id)
        : api.rejectOutcomeSuggestion(id),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["outcome-suggestions"] });
      await client.invalidateQueries({ queryKey: ["success-rate"] });
      await client.invalidateQueries({ queryKey: ["decisions"] });
    },
  });

  if (suggestions.isLoading) return <Loading rows={4} />;
  if (suggestions.isError) return <Failure error={suggestions.error} />;

  const rows = suggestions.data ?? [];
  const byTime = rows.filter((row) => row.entity_filter === "unavailable").length;

  return (
    <div className="flex flex-col gap-4">
      <SectionHeading
        right={
          <Link className="text-amber underline" to="/decisions">
            decisions
          </Link>
        }
      >
        outcome review queue
      </SectionHeading>

      {rate.data ? (
        <div className="meta flex flex-wrap gap-4 text-faint">
          <span className="text-affirm">{rate.data.worked} worked</span>
          <span className="text-deny">{rate.data.failed} failed</span>
          <span className="text-amber">{rate.data.mixed} mixed</span>
          {/* Outside the rate, and said so. A decision it is too soon to judge
              and a decision nobody has looked at are different facts, and
              neither is a failure. */}
          <span>{rate.data.too_early} too early</span>
          <span>{rate.data.undecided} not looked at</span>
          <span className="text-ink">
            {rate.data.rate === null
              ? "no resolved outcomes, so no success rate"
              : `${Math.round(rate.data.rate * 100)}% of ${rate.data.resolved} resolved`}
          </span>
        </div>
      ) : null}

      <div className="meta flex items-baseline gap-4 text-faint">
        <span className="flex gap-2">
          {(["pending", "accepted", "rejected"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setStatus(value)}
              className={
                value === status ? "text-amber underline" : "text-muted hover:text-ink"
              }
            >
              {value}
            </button>
          ))}
        </span>
        {byTime > 0 ? (
          <span className="text-amber">
            {byTime} found by time alone — nothing extracted for that decision&apos;s
            evidence
          </span>
        ) : null}
      </div>

      {review.isError ? <Failure error={review.error} /> : null}

      {rows.length === 0 ? (
        <Empty title={`nothing ${status}`}>
          Propose candidates with{" "}
          <code className="kbd">memoryos outcomes suggest --window-days 90</code>. Nothing
          it finds is committed: every candidate lands here with the decision beside it and
          the temporal gap stated, and becomes an outcome only when somebody accepts it.
        </Empty>
      ) : (
        <ul className="flex flex-col gap-6">
          {rows.map((row) => (
            <CandidateRow
              key={row.id}
              row={row}
              busy={review.isPending}
              onReview={(verdict) => review.mutate({ id: row.id, verdict })}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function CandidateRow({
  row,
  busy,
  onReview,
}: {
  row: OutcomeSuggestion;
  busy: boolean;
  onReview: (verdict: "accept" | "reject") => void;
}) {
  const draft = (row.draft ?? {}) as Draft;

  return (
    <li className="border-b border-rule-strong pb-5">
      <div className="meta mb-2 flex flex-wrap items-baseline gap-3 text-faint">
        <Tag>{row.status}</Tag>
        <span>{row.model_id}</span>
        <span>{timestamp(row.suggested_at)}</span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-2">
          <p className="meta-label text-muted">the decision</p>
          <Link
            className="prose-content text-sm text-amber"
            to={`/decisions/${row.decision_id}`}
          >
            {row.decision_question}
          </Link>
          <p className="meta text-faint">
            decided {timestamp(row.decision_decided_at)}
          </p>

          <p className="meta-label mt-2 text-muted">what the model says happened</p>
          <p className="prose-content text-sm text-ink">{draft.description}</p>
          <p className="meta text-muted">verdict: {draft.verdict}</p>
          {draft.rationale ? (
            <p className="meta max-w-prose leading-relaxed text-faint">
              {draft.rationale}
            </p>
          ) : null}
        </div>

        <div className="flex flex-col gap-2">
          <p className="meta-label text-muted">the memory that followed</p>
          <span className="meta text-ink">
            {row.source_name}:{row.external_key}
          </span>
          {/* The whole basis of the claim, in one line, before the passage. */}
          <p className="meta text-ink">
            <span className="text-amber">{describeGap(row.gap_days)} later</span>
            <span className="text-faint">
              {" "}
              · window {Math.round(row.window_days)}d ·{" "}
              {row.entity_filter === "applied" ? (
                <>shares {row.shared_entities.join(", ") || "—"}</>
              ) : (
                <span className="text-amber">
                  entity filter unavailable: found by time alone
                </span>
              )}
            </span>
          </p>
          <blockquote className="prose-content max-h-56 overflow-y-auto border-l-2 border-edge bg-raised p-3 text-sm leading-relaxed text-muted">
            {row.source_text}
          </blockquote>
        </div>
      </div>

      {row.status === "pending" ? (
        <div className="mt-3 flex gap-3">
          <button
            type="button"
            disabled={busy}
            onClick={() => onReview("accept")}
            className="meta-label border border-edge px-3 py-1 text-affirm"
          >
            accept as inferred
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onReview("reject")}
            className="meta-label border border-rule px-3 py-1 text-deny"
          >
            not an outcome
          </button>
          <span className="meta self-center text-faint">
            Accepting records this as <code className="kbd">inferred</code>, never as
            observed. If you watched it happen, record it on the decision instead.
          </span>
        </div>
      ) : null}
    </li>
  );
}
