/**
 * The decision list, and the completeness of each record.
 *
 * The three counts on every row are the content, not decoration. A decision with
 * two options, no assumptions and no evidence is one that M5.2 has nothing to
 * evaluate and M5.3 cannot find a pattern in — and the only place that is
 * visible is beside the ones that are complete. A list showing question and
 * date alone would make a corpus of stubs look finished.
 *
 * `decided_at` carries its provenance the way every date in this interface does.
 * The twelve seeded decisions are `parsed` — read out of the milestone they
 * belong to — so every one of them wears a `~`, which is exactly right: nobody
 * wrote the date down at the time.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api, type DecisionStatus } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Empty, Failure, Loading, SectionHeading } from "../../components/primitives";
import { count } from "../../lib/format";

const STATUSES: (DecisionStatus | "all")[] = ["all", "open", "settled", "reversed"];

export function DecisionsPage() {
  const [status, setStatus] = useState<DecisionStatus | "all">("all");
  const decisions = useQuery({
    queryKey: ["decisions", status],
    queryFn: () => api.decisions(status === "all" ? undefined : status),
  });
  const pending = useQuery({
    queryKey: ["suggestions", "pending"],
    queryFn: () => api.suggestions("pending"),
  });
  const outcomePending = useQuery({
    queryKey: ["outcome-suggestions", "pending"],
    queryFn: () => api.outcomeSuggestions("pending"),
  });
  const rate = useQuery({ queryKey: ["success-rate"], queryFn: api.successRate });

  if (decisions.isLoading) return <Loading rows={5} />;
  if (decisions.isError) return <Failure error={decisions.error} />;

  const rows = decisions.data ?? [];
  const queued = pending.data?.length ?? 0;
  const outcomeQueue = outcomePending.data?.length ?? 0;
  // What the corpus can actually be reasoned about. Reported rather than
  // implied: a decision with no assumptions is invisible to M5.2.
  const withAssumptions = rows.filter((row) => row.assumptions > 0).length;
  const withEvidence = rows.filter((row) => row.evidence > 0).length;

  return (
    <div className="flex flex-col gap-4">
      <SectionHeading
        right={
          <span className="flex items-baseline gap-3">
            <Link className="text-accent underline" to="/decisions/new">
              record one
            </Link>
            <Link className="text-accent underline" to="/decisions/review">
              review queue{queued ? ` (${queued})` : ""}
            </Link>
            <Link className="text-accent underline" to="/decisions/outcomes">
              outcomes{outcomeQueue ? ` (${outcomeQueue})` : ""}
            </Link>
            <Link className="text-accent underline" to="/decisions/assumptions">
              assumptions
            </Link>
            <Link className="text-accent underline" to="/decisions/patterns">
              patterns
            </Link>
          </span>
        }
      >
        decisions
      </SectionHeading>

      <div className="meta flex items-baseline gap-4 text-ink-3">
        <span className="flex gap-2">
          {STATUSES.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setStatus(value)}
              className={
                value === status ? "text-accent underline" : "text-ink-2 hover:text-ink"
              }
            >
              {value}
            </button>
          ))}
        </span>
        <span>
          {count(rows.length)} recorded · {count(withAssumptions)} with assumptions ·{" "}
          {count(withEvidence)} with evidence
        </span>
      </div>

      {rate.data ? (
        // How the corpus turned out, with `too_early` and `not looked at`
        // outside the rate. Three numbers rather than one percentage, because
        // "too soon to say" and "nobody checked" are different facts and
        // neither is a failure.
        <div className="meta flex flex-wrap gap-4 text-ink-3">
          <span className="text-affirm">{rate.data.worked} worked</span>
          <span className="text-deny">{rate.data.failed} failed</span>
          <span className="text-accent">{rate.data.mixed} mixed</span>
          <span>{rate.data.too_early} too early</span>
          <span>{rate.data.undecided} not looked at</span>
          <span className="text-ink">
            {rate.data.rate === null
              ? "no resolved outcomes, so no success rate"
              : `${Math.round(rate.data.rate * 100)}% of ${rate.data.resolved} resolved`}
          </span>
        </div>
      ) : null}

      {rows.length === 0 ? (
        <Empty title="nothing decided yet">
          Record one with <code className="kbd">memoryos decide --interactive</code>, which
          asks for the alternatives, the confidence you hold right now, and your assumptions
          one at a time. A decision with no alternatives is refused — it is a description of
          what happened rather than a decision.
        </Empty>
      ) : (
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-rule-strong">
              {["question", "chosen", "status", "conf", "opts", "assum", "evid", "decided"].map(
                (heading) => (
                  <th key={heading} className="meta-label py-(--row-py) pr-4 font-normal">
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="border-b border-rule/60 align-top">
                <td className="py-(--row-py) pr-4">
                  <Link className="prose-content text-sm text-accent" to={`/decisions/${row.id}`}>
                    {row.question}
                  </Link>
                </td>
                <td className="prose-content py-(--row-py) pr-4 text-sm">{row.chosen}</td>
                <td className="meta py-(--row-py) pr-4 text-ink-2">{row.status}</td>
                <td className="meta py-(--row-py) pr-4 text-ink">
                  {/* Never rounded away and never defaulted: an absent confidence
                      is a different claim from a low one. */}
                  {row.confidence === null ? "—" : row.confidence.toFixed(2)}
                </td>
                <td className="meta py-(--row-py) pr-4 text-ink-3">{row.options}</td>
                <td
                  className={`meta py-(--row-py) pr-4 ${
                    row.assumptions === 0 ? "text-deny" : "text-ink-3"
                  }`}
                >
                  {row.assumptions}
                </td>
                <td className="meta py-(--row-py) pr-4 text-ink-3">{row.evidence}</td>
                <td className="py-(--row-py)">
                  <DateStamp value={row.decided_at} provenance={row.decided_at_source} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
