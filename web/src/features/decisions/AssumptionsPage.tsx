/**
 * Recurring assumptions, worst first.
 *
 * **The group table is the point of this screen and the corpus-wide hold rate
 * is not.** A rate over every assumption mostly reflects which ones were easy
 * to check; a group of four with a 25% hold rate is a finding about how
 * somebody estimates, because the same belief failed four times in four
 * different decisions. That is what M5.3 reads.
 *
 * Sorted by failure rate rather than by size, so a group of two that broke both
 * times outranks a group of five that mostly held. `partially` counts towards
 * failure here and not towards holding in the rate beside it — the two are
 * deliberately not complements, because a belief that half held is a belief
 * that half broke and the view whose job is surfacing trouble should say so.
 *
 * Unevaluated assumptions are in neither half of any number on this page. The
 * count is shown instead, because on a young corpus it is usually the largest
 * of the three and a page that hid it would be quoting a percentage of whatever
 * happened to get attention.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api, type AssumptionDetail, type AssumptionGroup } from "../../api/client";
import { Empty, Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { count, percent } from "../../lib/format";

export function AssumptionsPage() {
  const stats = useQuery({ queryKey: ["assumption-stats"], queryFn: api.assumptionStats });
  const rows = useQuery({
    queryKey: ["assumptions"],
    queryFn: () => api.assumptions(),
  });

  if (stats.isLoading) return <Loading rows={5} />;
  if (stats.isError) return <Failure error={stats.error} />;
  if (!stats.data) return null;

  const data = stats.data;
  const unevaluated = (rows.data ?? []).filter((row) => row.held === null);

  return (
    <div className="flex flex-col gap-5">
      <SectionHeading
        right={
          <Link className="text-accent underline" to="/decisions">
            decisions
          </Link>
        }
      >
        assumptions
      </SectionHeading>

      <div className="meta flex flex-wrap items-baseline gap-4 text-faint">
        <span className="text-affirm">{count(data.held)} held</span>
        <span className="text-deny">{count(data.failed)} failed</span>
        <span className="text-accent">{count(data.partially)} partially</span>
        {/* In neither half of any rate on this page. */}
        <span>{count(data.unevaluated)} unevaluated</span>
        <span className="text-ink">
          {data.hold_rate === null
            ? "nothing evaluated, so no hold rate"
            : `${percent(data.hold_rate)} of ${count(data.evaluated)} evaluated`}
        </span>
      </div>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${data.groups.length}`}>
          recurring — the same belief across decisions
        </SectionHeading>
        {data.groups.length === 0 ? (
          <Empty title="nothing recurs yet">
            Every assumption in this corpus is held once, so there is no recurrence for a
            pattern to be made of. Group them with{" "}
            <code className="kbd">memoryos assumptions group</code> — a group of one is an
            assumption nothing else resembles, which is a fact about the corpus rather than
            a finding about anybody&apos;s judgement.
          </Empty>
        ) : (
          <ul className="flex flex-col gap-3">
            {data.groups.map((group) => (
              <GroupRow key={group.id} group={group} />
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${unevaluated.length}`}>
          not yet evaluated
        </SectionHeading>
        {unevaluated.length === 0 ? (
          <p className="meta text-faint">Everything has been looked at.</p>
        ) : (
          <>
            <p className="meta max-w-prose leading-relaxed text-muted">
              Left unevaluated on purpose rather than guessed at. &quot;Nothing has gone
              wrong&quot; is usually evidence that a belief was never exercised, which is a
              different fact from it having held.
            </p>
            <ul className="flex flex-col gap-1">
              {unevaluated.map((row) => (
                <UnevaluatedRow key={row.id} row={row} />
              ))}
            </ul>
          </>
        )}
      </section>
    </div>
  );
}

function GroupRow({ group }: { group: AssumptionGroup }) {
  return (
    <li className="border-l-2 border-rule pl-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="meta-label text-muted">
          {group.members} members · {group.evaluated} evaluated
        </span>
        <Tag>{group.strategy}</Tag>
        <span className="meta text-ink">
          {group.hold_rate === null
            ? "no verdicts yet"
            : `held ${percent(group.hold_rate)}`}
        </span>
        {group.failure_rate !== null && group.failure_rate > 0 ? (
          <span className="meta text-deny">
            broke or half-broke {percent(group.failure_rate)}
          </span>
        ) : null}
      </div>
      <ul className="mt-1">
        {group.statements.map((statement, index) => (
          <li key={index} className="prose-content text-sm text-ink">
            · {statement}
          </li>
        ))}
      </ul>
    </li>
  );
}

function UnevaluatedRow({ row }: { row: AssumptionDetail }) {
  return (
    <li className="flex items-baseline gap-3 border-b border-rule/60 pb-1">
      <span className="prose-content flex-1 text-sm text-ink">{row.statement}</span>
      <Link
        className="meta shrink-0 text-accent underline"
        to={`/decisions/${row.decision_id}`}
      >
        {row.decision_question.slice(0, 40)}
      </Link>
      <span className="meta w-10 shrink-0 text-right text-faint">
        {row.confidence === null ? "—" : row.confidence.toFixed(2)}
      </span>
    </li>
  );
}
