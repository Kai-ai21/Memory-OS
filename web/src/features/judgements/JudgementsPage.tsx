/**
 * The golden set, as it grows.
 *
 * Its job is to make the labelled data feel like an asset rather than a
 * side-effect: how many queries have been judged, how lopsided each one is, and
 * where the gaps are. A query with only `relevant` verdicts is weak evidence — it
 * says what should be returned and nothing about what should not.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../../api/client";
import { Empty, Failure, Loading, SectionHeading } from "../../components/primitives";
import { RelativeTime } from "../../components/RelativeTime";
import { count } from "../../lib/format";

export function JudgementsPage() {
  const summaries = useQuery({ queryKey: ["judgements"], queryFn: api.judgements });

  if (summaries.isLoading) return <Loading rows={4} />;
  if (summaries.isError) return <Failure error={summaries.error} />;

  const rows = summaries.data ?? [];
  const totals = rows.reduce(
    (sum, row) => ({
      relevant: sum.relevant + row.relevant,
      not_relevant: sum.not_relevant + row.not_relevant,
      missing: sum.missing + row.missing,
      total: sum.total + row.total,
    }),
    { relevant: 0, not_relevant: 0, missing: 0, total: 0 },
  );

  return (
    <div className="flex flex-col gap-4">
      <SectionHeading
        right={`${count(rows.length)} queries · ${count(totals.total)} judgements`}
      >
        golden set
      </SectionHeading>

      {rows.length === 0 ? (
        <Empty title="nothing judged yet">
          Run a search and mark results relevant or not relevant. Judgements are keyed on the
          query text and the item, so re-judging replaces rather than accumulates — and a full
          replay never touches them.
        </Empty>
      ) : (
        <>
          <div className="meta flex gap-4 text-ink-3">
            <span className="text-affirm">{count(totals.relevant)} relevant</span>
            <span className="text-deny">{count(totals.not_relevant)} not relevant</span>
            <span className="text-accent">{count(totals.missing)} missing</span>
          </div>

          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-rule-strong">
                {["query", "relevant", "not relevant", "missing", "last judged", ""].map(
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
                <tr key={row.query_text} className="border-b border-rule/60">
                  <td className="py-(--row-py) pr-4">
                    <span className="prose-content text-sm">{row.query_text}</span>
                  </td>
                  <td className="meta py-(--row-py) pr-4 text-affirm">{row.relevant}</td>
                  <td className="meta py-(--row-py) pr-4 text-deny">{row.not_relevant}</td>
                  <td className="meta py-(--row-py) pr-4 text-accent">{row.missing}</td>
                  <td className="meta py-(--row-py) pr-4 text-ink-3">
                    <RelativeTime value={row.last_judged_at} />
                  </td>
                  <td className="py-(--row-py)">
                    <Link
                      className="meta text-accent underline"
                      to={`/search?q=${encodeURIComponent(row.query_text)}`}
                    >
                      re-run
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="meta text-ink-3">
            Export with <code className="kbd">memoryos export-golden-set</code>, or fetch{" "}
            <code className="kbd">/judgements/export</code>. Ids are re-resolved from{" "}
            <code className="kbd">(source, external_key)</code> as it exports, so a set taken
            after a rebuild still points at rows that exist.
          </p>
        </>
      )}
    </div>
  );
}
