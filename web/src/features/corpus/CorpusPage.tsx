/**
 * What `memoryos stats` and `memoryos doctor` print, rendered.
 *
 * Both come from the same functions the CLI calls, so this cannot disagree with a
 * terminal. `doctor` is fetched on demand rather than on mount: it tokenizes
 * candidate chunks with the real tokenizer, which is a diagnostic somebody asks
 * for, not something a page should poll.
 */

import { useQuery } from "@tanstack/react-query";

import { api } from "../../api/client";
import { Failure, Loading, Meta, SectionHeading } from "../../components/primitives";
import { count, percent, timestamp } from "../../lib/format";

export function CorpusPage() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });
  const doctor = useQuery({
    queryKey: ["doctor"],
    queryFn: api.doctor,
    // On demand. See the module note.
    enabled: false,
  });

  if (stats.isLoading) return <Loading rows={4} />;
  if (stats.isError) return <Failure error={stats.error} />;
  if (!stats.data) return null;

  const data = stats.data;
  const unembedded = data.chunks - data.embedded_chunks;

  return (
    <div className="flex flex-col gap-6">
      <section className="flex flex-col gap-2">
        <SectionHeading>corpus</SectionHeading>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-1 sm:grid-cols-3 lg:grid-cols-4">
          <Figure label="memories" value={count(data.memories)} note={`${count(data.current_memories)} current`} />
          <Figure label="chunks" value={count(data.chunks)} />
          <Figure
            label="embedded"
            value={count(data.embedded_chunks)}
            note={percent(data.coverage)}
            alarm={unembedded > 0}
          />
          <Figure label="cache entries" value={count(data.cache_entries)} />
        </dl>
        <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1">
          <Meta label="model">{data.embedding_model}</Meta>
          <Meta label="window">{data.model_window} tokens</Meta>
          <Meta label="chunker">{data.chunker_version}</Meta>
        </div>
        {Object.keys(data.models).length > 1 ? (
          <p className="meta text-deny">
            chunks from more than one model: {Object.keys(data.models).join(", ")} — vectors from
            different models are not comparable
          </p>
        ) : null}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading>sources</SectionHeading>
        {sources.isError ? (
          <Failure error={sources.error} />
        ) : (
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-rule">
                {["name", "kind", "memories", "chunks", "last sync", "last full sync"].map(
                  (heading) => (
                    <th key={heading} className="meta-label py-1 pr-4 font-normal">
                      {heading}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {(sources.data ?? []).map((source) => (
                <tr key={source.id} className="border-b border-rule/60">
                  <td className="meta py-1 pr-4 text-ink">{source.name}</td>
                  <td className="meta py-1 pr-4 text-faint">{source.kind}</td>
                  <td className="meta py-1 pr-4">{count(source.memories)}</td>
                  <td className="meta py-1 pr-4">{count(source.chunks)}</td>
                  <td className="meta py-1 pr-4 text-faint">{timestamp(source.last_sync_at)}</td>
                  <td className="meta py-1 pr-4 text-faint">
                    {timestamp(source.last_full_sync_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading
          right={
            <button
              type="button"
              className="btn"
              onClick={() => void doctor.refetch()}
              disabled={doctor.isFetching}
            >
              {doctor.isFetching ? "checking…" : doctor.data ? "re-check" : "run doctor"}
            </button>
          }
        >
          health checks
        </SectionHeading>

        {doctor.isError ? <Failure error={doctor.error} /> : null}
        {!doctor.data && !doctor.isFetching ? (
          <p className="meta text-faint">
            Not run. It tokenizes candidate chunks with the real tokenizer, so it costs a moment.
          </p>
        ) : null}
        {doctor.data ? (
          <>
            <p className={`meta ${doctor.data.healthy ? "text-affirm" : "text-deny"}`}>
              {doctor.data.healthy ? "healthy" : "problems found"}
            </p>
            <ul>
              {doctor.data.findings.map((finding) => (
                <li key={finding.check} className="border-b border-rule/60 py-1.5">
                  <div className="flex items-baseline gap-3">
                    <span className={`meta ${finding.healthy ? "text-affirm" : "text-deny"}`}>
                      {finding.healthy ? "ok" : "FAIL"}
                    </span>
                    <span className="meta text-ink">{finding.check}</span>
                    <span className="meta text-faint">{count(finding.count)}</span>
                  </div>
                  {!finding.healthy ? (
                    <p className="meta mt-0.5 max-w-prose text-muted">{finding.detail}</p>
                  ) : null}
                  {finding.examples.length > 0 && !finding.healthy ? (
                    <ul className="mt-0.5">
                      {finding.examples.map((example) => (
                        <li key={example} className="meta text-faint">
                          {example}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </section>
    </div>
  );
}

function Figure({
  label,
  value,
  note,
  alarm,
}: {
  label: string;
  value: string;
  note?: string;
  alarm?: boolean;
}) {
  return (
    <div className="border-l-2 border-rule-strong pl-2">
      <dt className="meta-label">{label}</dt>
      <dd className="font-mono text-lg text-ink">
        {value}
        {note ? (
          <span className={`meta ml-1.5 ${alarm ? "text-deny" : "text-faint"}`}>{note}</span>
        ) : null}
      </dd>
    </div>
  );
}
