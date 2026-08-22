/**
 * What `memoryos stats` and `memoryos doctor` print, rendered.
 *
 * Both come from the same functions the CLI calls, so this cannot disagree with a
 * terminal. `doctor` is fetched on demand rather than on mount: it tokenizes
 * candidate chunks with the real tokenizer, which is a diagnostic somebody asks
 * for, not something a page should poll.
 */

import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../../api/client";
import { Empty, Failure, Loading, Meta, SectionHeading } from "../../components/primitives";
import { count, percent } from "../../lib/format";

export function CorpusPage() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
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
          {/* The graph half of the corpus. Here as well as on the overview
              because this is the page somebody opens to find out why an
              entity-scoped view is empty, and the answer is usually one of
              these two numbers. */}
          <Figure label="entities" value={count(data.entities)} note="unmerged" />
          <Figure
            label="relationships"
            value={count(data.relationships)}
            // Distinct claims, not stored rows: the same relationship asserted
            // in five chunks is five rows deliberately, and reporting those as
            // "relationships" would overstate the graph fivefold.
            note={data.relationships === 0 ? "none extracted" : "distinct claims"}
          />
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

      {/* The sources table lived here until M10.0 and now lives on `/sources`,
          where it has the two controls it always needed — register, and sync.
          Moved rather than copied: a read-only duplicate of a page you can act
          on is a page somebody edits and then wonders why nothing changed. */}
      <p className="meta text-ink-3">
        Sources moved to{" "}
        <Link to="/sources" className="text-accent underline">
          sources
        </Link>
        , where they can be registered and synced rather than only counted.
      </p>

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
          <p className="meta text-ink-3">
            Not run. It tokenizes candidate chunks with the real tokenizer, so it costs a moment.
          </p>
        ) : null}
        {doctor.data ? (
          <>
            <p className={`meta ${doctor.data.healthy ? "text-affirm" : "text-deny"}`}>
              {doctor.data.healthy ? "healthy" : "problems found"}
            </p>
            {/* A doctor run that returned no checks at all. Not the same as a
                healthy one — healthy is a list of `ok` rows — and rendering it
                as an empty `<ul>` under a green "healthy" would claim the
                corpus passed checks that never ran. */}
            {doctor.data.findings.length === 0 ? (
              <Empty title="No checks reported">
                The run completed but returned no checks, which usually means the
                API is a version behind this page. Re-run it, or check the server
                log for a check that failed to load.
              </Empty>
            ) : null}
            <ul>
              {doctor.data.findings.map((finding) => {
                // Three states, not two. A non-zero advisory is a capability
                // nobody has exercised rather than damage, and rendering it as
                // a green "ok" is how a corpus with no entity extraction at all
                // stayed invisible through two full replays.
                const note = finding.advisory && finding.count > 0;
                return (
                <li key={finding.check} className="border-b border-rule/60 py-1.5">
                  <div className="flex items-baseline gap-3">
                    <span
                      className={`meta ${
                        note ? "text-accent" : finding.healthy ? "text-affirm" : "text-deny"
                      }`}
                    >
                      {note ? "note" : finding.healthy ? "ok" : "FAIL"}
                    </span>
                    <span className="meta text-ink">{finding.check}</span>
                    <span className="meta text-ink-3">{count(finding.count)}</span>
                  </div>
                  {!finding.healthy || note ? (
                    <p className="meta mt-0.5 max-w-prose text-ink-2">{finding.detail}</p>
                  ) : null}
                  {finding.examples.length > 0 && (!finding.healthy || note) ? (
                    <ul className="mt-0.5">
                      {finding.examples.map((example) => (
                        <li key={example} className="meta text-ink-3">
                          {example}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </li>
                );
              })}
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
          <span className={`meta ml-1.5 ${alarm ? "text-deny" : "text-ink-3"}`}>{note}</span>
        ) : null}
      </dd>
    </div>
  );
}
