/**
 * One memory in full, with chunk boundaries drawn on the real content.
 *
 * This is the diagnostic view, and the boundaries are the reason it exists. A
 * chunker splitting mid-thought is invisible in every test — the row counts are
 * right, the offsets are right, the tokens are under the window — and obvious the
 * moment you see where the rules fall in a paragraph.
 *
 * The content is rendered once, in document order, with a rule at each boundary,
 * rather than as a list of chunk texts. Concatenating chunk texts would hide
 * exactly what is being inspected, because the overlap prefix means neighbouring
 * chunks share text and the seams would not be where the rules are.
 */

import { useEffect, useMemo } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api, type MemoryChunk } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { RelativeTime } from "../../components/RelativeTime";
import { Evolution } from "./Evolution";
import { CopyButton, Empty, Failure, Loading, Meta, SectionHeading, Tag } from "../../components/primitives";
import { count, isCode, range, shortHash } from "../../lib/format";
import { VersionTimeline } from "./VersionTimeline";
import { recordRecent } from "../../lib/recents";

export function MemoryPage({ id: given }: { id?: string } = {}) {
  /* The route's id, unless one was handed in. The split panel renders this
     same component beside another page — see `app/SplitPanel` — where there is
     no route parameter to read, and forking a second memory view so the two
     could drift is exactly what should not happen to the most detailed screen
     in the application. */
  const { id: routeId = "" } = useParams();
  const id = given ?? routeId;
  // `?offset=` is where a citation points. Landing on the cited span rather than
  // at the top of the file is the difference between a reference and a
  // suggestion to go looking.
  const [params] = useSearchParams();
  const offset = Number(params.get("offset"));
  const target = Number.isFinite(offset) && params.has("offset") ? offset : null;
  const memory = useQuery({ queryKey: ["memory", id], queryFn: () => api.memory(id) });

  /* Opened memories are the half of `recents` the palette cannot record for
     itself — a memory reached by clicking a search result never goes through
     the palette, and a path is exactly the kind of address nobody can retype.
     Keyed on the id rather than on the object, so a background refetch does not
     re-record the same visit. */
  const externalKey = memory.data?.external_key;
  useEffect(() => {
    if (!id || !externalKey) return;
    recordRecent({ to: `/memory/${id}`, label: externalKey, kind: "memory" });
  }, [id, externalKey]);

  if (memory.isLoading) return <Loading rows={6} />;
  if (memory.isError) return <Failure error={memory.error} />;
  if (!memory.data) return null;

  const detail = memory.data;
  const code = isCode(detail.kind);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <div className="flex flex-wrap items-baseline gap-3">
          {/* Mono rather than the display face, and deliberately: this heading
              is a path. Setting a filename in a display serif makes it look
              like a title, and the one thing a reader must be able to do with
              it is match it character for character against a shell. */}
          <span className="group flex items-baseline gap-1.5">
            <h1 className="font-mono text-base break-all text-ink">{detail.external_key}</h1>
            <CopyButton value={detail.external_key} label="path" />
          </span>
          <Tag>{detail.kind}</Tag>
          {detail.deleted_at ? <Tag>deleted</Tag> : null}
          {!detail.is_current ? <Tag>superseded</Tag> : null}
        </div>
        {detail.title ? <p className="prose-content text-ink-2">{detail.title}</p> : null}
        <div className="flex flex-wrap gap-x-5 gap-y-1">
          <Meta label="source">{detail.source_name}</Meta>
          <Meta label="version">{detail.version}</Meta>
          {/* The provenance rides the date rather than sitting in its own field.
              As a separate `Meta` it read as one more attribute; attached, it
              qualifies the number it is about, which is what it does. */}
          <Meta label="occurred">
            <DateStamp
              value={detail.occurred_at}
              provenance={detail.occurred_at_source}
              showProvenance
            />
          </Meta>
          <Meta label="ingested">
            <RelativeTime value={detail.ingested_at} />
          </Meta>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-1">
          {/* Both hashes. The pair is the whole story of M1.4: different bytes
              with identical normalized text is a new artifact and no re-chunking. */}
          {/* The shortened hash is for reading; the copy button carries the
              whole thing. A truncated hash that silently copies as truncated is
              worse than no copy button — it produces a value that looks right
              and matches nothing. */}
          <Meta label="content hash">
            <span className="group inline-flex items-baseline gap-1">
              <span title={detail.content_hash}>{shortHash(detail.content_hash)}</span>
              {detail.content_hash ? (
                <CopyButton value={detail.content_hash} label="content hash" />
              ) : null}
            </span>
          </Meta>
          <Meta label="normalized hash">
            <span className="group inline-flex items-baseline gap-1">
              <span title={detail.normalized_hash ?? ""}>{shortHash(detail.normalized_hash)}</span>
              {detail.normalized_hash ? (
                <CopyButton value={detail.normalized_hash} label="normalized hash" />
              ) : null}
            </span>
          </Meta>
          <Meta label="chunks">{count(detail.chunks.length)}</Meta>
          {/* The id is not shown anywhere else in the interface and is what
              every CLI command and log line refers to a memory by. Shortened
              like the hashes, and copied in full for the same reason. */}
          <Meta label="id">
            <span className="group inline-flex items-baseline gap-1">
              <span title={detail.id}>{shortHash(detail.id, 8)}</span>
              <CopyButton value={detail.id} label="memory id" />
            </span>
          </Meta>
        </div>
      </header>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${detail.chunks.length} boundaries`}>
          content with chunk boundaries
        </SectionHeading>
        <BoundedContent
          content={detail.content}
          chunks={detail.chunks}
          code={code}
          target={target}
        />
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading>chunks</SectionHeading>
        {/* A memory can genuinely have none: ingestion stores the artifact
            first and chunks it after, so anything caught between those two —
            or anything whose chunker failed — is a real row with an empty
            table. Before this the page rendered a bare header with no body,
            which reads as a broken table rather than as a fact about the
            memory. */}
        {detail.chunks.length === 0 ? (
          <Empty title="Not chunked">
            This memory was stored but never split into chunks, so nothing about it
            is searchable yet. A re-ingest of its source will chunk and embed it.
          </Empty>
        ) : (
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-rule">
              {["#", "range", "tokens", "chunker", "model", "embedded", "definition"].map(
                (heading) => (
                  <th key={heading} className="meta-label py-1 pr-4 font-normal">
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {detail.chunks.map((chunk) => (
              <tr key={chunk.id} className="border-b border-rule/60">
                <td className="meta py-1 pr-4 text-ink">{chunk.ordinal}</td>
                <td className="meta py-1 pr-4">{range(chunk.char_start, chunk.char_end)}</td>
                <td className="meta py-1 pr-4">{chunk.token_count}</td>
                <td className="meta py-1 pr-4 text-ink-3">{chunk.chunker_version}</td>
                <td className="meta py-1 pr-4 text-ink-3">{chunk.embedding_model ?? "—"}</td>
                <td className="meta py-1 pr-4">
                  {chunk.embedded ? <RelativeTime value={chunk.embedded_at} /> : "not embedded"}
                </td>
                <td className="meta py-1 pr-4 text-accent">
                  {typeof chunk.metadata?.definition === "string"
                    ? chunk.metadata.definition
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${detail.versions.length} versions`}>
          evolution
        </SectionHeading>
        <Evolution memoryId={detail.id} kind={detail.kind} />
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${detail.versions.length} versions`}>
          version history
        </SectionHeading>
        <VersionTimeline versions={detail.versions} />
        {detail.versions.length > 1 ? (
          <div className="flex flex-wrap gap-x-4">
            {detail.versions
              .filter((version) => version.id !== detail.id)
              .map((version) => (
                <Link
                  key={version.id}
                  to={`/memory/${version.id}`}
                  className="meta text-accent underline"
                >
                  open v{version.version}
                </Link>
              ))}
          </div>
        ) : null}
      </section>
    </div>
  );
}

/**
 * The content, with a rule wherever a chunk begins.
 *
 * Boundaries are taken from `char_start`, deduplicated and sorted — chunks
 * overlap, so several can begin inside the previous one, and drawing every
 * `char_end` too would produce a rule every few words and say nothing.
 */
function BoundedContent({
  content,
  chunks,
  code,
  target,
}: {
  content: string | null;
  chunks: MemoryChunk[];
  code: boolean;
  /** Where a citation pointed, so the cited slice can be marked and scrolled to. */
  target?: number | null;
}) {
  const pieces = useMemo(() => {
    if (!content) return [];
    const starts = [...new Set(chunks.map((chunk) => chunk.char_start))]
      .filter((offset) => offset > 0 && offset < content.length)
      .sort((a, b) => a - b);

    const slices: { from: number; text: string; ordinal: number | null }[] = [];
    let cursor = 0;
    for (const start of starts) {
      slices.push({
        from: cursor,
        text: content.slice(cursor, start),
        ordinal: ordinalStartingAt(chunks, cursor),
      });
      cursor = start;
    }
    slices.push({
      from: cursor,
      text: content.slice(cursor),
      ordinal: ordinalStartingAt(chunks, cursor),
    });
    return slices.filter((slice) => slice.text.length > 0);
  }, [content, chunks]);

  if (!content) {
    return (
      <p className="meta text-ink-3">
        no normalized text — this memory has not been normalized, or parsed to nothing
      </p>
    );
  }

  return (
    <div className="border border-rule bg-surface" data-testid="bounded-content">
      {pieces.map((piece, index) => {
        // The slice a citation landed on. Marked rather than scrolled-to with a
        // ref: the page is short enough that an anchor is enough, and a
        // highlight survives the reader scrolling away and back.
        const cited =
          target !== null &&
          target !== undefined &&
          target >= piece.from &&
          target < piece.from + piece.text.length;
        return (
        <div
          key={piece.from}
          id={`offset-${piece.from}`}
          className={`${index > 0 ? "border-t border-dashed border-edge/45" : ""}${
            cited ? " bg-accent/10" : ""
          }`}
          data-testid="boundary-slice"
          data-cited={cited ? "true" : undefined}
        >
          <div className="flex gap-3 px-3 py-2">
            <span className="meta w-8 shrink-0 pt-0.5 text-right text-ink-3">
              {piece.ordinal !== null ? `#${piece.ordinal}` : ""}
            </span>
            <div className={code ? "code-content flex-1" : "prose-content flex-1"}>
              {piece.text}
            </div>
          </div>
        </div>
        );
      })}
    </div>
  );
}

function ordinalStartingAt(chunks: MemoryChunk[], offset: number): number | null {
  const match = chunks.find((chunk) => chunk.char_start === offset);
  return match ? match.ordinal : null;
}
