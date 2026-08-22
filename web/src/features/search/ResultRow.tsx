/**
 * One memory hit, with its matched chunks.
 *
 * The information hierarchy is the design: score first because it is what you
 * scan, then `external_key` because it is what you recognise, then the matched
 * text because it is what you judge. Kind, time and chunk provenance sit in mono
 * metadata that stays out of the way until looked at.
 *
 * Expanding a chunk fetches the parent memory and shows the neighbouring chunks
 * by ordinal, which is what makes "chunk 7 matched" interpretable. The highlight
 * does not depend on that fetch: it is arithmetic on the offsets the search
 * response already carries (see `lib/highlight.ts`), so every result is legible
 * the moment it arrives and the list view stays a single request.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api, type MatchedChunk, type MemoryHit } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Highlighted } from "../../components/Highlighted";
import { CopyButton, Tag } from "../../components/primitives";
import { isCode, range, score as fmtScore } from "../../lib/format";
import { JudgementButtons, type JudgementTarget } from "../judgements/JudgementButtons";
import { ExplanationPanel } from "./Explanation";
import type { Verdict } from "../../api/client";
import { SplitOpenButton } from "../../app/SplitPanel";

interface Props {
  hit: MemoryHit;
  rank: number;
  queryText: string;
  sourceName: string;
  filters: Record<string, unknown>;
  verdict?: Verdict | null;
  /** Keyed `externalKey#ordinal`, so a chunk verdict is separate from the file's. */
  chunkVerdicts?: Record<string, Verdict>;
  onJudged?: (key: string, verdict: Verdict) => void;
}

/** The key a chunk-level verdict is remembered under for this session. */
function chunkVerdictKey(externalKey: string, ordinal: number): string {
  return `${externalKey}#${ordinal}`;
}

export function ResultRow({
  hit,
  rank,
  queryText,
  sourceName,
  filters,
  verdict,
  chunkVerdicts,
  onJudged,
}: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [showAllChunks, setShowAllChunks] = useState(false);
  const code = isCode(hit.kind);

  // The search use case fetches five chunks per requested memory to get k
  // distinct memories, so a single hit can arrive with five. Rendering all of
  // them made ten results nineteen screens long; two is enough to see *why* the
  // memory ranked, and the rest are one click away.
  const visibleChunks = showAllChunks ? hit.matched_chunks : hit.matched_chunks.slice(0, 2);
  const hidden = hit.matched_chunks.length - visibleChunks.length;

  // Only fetched once something is expanded, and only for the neighbouring
  // chunks — the highlight itself is arithmetic on the search response, so the
  // list view is one request and every result is legible immediately.
  const detail = useQuery({
    queryKey: ["memory", hit.memory_id],
    queryFn: () => api.memory(hit.memory_id),
    enabled: expanded !== null,
  });

  const target: JudgementTarget = {
    queryText,
    sourceName,
    externalKey: hit.external_key,
    memoryId: hit.memory_id,
    rank,
    score: hit.score,
    filters,
  };

  return (
    /* A hairline-separated row rather than a boxed card. Ten bordered white
       boxes on a near-white ground is ten rectangles competing for the same
       edge, and the ranking — which is what the page is for — stops being
       scannable. One rule between rows says "next result" with a tenth of the
       ink. The expanded state takes an ink inset bar, not the accent: an
       opened row is a position, not an action. */
    <article
      className={`border-b border-rule py-5 last:border-b-0 ${
        expanded !== null ? "-mx-4 border-l-2 border-l-ink bg-surface px-4" : ""
      }`}
      data-testid="result"
    >
      {/* Wraps, and the path is the thing that must survive it.
          Unwrapped, at 768 the flex row squeezed `flex-1 min-w-0 truncate` to
          zero and the external key — the one element that says *which result
          this is* — disappeared entirely while the judgement buttons stayed.
          A basis wide enough to be worth reading forces the metadata onto a
          second line instead. */}
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        {/* Rank and score in a fixed-width gutter, so scores form a column that
            can be compared down the page rather than a ragged edge. */}
        <span className="meta w-6 shrink-0 text-right text-ink-3">{rank}</span>
        {/* The score in ink at the head of the row, bold and a size up. On dark
            this was cyan, because a column of white numbers on a void is not
            findable; on light, weight does that job and the accent stays with
            the link beside it. */}
        <span
          className="w-16 shrink-0 font-mono text-base font-semibold text-ink"
          data-testid="score"
        >
          {fmtScore(hit.score)}
        </span>
        {/* The path and the control that copies it, as one unit. `group` is
            what `CopyButton` hangs its reveal off — the button is invisible
            until this row is hovered or something in it has focus, so a page
            of twenty results is not a page of twenty clipboard icons. */}
        <span className="group flex min-w-48 flex-1 basis-48 items-baseline gap-1">
          <Link
            to={`/memory/${hit.memory_id}`}
            // What the arrow keys walk. See `SearchPage.step` — the row's other
            // controls are reachable by tab, but the arrows move between results
            // rather than between every focusable thing inside one.
            data-result-link
            className="truncate font-mono text-sm text-accent underline decoration-rule-strong decoration-1 underline-offset-2 hover:decoration-edge"
            title={hit.external_key}
          >
            {hit.external_key}
          </Link>
          <CopyButton value={hit.external_key} label="path" />
          {/* Read a result beside the list it came from, rather than instead
              of it. The comparison between hit three and hit four is the whole
              reason somebody is on this page. */}
          <SplitOpenButton memoryId={hit.memory_id} label={hit.external_key} />
        </span>
        <Tag>{hit.kind}</Tag>
        <span className="hidden shrink-0 sm:inline">
          <DateStamp value={hit.occurred_at} provenance={hit.occurred_at_source} />
        </span>
        <JudgementButtons
          target={target}
          current={verdict}
          onRecorded={(given) => onJudged?.(hit.external_key, given)}
        />
      </header>

      {hit.title ? <p className="meta mt-1 lg:pl-25 text-ink-2">{hit.title}</p> : null}

      <div className="mt-2 space-y-2 lg:pl-25">
        {visibleChunks.map((chunk) => (
          <ChunkBlock
            key={chunk.chunk_id}
            chunk={chunk}
            code={code}
            // The chunk-level target differs from the memory-level one only in
            // the ordinal and the chunk id. Rank and score stay the memory's,
            // because the ranking being judged is the memory ranking — the
            // ordinal narrows *what was right*, not *where it appeared*.
            target={{ ...target, chunkOrdinal: chunk.ordinal, chunkId: chunk.chunk_id }}
            verdict={chunkVerdicts?.[chunkVerdictKey(hit.external_key, chunk.ordinal)] ?? null}
            onJudged={(given) =>
              onJudged?.(chunkVerdictKey(hit.external_key, chunk.ordinal), given)
            }
            expanded={expanded === chunk.ordinal}
            onToggle={() =>
              setExpanded((current) => (current === chunk.ordinal ? null : chunk.ordinal))
            }
            neighbours={
              expanded === chunk.ordinal
                ? (detail.data?.chunks ?? []).filter(
                    (other) =>
                      Math.abs(other.ordinal - chunk.ordinal) === 1 &&
                      other.ordinal !== chunk.ordinal,
                  )
                : []
            }
            loadingNeighbours={expanded === chunk.ordinal && detail.isLoading}
          />
        ))}
        {hidden > 0 ? (
          <button
            type="button"
            className="meta text-accent hover:underline"
            onClick={() => setShowAllChunks(true)}
          >
            + {hidden} more matched {hidden === 1 ? "chunk" : "chunks"} in this memory
          </button>
        ) : null}
      </div>

      {/* Diagnostics, collapsed. The one-line `why` shows always; the shares and
          the cited spans are one click away, because the ranking is what the
          reader came for and this explains it rather than competing with it. */}
      <ExplanationPanel
        explanation={hit.explanation}
        citations={hit.citations}
        code={code}
      />
    </article>
  );
}

interface ChunkProps {
  chunk: MatchedChunk;
  code: boolean;
  target: JudgementTarget;
  verdict: Verdict | null;
  onJudged: (verdict: Verdict) => void;
  expanded: boolean;
  onToggle: () => void;
  neighbours: { ordinal: number; content: string; token_count: number }[];
  loadingNeighbours: boolean;
}

function ChunkBlock({
  chunk,
  code,
  target,
  verdict,
  onJudged,
  expanded,
  onToggle,
  neighbours,
  loadingNeighbours,
}: ChunkProps) {
  const definition =
    typeof chunk.metadata?.definition === "string" ? chunk.metadata.definition : null;

  return (
    <div className="min-w-0 border-l border-rule pl-3" data-testid="matched-chunk">
      {/* Wraps, and the definition truncates.
          A Python test name runs to sixty characters and this row does not
          wrap by default, so one of them stretched the whole result column to
          654px at a 375px viewport — and because the column then fitted the
          code, `.code-content`'s own scroll box never engaged. The widest
          element on a page sets the body's scroll width, so this row was
          making every view scroll sideways. */}
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1">
        <button
          type="button"
          className="meta text-accent hover:text-accent-strong"
          onClick={onToggle}
          aria-expanded={expanded}
        >
          {expanded ? "−" : "+"} #{chunk.ordinal}
        </button>
        <span className="meta text-ink-2">{fmtScore(chunk.score)}</span>
        <span className="meta text-ink-3">{range(chunk.char_start, chunk.char_end)}</span>
        {/* What M1.7 persisted the chunk metadata for: a citation that can name
            the function it came from rather than only the file. */}
        {definition ? (
          <span
            className="meta min-w-0 max-w-full truncate text-accent"
            data-testid="definition"
            title={`${definition}()`}
          >
            {definition}()
          </span>
        ) : null}
        {/* The verdict on *this chunk*, distinct from the one on the file. It is
            the only way to say "right file, wrong chunk" — the failure a
            memory-level golden set scores as a success. Compact and last in the
            row, because judging the file is still the common case. */}
        <span className="ml-auto" data-testid="chunk-judgement">
          <JudgementButtons target={target} current={verdict} onRecorded={onJudged} compact />
        </span>
      </div>

      <div className="mt-1">
        <Highlighted
          text={chunk.text}
          charStart={chunk.char_start}
          charEnd={chunk.char_end}
          code={code}
          // Expanded shows the whole borrowed lead-in as well as the neighbours.
          full={expanded}
          // Collapsed, this is a list to compare rather than a document to
          // read, so the clamp is the tight one.
          tight
        />
      </div>

      {expanded ? (
        <div className="mt-2 space-y-2 border-t border-dashed border-rule pt-2">
          <p className="meta-label text-ink-3">context by ordinal</p>
          {loadingNeighbours ? (
            <p className="meta text-ink-3">loading…</p>
          ) : neighbours.length === 0 ? (
            <p className="meta text-ink-3">no neighbouring chunks — this is the whole memory</p>
          ) : (
            neighbours
              .sort((a, b) => a.ordinal - b.ordinal)
              .map((other) => (
                <div key={other.ordinal} className="opacity-70">
                  <span className="meta text-ink-3">
                    #{other.ordinal} · {other.token_count} tok
                  </span>
                  <div className={code ? "code-content mt-1" : "prose-content mt-1"}>
                    {other.content}
                  </div>
                </div>
              ))
          )}
        </div>
      ) : null}
    </div>
  );
}
