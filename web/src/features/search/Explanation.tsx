/**
 * Why a result is where it is, and what it quotes.
 *
 * Diagnostics, not a redesign. Collapsed by default because the ranking is what
 * a reader came for; expanded it answers the one question a ranked list cannot
 * — "why is *that* third?" — with the share bars, which is the only number that
 * answers it directly.
 *
 * Nothing here computes anything. The API sends shares, offsets and the
 * sentence already assembled from the numbers, precisely so that the UI cannot
 * disagree with the ranker about why something ranked.
 */

import { useState } from "react";
import { Link } from "react-router-dom";

import type { Citation, Explanation } from "../../api/client";
import { Highlighted } from "../../components/Highlighted";

interface Props {
  explanation: Explanation | null | undefined;
  citations: Citation[] | null | undefined;
  code: boolean;
}

export function ExplanationPanel({ explanation, citations, code }: Props) {
  const [open, setOpen] = useState(false);

  if (!explanation) return null;

  return (
    <div className="mt-2 pl-25" data-testid="explanation">
      <button
        type="button"
        className="meta text-amber hover:text-amber-bright"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        {open ? "−" : "+"} why this ranked
      </button>

      {/* The sentence is always visible. It is one line, it is free, and it is
          the whole explanation for most readers. */}
      <p className="meta mt-1 text-muted">{explanation.why}</p>

      {open ? (
        <div className="mt-2 space-y-3 border-l border-rule pl-3">
          <table className="w-full">
            <tbody>
              {explanation.contributions.map((item) => (
                <tr key={item.name} data-testid="contribution">
                  <td className="meta w-24 py-0.5 text-muted">{item.name}</td>
                  <td className="meta w-16 py-0.5 text-faint">rank {item.rank}</td>
                  <td className="meta w-14 py-0.5 text-right text-ink">
                    {(item.share * 100).toFixed(0)}%
                  </td>
                  <td className="py-0.5 pl-2">
                    {/* A bar rather than a number alone: the comparison between
                        two signals is the point, and a bar is read faster than
                        two percentages. */}
                    <span
                      className="inline-block h-1.5 bg-amber align-middle"
                      style={{ width: `${Math.max(2, item.share * 100)}%` }}
                      aria-hidden
                    />
                  </td>
                </tr>
              ))}
              {explanation.rerank_score !== null &&
              explanation.rerank_score !== undefined ? (
                <tr>
                  <td className="meta w-24 py-0.5 text-muted">reranked</td>
                  <td className="meta py-0.5 text-faint" colSpan={3}>
                    cross-encoder score {explanation.rerank_score.toFixed(3)}
                  </td>
                </tr>
              ) : null}
              {/* The entity route, when the graph is what put this result here.
                  Shown as its own row rather than folded into the share table,
                  because it is the only contribution a reader cannot check against
                  the text in front of them: a result the retrievers never found
                  shares no word with the query, and the route is the whole
                  argument for it being here. */}
              {explanation.graph_path ? (
                <tr>
                  <td className="meta w-24 py-0.5 text-muted">graph route</td>
                  <td className="meta py-0.5 text-ink" colSpan={3} data-testid="graph-path">
                    {explanation.graph_path}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>

          {(citations ?? []).map((citation) => (
            <CitationBlock
              key={`${citation.chunk_ordinal}-${citation.char_start}`}
              citation={citation}
              code={code}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

/**
 * One citation: where it came from, and the quoted span in its context.
 *
 * The link carries the offset, so opening the memory lands on the cited span
 * rather than at the top of a file the reader then has to search.
 */
function CitationBlock({ citation, code }: { citation: Citation; code: boolean }) {
  const context = citation.context;

  return (
    <div data-testid="citation">
      <div className="flex flex-wrap items-baseline gap-2">
        <Link
          to={`/memory/${citation.memory_id}?offset=${citation.char_start}`}
          className="meta text-amber underline decoration-rule-strong underline-offset-2 hover:decoration-edge"
        >
          #{citation.chunk_ordinal} @{citation.char_start}–{citation.char_end}
        </Link>
        <span className="meta text-faint">v{citation.version}</span>
        {citation.definition ? (
          <span className="meta text-amber" data-testid="citation-definition">
            {citation.definition}()
          </span>
        ) : null}
      </div>

      {context ? (
        <div className="mt-1">
          {/* Context muted, the cited span at full strength. `Highlighted` takes
              offsets *into the excerpt*, which the API computed — the UI never
              redoes that arithmetic, because that is the arithmetic M1.4a got
              wrong. */}
          <Highlighted
            text={context.text}
            charStart={context.span_start}
            charEnd={context.span_end}
            code={code}
            full
            absolute
          />
        </div>
      ) : (
        <div className={code ? "code-content mt-1" : "prose-content mt-1"}>
          {citation.excerpt}
        </div>
      )}
    </div>
  );
}
