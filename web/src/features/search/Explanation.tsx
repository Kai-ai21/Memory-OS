/**
 * Why a result is where it is, and what it quotes.
 *
 * **The strongest thing in the Luminous reference, and the one screen where the
 * real data and the mockup disagree about something that matters.** The mockup
 * draws four signals — SEMANTIC 52%, KEYWORD 31%, RECENCY 12%, GRAPH 5% — with
 * the first two in accent and the last two in warn. Against this backend only
 * two of those four can ever have a number, and it is not an oversight in the
 * ranker: `weight_recency`, `weight_importance` and `weight_graph` are all 0.0
 * in `config.py`, each with a measurement written above it saying why. Recency
 * monotonically lowered nDCG; graph expansion at 0.5 was arithmetically inert
 * and at 1.0 did measurable harm.
 *
 * So all four rows render, always, and the two that did not contribute say so
 * instead of showing a fabricated percentage. A row reading 0% would assert
 * that the signal ran and found nothing, which is a different and false claim.
 *
 * **What the response cannot tell us, this panel does not pretend to know.**
 * `build_explanation` drops a ranking from `contributions` for two different
 * reasons — its weight is zero, or it never returned this chunk — and the
 * serialised form keeps no trace of which. The row therefore says "no
 * contribution" rather than "switched off", and the note under the table names
 * both possibilities. Distinguishing them needs a field the API does not send;
 * see the milestone report.
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

/**
 * The four the reference draws, in its order, with its colours.
 *
 * Cyan is a retriever that read the text; warn is a signal that reordered or
 * introduced the result on grounds the reader cannot check by looking at it.
 * That is the same distinction the palette carries everywhere else in the
 * application, and it is why the reference's colour split is worth keeping even
 * though two of these rows are currently always empty.
 */
const SIGNALS = [
  { name: "semantic", tone: "accent" },
  { name: "keyword", tone: "accent" },
  { name: "recency", tone: "warn" },
  { name: "graph", tone: "warn" },
] as const;

export function ExplanationPanel({ explanation, citations, code }: Props) {
  const [open, setOpen] = useState(false);

  if (!explanation) return null;

  const byName = new Map(explanation.contributions.map((item) => [item.name, item]));
  // Anything the ranker contributed that the reference's four do not name —
  // `importance`, or a signal added later. Appended rather than dropped: a
  // panel that hid a live contribution because the mockup had no row for it
  // would be lying by omission about the ranking it exists to explain.
  const extra = explanation.contributions.filter(
    (item) => !SIGNALS.some((signal) => signal.name === item.name),
  );

  return (
    <div className="mt-3 lg:pl-25" data-testid="explanation">
      <button
        type="button"
        className="meta text-accent hover:text-accent-strong"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        {open ? "−" : "+"} why this ranked
      </button>

      {/* The sentence is always visible. It is one line, it is free, and it is
          the whole explanation for most readers. */}
      <p className="meta mt-1 text-ink-2">{explanation.why}</p>

      {open ? (
        <div className="mt-3 space-y-4">
          {/* Written for somebody who has not read the ranker: "semantic 55%"
              is meaningless without knowing that several retrievers run and
              their opinions are combined. */}
          <p className="prose-content max-w-prose text-sm text-ink-2">
            Several retrievers look for this result independently and their opinions are
            combined. Each row is one of them: where it placed this result, and how much
            of the final score came from it.
          </p>

          <div className="panel p-5">
            <div className="flex flex-col gap-4" data-testid="signals">
              {SIGNALS.map((signal) => (
                <SignalRow
                  key={signal.name}
                  name={signal.name}
                  tone={signal.tone}
                  item={byName.get(signal.name)}
                />
              ))}
              {extra.map((item) => (
                <SignalRow key={item.name} name={item.name} tone="accent" item={item} />
              ))}
            </div>

            <div className="mt-5 space-y-2 border-t border-rule pt-4">
              <p className="prose-content text-sm text-ink-2" data-testid="why-sentence">
                {explanation.why}
              </p>

              {/* Stated once, under the table, rather than as a tooltip on each
                  empty row. Both halves are true and the response does not say
                  which — see the file header. */}
              {SIGNALS.some((signal) => !byName.has(signal.name)) ? (
                <p className="meta text-ink-3" data-testid="absent-note">
                  A signal with no contribution either carries a fusion weight of zero or
                  did not return this result. The search response does not distinguish
                  the two.
                </p>
              ) : null}

              {explanation.rerank_score !== null &&
              explanation.rerank_score !== undefined ? (
                <p className="meta text-ink-3">
                  A second model read the query and this text together and scored the
                  pair{" "}
                  <span className="text-ink">{explanation.rerank_score.toFixed(3)}</span>.
                </p>
              ) : null}

              {/* The entity route, when the graph is what put this result here.
                  Its own line rather than folded into the table, because it is
                  the only contribution a reader cannot check against the text in
                  front of them: a result the retrievers never found shares no
                  word with the query, and the route is the whole argument for it
                  being here. */}
              {explanation.graph_path ? (
                <p className="meta text-warn" data-testid="graph-path">
                  reached through the entity graph: {explanation.graph_path}
                </p>
              ) : null}
            </div>
          </div>

          {citations && citations.length > 0 ? (
            <p className="meta-label">the passages it quoted</p>
          ) : null}
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
 * One signal: its name, its share as a lit bar, and its share as a number.
 *
 * A bar *and* a number because they answer different questions. The number is
 * what you quote; the bar is what lets you see at a glance that one signal
 * carried this result and the other three did not, which is the thing a column
 * of percentages makes you compute.
 *
 * The bar is 3px and solid. On dark it was 2px with a glow; on light a 2px
 * bar at 5% width is a dot, and there is no glow to make up the difference, so
 * it gained a pixel instead.
 */
function SignalRow({
  name,
  tone,
  item,
}: {
  name: string;
  tone: "accent" | "warn";
  item?: { name: string; rank: number; share: number };
}) {
  const share = item ? item.share * 100 : 0;
  const fill = tone === "accent" ? "bg-accent" : "bg-warn";

  return (
    <div className="flex items-center gap-4" data-testid="contribution" data-signal={name}>
      <span className={`meta-label w-20 shrink-0 ${item ? "text-ink-2" : "text-ink-3"}`}>
        {name}
      </span>
      <div className="relative h-[3px] flex-1 overflow-hidden rounded-full bg-rule">
        {item ? (
          <div
            className={`absolute inset-y-0 left-0 ${fill}`}
            // Floored at 2% so a signal that contributed almost nothing is still
            // visibly a signal that contributed something, rather than an empty
            // track indistinguishable from one that did not run.
            style={{ width: `${Math.max(2, share)}%` }}
            aria-hidden
          />
        ) : null}
      </div>
      {item ? (
        <>
          <span className="meta w-10 shrink-0 text-right text-ink-3">#{item.rank}</span>
          <span
            className={`meta w-12 shrink-0 text-right ${tone === "accent" ? "text-accent" : "text-warn"}`}
          >
            {share.toFixed(0)}%
          </span>
        </>
      ) : (
        <span className="meta w-22 shrink-0 text-right text-ink-3">no contribution</span>
      )}
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
          className="meta text-accent underline decoration-rule-strong underline-offset-2 hover:decoration-edge"
        >
          #{citation.chunk_ordinal} @{citation.char_start}–{citation.char_end}
        </Link>
        <span className="meta text-ink-3">v{citation.version}</span>
        {citation.definition ? (
          <span className="meta text-accent" data-testid="citation-definition">
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
