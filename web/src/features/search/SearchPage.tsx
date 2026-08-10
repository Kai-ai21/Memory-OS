/**
 * The main view.
 *
 * All search state is in the URL (see `searchParams.ts`), so the component holds
 * only two pieces of genuinely local state: what is currently typed in the box
 * before it is submitted, and which verdicts have been recorded this session so
 * the buttons can show an active state without refetching.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api, type Verdict } from "../../api/client";
import { Empty, Failure, Loading } from "../../components/primitives";
import { count } from "../../lib/format";
import { Filters } from "./Filters";
import { ResultRow } from "./ResultRow";
import {
  DEFAULT_K,
  isRunnable,
  parseSearchState,
  toSearchParams,
  type SearchState,
} from "./searchParams";

export function SearchPage() {
  const [params, setParams] = useSearchParams();
  const state = useMemo(() => parseSearchState(params), [params]);

  const [draft, setDraft] = useState(state.q);
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const input = useRef<HTMLInputElement>(null);

  // The box follows the URL, so the back button visibly restores the query rather
  // than only the results.
  useEffect(() => setDraft(state.q), [state.q]);

  // Autofocus on load, and `/` from anywhere. The shortcut is deliberately not
  // bound while typing — `/` is a character, and a search box that swallows it
  // would be unusable for path queries.
  useEffect(() => {
    input.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable === true;
      if (event.key === "/" && !typing) {
        event.preventDefault();
        input.current?.focus();
      }
      if (event.key === "Escape") input.current?.blur();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const update = useCallback(
    (next: Partial<SearchState>) => {
      setParams(toSearchParams({ ...state, ...next }), { replace: false });
    },
    [setParams, state],
  );

  const results = useQuery({
    queryKey: ["search", state.q, state.k, state.sources, state.kind, state.exact],
    queryFn: () =>
      api.search({
        q: state.q,
        k: state.k,
        sources: state.sources,
        kind: state.kind ?? undefined,
        exact: state.exact,
      }),
    enabled: isRunnable(state),
    // A search is a question about a fixed corpus; re-asking it on every window
    // focus would burn a model forward pass to produce the same answer.
    staleTime: 60_000,
  });

  const filters = useMemo(
    () => ({ k: state.k, sources: state.sources, kind: state.kind, exact: state.exact }),
    [state],
  );

  return (
    <div className="flex flex-col gap-4">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          update({ q: draft.trim() });
        }}
        className="flex items-center gap-2"
        role="search"
      >
        <input
          ref={input}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="search the corpus"
          aria-label="Search query"
          className="field flex-1 font-mono text-sm"
          spellCheck={false}
          autoComplete="off"
        />
        <button type="submit" className="btn" disabled={!draft.trim()}>
          search
        </button>
        <span className="meta hidden text-faint md:inline">
          <span className="kbd">/</span> to focus
        </span>
      </form>

      <Filters state={state} onChange={update} />

      {/* Timing, in small mono. Present always so it is noticed when it moves —
          the point of showing it is to see when something gets slow. */}
      {results.data ? (
        <div className="meta flex items-baseline gap-4 border-y border-rule py-1 text-faint">
          <span data-testid="timing">
            embed {results.data.timing.embed_ms}ms · search {results.data.timing.search_ms}ms ·
            total {results.data.timing.total_ms}ms
          </span>
          <span>{count(results.data.hits.length)} memories</span>
          <span>
            {count(results.data.hits.reduce((sum, hit) => sum + hit.matched_chunks.length, 0))}{" "}
            chunks matched
          </span>
          {state.exact ? <span className="text-amber">exact scan</span> : null}
        </div>
      ) : null}

      {!isRunnable(state) ? (
        <Empty title="nothing searched yet">
          Type a question and press Enter. Results are memories, ranked by their best matching
          chunk; the matched span is highlighted, and <span className="kbd">+</span> on a chunk
          shows its neighbours by ordinal.
        </Empty>
      ) : results.isLoading ? (
        <Loading />
      ) : results.isError ? (
        <Failure error={results.error} />
      ) : results.data && results.data.hits.length === 0 ? (
        <Empty title="no results">
          Nothing matched <span className="text-ink">{state.q}</span>
          {state.sources.length || state.kind ? " under these filters" : ""}. Try{" "}
          {state.sources.length || state.kind ? (
            <>clearing the source and kind filters, or </>
          ) : null}
          raising <span className="kbd">k</span> above {state.k}, or rephrasing — this is semantic
          search, so wording matters more than keywords.
        </Empty>
      ) : (
        <div>
          {results.data?.hits.map((hit, index) => (
            <ResultRow
              key={hit.memory_id}
              hit={hit}
              rank={index + 1}
              queryText={state.q}
              sourceName={hit.source_name}
              filters={filters}
              verdict={verdicts[hit.external_key] ?? null}
              // One map for both granularities: chunk verdicts are stored under
              // `externalKey#ordinal`, which cannot collide with a bare key.
              chunkVerdicts={verdicts}
              onJudged={(key, verdict) =>
                setVerdicts((current) => ({ ...current, [key]: verdict }))
              }
            />
          ))}
        </div>
      )}

      <MissingCapture state={state} filters={filters} />
    </div>
  );
}

/**
 * Recording something the search *should* have returned.
 *
 * The only verdict that cannot be inferred from a ranking, and the only one that
 * makes recall measurable. It is a separate control because by definition its
 * subject is not on screen.
 */
function MissingCapture({
  state,
  filters,
}: {
  state: SearchState;
  filters: Record<string, unknown>;
}) {
  const [open, setOpen] = useState(false);
  const [key, setKey] = useState("");
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isRunnable(state)) return null;

  return (
    <section className="mt-2 border-t border-rule pt-3">
      {!open ? (
        <button type="button" className="btn" onClick={() => setOpen(true)}>
          + mark a missing result
        </button>
      ) : (
        <form
          className="flex flex-wrap items-center gap-2"
          onSubmit={async (event) => {
            event.preventDefault();
            setError(null);
            try {
              await api.judge({
                query_text: state.q,
                source_name: state.sources[0] ?? "self",
                external_key: key.trim(),
                verdict: "missing",
                // No rank and no score: the point is that it was not ranked.
                rank_at_judgement: null,
                score_at_judgement: null,
                filters,
              });
              setSaved(key.trim());
              setKey("");
            } catch (caught) {
              setError(caught instanceof Error ? caught.message : String(caught));
            }
          }}
        >
          <label className="meta-label text-muted" htmlFor="missing-key">
            should have ranked
          </label>
          <input
            id="missing-key"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            placeholder="external key, e.g. src/memoryos/application/sync.py"
            className="field min-w-80 flex-1 font-mono text-sm"
            spellCheck={false}
          />
          <button type="submit" className="btn" disabled={!key.trim()}>
            record
          </button>
          <button type="button" className="btn" onClick={() => setOpen(false)}>
            done
          </button>
          {saved ? <span className="meta text-affirm">recorded {saved}</span> : null}
          {error ? (
            <span className="meta text-deny" role="alert">
              {error}
            </span>
          ) : null}
        </form>
      )}
      <p className="meta mt-2 text-faint">
        k is {state.k}
        {state.k === DEFAULT_K ? " (default)" : ""} — a result below it is not missing, only
        unranked.
      </p>
    </section>
  );
}
