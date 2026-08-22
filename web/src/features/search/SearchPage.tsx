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
import { SearchHistory } from "./SearchHistory";
import { recordQuery } from "../../lib/history";
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
  const list = useRef<HTMLDivElement>(null);

  // The box follows the URL, so the back button visibly restores the query rather
  // than only the results.
  useEffect(() => setDraft(state.q), [state.q]);

  // `/` belongs to the shell now — bound once there so it works on all fourteen
  // routes rather than on this one alone. What stays here are the two
  // behaviours that only mean anything with a result list on screen: focus on
  // arrival, and Escape to get back out of the box.
  useEffect(() => {
    input.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && document.activeElement === input.current) {
        input.current?.blur();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /**
   * Arrows walk the results.
   *
   * ArrowDown from the query box steps into the list, which is the gesture that
   * makes the page keyboard-drivable: the box is where you already are after
   * typing, and reaching the first result should not cost six tabs through the
   * filter row.
   *
   * Result *links* only, not every focusable thing. A row also carries two
   * judgement buttons and a chunk toggle, and arrowing through five controls
   * per result is not navigation — it is the tab key with extra steps.
   */
  const step = useCallback((delta: number) => {
    const links = Array.from(
      list.current?.querySelectorAll<HTMLElement>("[data-result-link]") ?? [],
    );
    if (links.length === 0) return;
    const here = links.indexOf(document.activeElement as HTMLElement);
    const next = here < 0 ? (delta > 0 ? 0 : links.length - 1) : here + delta;
    links[Math.max(0, Math.min(links.length - 1, next))]?.focus();
  }, []);

  const onArrow = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      step(event.key === "ArrowDown" ? 1 : -1);
    },
    [step],
  );

  const update = useCallback(
    (next: Partial<SearchState>) => {
      setParams(toSearchParams({ ...state, ...next }), { replace: false });
    },
    [setParams, state],
  );

  const results = useQuery({
    queryKey: [
      "search",
      state.q,
      state.k,
      state.sources,
      state.kind,
      state.exact,
      state.tags,
    ],
    queryFn: () =>
      api.search({
        q: state.q,
        k: state.k,
        sources: state.sources,
        kind: state.kind ?? undefined,
        exact: state.exact,
        tags: state.tags,
      }),
    enabled: isRunnable(state),
    // A search is a question about a fixed corpus; re-asking it on every window
    // focus would burn a model forward pass to produce the same answer.
    staleTime: 60_000,
  });

  const filters = useMemo(
    () => ({
      k: state.k,
      sources: state.sources,
      kind: state.kind,
      exact: state.exact,
      tags: state.tags,
    }),
    [state],
  );

  return (
    <div className="flex flex-col gap-4">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const query = draft.trim();
          // Recorded on submit rather than on every keystroke or on every
          // `state.q` change: a history of what you typed on the way to a
          // query is not a history of your queries.
          recordQuery(query);
          update({ q: query });
        }}
        className="flex flex-col gap-1"
        role="search"
      >
        <label htmlFor="search-query" className="meta-label">
          search
        </label>
        <div className="flex items-baseline gap-3">
          <input
            id="search-query"
            ref={input}
            // The shell's `/` looks for this, so one key focuses the search box
            // on whichever page has one.
            data-search-input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            // ArrowDown out of the box and into the results.
            onKeyDown={onArrow}
            placeholder="search the corpus"
            aria-label="Search query"
            className="field-prominent"
            spellCheck={false}
            autoComplete="off"
          />
          <button type="submit" className="btn shrink-0" disabled={!draft.trim()}>
            search
          </button>
        </div>
        <span className="meta text-ink-3">
          <span className="kbd">/</span> focuses · <span className="kbd">↓</span> walks the
          results
        </span>
      </form>

      {/* Only when the box is empty, which is when it is the only thing that
          could help. Under a query in progress it would be a list of near
          misses competing with the results the reader is actually reading. */}
      {draft.trim().length === 0 ? (
        <SearchHistory
          onPick={(query) => {
            setDraft(query);
            update({ q: query });
          }}
        />
      ) : null}

      <Filters state={state} onChange={update} />

      {/* Timing, in small mono. Present always so it is noticed when it moves —
          the point of showing it is to see when something gets slow. */}
      {results.data ? (
        <div className="meta flex flex-wrap items-baseline justify-end gap-x-4 border-b border-rule pb-3 text-ink-3">
          <span data-testid="timing">
            embed {results.data.timing.embed_ms}ms · search {results.data.timing.search_ms}ms ·
            total {results.data.timing.total_ms}ms
          </span>
          <span>{count(results.data.hits.length)} memories</span>
          <span>
            {count(results.data.hits.reduce((sum, hit) => sum + hit.matched_chunks.length, 0))}{" "}
            chunks matched
          </span>
          {state.exact ? <span className="text-accent">exact scan</span> : null}
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
        <div ref={list} onKeyDown={onArrow} className="flex flex-col">
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
          <label className="meta-label text-ink-2" htmlFor="missing-key">
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
      <p className="meta mt-2 text-ink-3">
        k is {state.k}
        {state.k === DEFAULT_K ? " (default)" : ""} — a result below it is not missing, only
        unranked.
      </p>
    </section>
  );
}
