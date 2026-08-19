/**
 * Source, kind and k — in one dense row, not a sidebar of cards.
 *
 * Sources are toggles rather than a `<select multiple>`: with a handful of
 * connectors, toggles show what is active without being opened, and a native
 * multi-select is one of the worst controls on the web.
 */

import { useQuery } from "@tanstack/react-query";

import { api } from "../../api/client";
import { count } from "../../lib/format";
import { DEFAULT_K, KINDS, type SearchState } from "./searchParams";

interface Props {
  state: SearchState;
  onChange: (next: Partial<SearchState>) => void;
}

export function Filters({ state, onChange }: Props) {
  const sources = useQuery({
    queryKey: ["sources"],
    queryFn: api.sources,
    // The set of connectors changes when somebody registers one, which is not
    // something that happens while a search page is open.
    staleTime: 5 * 60_000,
  });

  const toggleSource = (name: string) => {
    const active = state.sources.includes(name);
    onChange({
      sources: active
        ? state.sources.filter((other) => other !== name)
        : [...state.sources, name],
    });
  };

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-rule py-2">
      <div className="flex items-center gap-1.5">
        <span className="meta-label">source</span>
        {sources.isLoading ? <span className="meta text-ink-3">…</span> : null}
        {sources.isError ? <span className="meta text-deny">unavailable</span> : null}
        {(sources.data ?? []).map((source) => (
          <button
            key={source.id}
            type="button"
            className={`btn ${state.sources.includes(source.name) ? "btn-on" : ""}`}
            aria-pressed={state.sources.includes(source.name)}
            onClick={() => toggleSource(source.name)}
            // The count is the useful part: it says whether filtering to this
            // source will return anything at all.
            title={`${count(source.memories)} memories, ${count(source.chunks)} chunks`}
          >
            {source.name}
            <span className="ml-1 text-ink-3">{count(source.memories)}</span>
          </button>
        ))}
        {state.sources.length > 0 ? (
          <button type="button" className="btn" onClick={() => onChange({ sources: [] })}>
            clear
          </button>
        ) : null}
      </div>

      <label className="flex items-center gap-1.5">
        <span className="meta-label">kind</span>
        <select
          className="field meta"
          value={state.kind ?? ""}
          onChange={(event) => onChange({ kind: event.target.value || null })}
          aria-label="Kind"
        >
          <option value="">any</option>
          {KINDS.map((kind) => (
            <option key={kind} value={kind}>
              {kind}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-1.5">
        <span className="meta-label">k</span>
        <input
          type="number"
          min={1}
          max={100}
          value={state.k}
          onChange={(event) => onChange({ k: Number(event.target.value) || DEFAULT_K })}
          className="field w-16 font-mono text-xs"
          aria-label="Number of results"
        />
      </label>

      <label className="flex items-center gap-1.5" title="Sequential scan instead of the index">
        <input
          type="checkbox"
          checked={state.exact}
          onChange={(event) => onChange({ exact: event.target.checked })}
          aria-label="Exact scan"
        />
        <span className="meta-label">exact</span>
      </label>
    </div>
  );
}
