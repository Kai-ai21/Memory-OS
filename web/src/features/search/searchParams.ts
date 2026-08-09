/**
 * Search state lives in the URL, not in React.
 *
 * Every filter is a query parameter, which buys three things a `useState` would
 * not: a search is linkable, the back button steps through the searches you
 * actually ran, and a reload lands where you were. For a tool whose whole job is
 * comparing one query's results against another's, that is not a nicety.
 *
 * Parsing is total — anything unrecognised falls back to a default rather than
 * throwing — because the URL is user-editable and a hand-typed `k=banana` should
 * degrade, not blank the page.
 */

export const DEFAULT_K = 10;
const MAX_K = 100;

/** The kinds the backend's `MemoryKind` enum allows, for the filter control. */
export const KINDS = [
  "note",
  "document",
  "code",
  "email",
  "commit",
  "meeting",
  "bookmark",
  "other",
] as const;

export interface SearchState {
  q: string;
  k: number;
  /** Source names. Multi-select, comma-joined in the URL. */
  sources: string[];
  kind: string | null;
  exact: boolean;
}

export const EMPTY: SearchState = {
  q: "",
  k: DEFAULT_K,
  sources: [],
  kind: null,
  exact: false,
};

export function parseSearchState(params: URLSearchParams): SearchState {
  return {
    q: params.get("q") ?? "",
    k: parseK(params.get("k")),
    sources: (params.get("source") ?? "")
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
    kind: KINDS.includes((params.get("kind") ?? "") as (typeof KINDS)[number])
      ? params.get("kind")
      : null,
    exact: params.get("exact") === "1",
  };
}

/**
 * Back to a query string, omitting anything at its default.
 *
 * Omitting defaults keeps the URL short enough to read, which matters when the
 * URL is the thing you paste into a note to record what you searched.
 */
export function toSearchParams(state: SearchState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.q) params.set("q", state.q);
  if (state.k !== DEFAULT_K) params.set("k", String(state.k));
  if (state.sources.length) params.set("source", state.sources.join(","));
  if (state.kind) params.set("kind", state.kind);
  if (state.exact) params.set("exact", "1");
  return params;
}

function parseK(raw: string | null): number {
  const parsed = Number.parseInt(raw ?? "", 10);
  if (!Number.isFinite(parsed) || parsed < 1) return DEFAULT_K;
  return Math.min(parsed, MAX_K);
}

/** Whether this state is worth sending. An empty query is not a search. */
export function isRunnable(state: SearchState): boolean {
  return state.q.trim().length > 0;
}
