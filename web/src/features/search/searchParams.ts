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
  /**
   * Canonical tag names, without the `#`. Conjunctive server-side: two tags
   * narrow, because a second filter is somebody narrowing.
   *
   * Repeated in the URL rather than comma-joined, unlike `source`, and the
   * difference is not an inconsistency — the API reads `tag` as a repeated
   * parameter and `source` as one too, while this module has always comma-joined
   * sources and split them again on the way out. Tags follow the API's shape
   * because they are also produced by links written elsewhere in the interface
   * (`/search?tag=idea` from a chat message), and those cannot know this module's
   * convention.
   */
  tags: string[];
}

export const EMPTY: SearchState = {
  q: "",
  k: DEFAULT_K,
  sources: [],
  kind: null,
  exact: false,
  tags: [],
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
    // Sigil-tolerant and casefolded, so a hand-typed `?tag=#Idea` and a link
    // written as `?tag=idea` are the same filter.
    tags: [
      ...new Set(
        params
          .getAll("tag")
          .map((tag) => tag.trim().replace(/^#/, "").toLowerCase())
          .filter(Boolean),
      ),
    ],
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
  for (const tag of state.tags) params.append("tag", tag);
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
