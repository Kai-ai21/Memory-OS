/**
 * URL state, round-tripped.
 *
 * Filters live in the query string so a search is linkable and the back button
 * works. That only holds if parse and serialise are exact inverses, and if a
 * hand-edited URL degrades instead of blanking the page.
 */

import { describe, expect, it } from "vitest";

import {
  DEFAULT_K,
  EMPTY,
  isRunnable,
  parseSearchState,
  toSearchParams,
  type SearchState,
} from "./searchParams";

function roundTrip(state: SearchState): SearchState {
  return parseSearchState(new URLSearchParams(toSearchParams(state).toString()));
}

describe("search params", () => {
  it("survives a round trip with everything set", () => {
    const state: SearchState = {
      q: "how a lease is renewed",
      k: 25,
      sources: ["self", "notes"],
      kind: "code",
      exact: true,
      tags: ["idea", "postgres"],
    };
    expect(roundTrip(state)).toEqual(state);
  });

  it("survives a round trip with only a query", () => {
    const state: SearchState = { ...EMPTY, q: "two timestamps" };
    expect(roundTrip(state)).toEqual(state);
  });

  it("omits defaults from the URL so it stays readable", () => {
    const params = toSearchParams({ ...EMPTY, q: "x" });
    expect(params.toString()).toBe("q=x");
  });

  it("keeps k in the URL only when it differs from the default", () => {
    expect(toSearchParams({ ...EMPTY, q: "x", k: DEFAULT_K }).get("k")).toBeNull();
    expect(toSearchParams({ ...EMPTY, q: "x", k: 50 }).get("k")).toBe("50");
  });

  it("preserves a query containing characters that need encoding", () => {
    const state: SearchState = { ...EMPTY, q: "why two clocks are recorded & k=5?" };
    expect(roundTrip(state)).toEqual(state);
  });

  it("preserves multiple sources", () => {
    expect(parseSearchState(new URLSearchParams("source=a,b,c")).sources).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  describe("degrading on hand-edited URLs", () => {
    it("falls back to the default k for nonsense", () => {
      expect(parseSearchState(new URLSearchParams("k=banana")).k).toBe(DEFAULT_K);
      expect(parseSearchState(new URLSearchParams("k=0")).k).toBe(DEFAULT_K);
      expect(parseSearchState(new URLSearchParams("k=-4")).k).toBe(DEFAULT_K);
    });

    it("caps k rather than asking the API for ten thousand results", () => {
      expect(parseSearchState(new URLSearchParams("k=99999")).k).toBe(100);
    });

    it("ignores a kind that is not in the enum", () => {
      // Sending it would be a 422 from the API for something the user cannot fix.
      expect(parseSearchState(new URLSearchParams("kind=sandwich")).kind).toBeNull();
    });

    it("drops empty source names", () => {
      expect(parseSearchState(new URLSearchParams("source=,,a,")).sources).toEqual(["a"]);
    });

    it("treats a missing query as no search rather than an empty one", () => {
      const state = parseSearchState(new URLSearchParams(""));
      expect(state.q).toBe("");
      expect(isRunnable(state)).toBe(false);
    });

    it("treats a whitespace-only query as no search", () => {
      expect(isRunnable({ ...EMPTY, q: "   " })).toBe(false);
    });
  });
});
