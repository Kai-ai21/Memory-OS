/**
 * The search view: states, keyboard, and what actually gets requested.
 *
 * Kept proportionate — this is a local tool, not a product — so the tests cover
 * the things that would be quietly wrong rather than every branch: that the
 * loading, empty and error states appear at all, that the keyboard model works,
 * that filters reach the API in the query string, and that the highlight renders
 * only when it can be positioned correctly.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SearchPage } from "./SearchPage";
import {
  MATCHED_TEXT,
  borrowingHit,
  renderWithProviders,
  searchResponse,
  stubFetch,
} from "../../test/harness";

const SOURCES = [
  {
    id: "33333333-3333-7333-8333-333333333333",
    kind: "filesystem",
    name: "self",
    config: {},
    last_sync_at: null,
    last_full_sync_at: null,
    created_at: "2026-08-01T10:00:00Z",
    memories: 111,
    chunks: 854,
  },
];

afterEach(() => vi.unstubAllGlobals());

describe("states", () => {
  it("says nothing has been searched, with what to do next", () => {
    stubFetch([{ match: "/sources", body: SOURCES }]);
    renderWithProviders(<SearchPage />);

    expect(screen.getByTestId("empty")).toHaveTextContent(/nothing searched yet/i);
    expect(screen.getByTestId("empty")).toHaveTextContent(/press Enter/i);
  });

  it("shows a loading state while the search is in flight", async () => {
    // A search embeds the query on a CPU-bound thread, so this state is real
    // rather than theoretical.
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=leases" });

    expect(screen.getByTestId("loading")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("result")).toBeInTheDocument());
  });

  it("suggests broadening when there are no results", async () => {
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse({ hits: [] }) },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=nothing+matches+this&k=5" });

    const empty = await screen.findByTestId("empty");
    expect(empty).toHaveTextContent(/no results/i);
    // Actionable, not just a dead end.
    expect(empty).toHaveTextContent(/raising/i);
    expect(empty).toHaveTextContent(/5/);
  });

  it("mentions the filters when they could be the reason", async () => {
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse({ hits: [] }) },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=x&kind=code&source=self" });

    expect(await screen.findByTestId("empty")).toHaveTextContent(/clearing the source and kind/i);
  });

  it("names the failure when the API returns an error", async () => {
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", status: 404, body: { detail: "no source named 'ghost'" } },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=x&source=ghost" });

    const error = await screen.findByTestId("error");
    expect(error).toHaveTextContent(/no source named 'ghost'/);
  });

  it("says the API is unreachable rather than showing a generic failure", async () => {
    // The most common failure when running this locally, and the one where a
    // useful message saves the most time.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    renderWithProviders(<SearchPage />, { route: "/?q=x" });

    const error = await screen.findByTestId("error");
    expect(error).toHaveTextContent(/cannot reach the api/i);
    expect(error).toHaveTextContent(/make dev/);
  });
});

describe("results", () => {
  it("renders score, key, timing and the definition metadata", async () => {
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=leases" });

    expect(await screen.findByTestId("score")).toHaveTextContent("0.8147");
    expect(screen.getByText("src/memoryos/application/worker.py")).toBeInTheDocument();
    expect(screen.getByTestId("timing")).toHaveTextContent("embed 12ms");
    expect(screen.getByTestId("timing")).toHaveTextContent("total 15ms");
    // What M1.7 persisted chunk metadata for.
    expect(screen.getByTestId("definition")).toHaveTextContent("Worker.run");
  });

  it("highlights immediately, without waiting for the memory", async () => {
    // The offsets are enough on their own: the chunk's own text is the tail of
    // what is stored. No fetch, so a result is legible the moment it arrives.
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=leases" });

    expect(await screen.findByTestId("mark")).toHaveTextContent(MATCHED_TEXT);
  });

  it("mutes the borrowed lead-in so it does not read as the match", async () => {
    // 28.1% of stored chunk text in this corpus is borrowed from the previous
    // chunk. Rendering it as part of the match is the misattribution these
    // offsets invite.
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse({ hits: [borrowingHit()] }) },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=leases" });

    expect(await screen.findByTestId("lead-in")).toHaveTextContent("borrowed context");
    expect(screen.getByTestId("mark")).toHaveTextContent(MATCHED_TEXT);
  });

  it("shows the neighbouring chunks when a chunk is expanded", async () => {
    // What makes "chunk 3 matched" interpretable. The fetch is for the
    // neighbours, not for the highlight.
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
      {
        match: "/memories/",
        body: {
          id: "11111111-1111-7111-8111-111111111111",
          source_id: "33333333-3333-7333-8333-333333333333",
          source_name: "self",
          external_key: "src/memoryos/application/worker.py",
          version: 1,
          is_current: true,
          kind: "code",
          title: null,
          content: "irrelevant to the highlight",
          content_hash: "a".repeat(64),
          normalized_hash: "b".repeat(64),
          occurred_at: null,
          occurred_at_source: "unknown",
          ingested_at: "2026-08-01T10:00:00Z",
          deleted_at: null,
          metadata: {},
          chunks: [
            {
              id: "55555555-5555-7555-8555-555555555555",
              ordinal: 4,
              content: "the chunk that comes next",
              token_count: 7,
              char_start: 200,
              char_end: 225,
              chunker_version: "structural-v2",
              content_hash: "c".repeat(64),
              embedding_model: "m@1",
              embedded_at: null,
              metadata: {},
              embedded: true,
            },
          ],
          versions: [],
        },
      },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=leases" });

    await userEvent.click(await screen.findByRole("button", { name: /#3/ }));

    expect(await screen.findByText(/the chunk that comes next/)).toBeInTheDocument();
  });
});

describe("filters and the URL", () => {
  it("sends every filter to the API in the query string", async () => {
    const calls = stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
    ]);
    renderWithProviders(<SearchPage />, {
      route: "/?q=leases&k=25&kind=code&source=self&exact=1",
    });

    await screen.findByTestId("result");
    const search = calls.find((call) => call.url.includes("/search"));
    expect(search?.url).toContain("q=leases");
    expect(search?.url).toContain("k=25");
    expect(search?.url).toContain("kind=code");
    expect(search?.url).toContain("source=self");
    expect(search?.url).toContain("exact=true");
  });

  it("repeats the source parameter for a multi-select", async () => {
    const calls = stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=x&source=self,notes" });

    await screen.findByTestId("result");
    const search = calls.find((call) => call.url.includes("/search"));
    expect(search?.url).toContain("source=self&source=notes");
  });

  it("restores the query into the box from the URL, so a reload lands where you were", () => {
    stubFetch([
      { match: "/sources", body: SOURCES },
      { match: "/search", body: searchResponse() },
    ]);
    renderWithProviders(<SearchPage />, { route: "/?q=two+timestamps" });

    expect(screen.getByLabelText("Search query")).toHaveValue("two timestamps");
  });
});

describe("keyboard", () => {
  it("focuses the query box on load", () => {
    stubFetch([{ match: "/sources", body: SOURCES }]);
    renderWithProviders(<SearchPage />);
    expect(screen.getByLabelText("Search query")).toHaveFocus();
  });

  it("focuses the query box when / is pressed from elsewhere", async () => {
    stubFetch([{ match: "/sources", body: SOURCES }]);
    renderWithProviders(<SearchPage />);

    const input = screen.getByLabelText("Search query");
    input.blur();
    expect(input).not.toHaveFocus();

    await userEvent.keyboard("/");
    expect(input).toHaveFocus();
  });

  it("does not steal / while typing, because it is a character", async () => {
    stubFetch([{ match: "/sources", body: SOURCES }]);
    renderWithProviders(<SearchPage />);

    const input = screen.getByLabelText("Search query");
    await userEvent.type(input, "src/memoryos");
    // A search box that swallowed `/` would be unusable for path queries.
    expect(input).toHaveValue("src/memoryos");
  });

  it("blurs the query box on Escape", async () => {
    stubFetch([{ match: "/sources", body: SOURCES }]);
    renderWithProviders(<SearchPage />);

    const input = screen.getByLabelText("Search query");
    expect(input).toHaveFocus();
    await userEvent.keyboard("{Escape}");
    expect(input).not.toHaveFocus();
  });
});
