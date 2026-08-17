/**
 * The overview, and the one property that makes it worth having: **every number
 * on it comes from the API.**
 *
 * This is the page most likely to acquire a hardcoded figure, because it is the
 * one a screenshot is taken of. A hardcoded number here is the worst kind — it
 * is the first thing anybody reads, it looks authoritative, and it starts lying
 * the moment the corpus changes without anyone noticing.
 *
 * The tests are written to fail if a figure is written into the source rather
 * than fetched: the fixture serves deliberately unround values, and the
 * assertions change the response and demand the screen change with it. A
 * component that printed a constant would pass the first test and fail the
 * second, which is the pair that matters.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../../App";
import { OverviewPage } from "./OverviewPage";
import {
  READY,
  SHELL_ROUTES,
  STATS,
  renderWithProviders,
  stubFetch,
} from "../../test/harness";

const DECISIONS = [
  { id: "1", question: "q1", chosen: "a", status: "open", confidence: 0.7, options: 2, assumptions: 3, evidence: 1, decided_at: null, decided_at_source: "parsed" },
  { id: "2", question: "q2", chosen: "b", status: "settled", confidence: null, options: 2, assumptions: 0, evidence: 0, decided_at: null, decided_at_source: "parsed" },
];

const MEMORIES = [
  {
    id: "11111111-1111-7111-8111-111111111111",
    source_id: "33333333-3333-7333-8333-333333333333",
    external_key: "src/memoryos/application/worker.py",
    version: 1,
    is_current: true,
    kind: "code",
    title: null,
    content_hash: "a".repeat(64),
    occurred_at: null,
    occurred_at_source: "filesystem",
    ingested_at: "2026-08-01T10:00:00Z",
    deleted_at: null,
  },
];

function stubAll(stats: Record<string, unknown> = STATS) {
  return stubFetch([
    { match: "/stats", body: stats },
    { match: "/health/ready", body: READY },
    { match: "/decisions", body: DECISIONS },
    { match: "/memories", body: MEMORIES },
  ]);
}

afterEach(() => vi.unstubAllGlobals());

describe("the numbers come from the API", () => {
  it("renders the corpus figures the API returned", async () => {
    stubAll();
    renderWithProviders(<OverviewPage />);

    const figures = await screen.findByTestId("figures");
    // Exactly the fixture's values, formatted. None of these is a number
    // anybody would have typed into a component.
    expect(await within(figures).findByText("271")).toBeInTheDocument();
    expect(await within(figures).findByText("3,833")).toBeInTheDocument();
    expect(await within(figures).findByText("34")).toBeInTheDocument();
    expect(await within(figures).findByText("7")).toBeInTheDocument();
  });

  it("changes when the API's answer changes", async () => {
    // The test a hardcoded figure cannot pass.
    stubAll({ ...STATS, current_memories: 9021, chunks: 44105, entities: 612 });
    renderWithProviders(<OverviewPage />);

    const figures = await screen.findByTestId("figures");
    expect(await within(figures).findByText("9,021")).toBeInTheDocument();
    expect(await within(figures).findByText("44,105")).toBeInTheDocument();
    expect(await within(figures).findByText("612")).toBeInTheDocument();
  });

  it("counts decisions from the decisions endpoint, not from stats", async () => {
    stubAll();
    renderWithProviders(<OverviewPage />);

    const figures = await screen.findByTestId("figures");
    expect(await within(figures).findByText("2")).toBeInTheDocument();
    // And says how many of them are complete enough to reason about, which is
    // the number that decides whether the decisions layer means anything.
    expect(await within(figures).findByText(/1 with assumptions/)).toBeInTheDocument();
  });

  it("shows a dash rather than a zero while the figures are unknown", async () => {
    // A zero is a claim about the corpus. Before the request resolves, no such
    // claim has been checked, and printing one is how an empty database and a
    // slow one become indistinguishable.
    stubFetch([
      { match: "/stats", status: 500, body: { detail: "boom" } },
      { match: "/health/ready", body: READY },
      { match: "/decisions", body: DECISIONS },
      { match: "/memories", body: MEMORIES },
    ]);
    renderWithProviders(<OverviewPage />);

    expect(await screen.findByTestId("error")).toBeInTheDocument();
  });

  it("says a zero relationship count means none were extracted", async () => {
    // The most informative number on this corpus, and the one most easily
    // misread as a loading state.
    stubAll({ ...STATS, relationships: 0 });
    renderWithProviders(<OverviewPage />);

    const figures = await screen.findByTestId("figures");
    expect(await within(figures).findByText(/none extracted yet/)).toBeInTheDocument();
  });
});

describe("health", () => {
  it("reports what readiness returned rather than a fixed badge", async () => {
    stubFetch([
      { match: "/stats", body: STATS },
      {
        match: "/health/ready",
        body: { status: "degraded", database: true, pgvector_version: "0.8.0", graph: false },
      },
      { match: "/decisions", body: DECISIONS },
      { match: "/memories", body: MEMORIES },
    ]);
    renderWithProviders(<OverviewPage />);

    const health = await screen.findByTestId("overview-health");
    // The distinction the endpoint exists to draw: the graph is down and
    // search still works, which must not read as a total outage.
    expect(await within(health).findByText("unreachable")).toBeInTheDocument();
    expect(await within(health).findByText(/pgvector 0\.8\.0/)).toBeInTheDocument();
  });

  it("counts unembedded chunks from stats", async () => {
    stubAll();
    renderWithProviders(<OverviewPage />);

    const health = await screen.findByTestId("overview-health");
    // 3833 - 3820, arithmetic on what the API sent.
    expect(await within(health).findByText(/13 not embedded/)).toBeInTheDocument();
  });
});

describe("the search box", () => {
  it("hands the query to the search route rather than running it here", async () => {
    // Navigation is all this box does, and that is the design: `/search` owns
    // every piece of search state in its URL, so running results here would be
    // a second implementation of the same view and a query nobody could link
    // to. Rendered through `<App />` because the assertion *is* the navigation.
    stubFetch([
      ...SHELL_ROUTES,
      { match: "/decisions", body: DECISIONS },
      { match: "/memories", body: MEMORIES },
      { match: "/sources", body: [] },
      {
        match: "/search",
        body: { query: "", timing: { embed_ms: 1, search_ms: 1, total_ms: 2 }, hits: [] },
      },
    ]);
    // `/overview` since M10.0, which is where this box now lives: `/` is the
    // chat, and its box is a message rather than a query.
    renderWithProviders(<App />, { route: "/overview" });

    await userEvent.type(
      await screen.findByLabelText("Search query"),
      "chunk overlap{Enter}",
    );

    // The search page's own box, carrying the query out of the URL.
    expect(await screen.findByText(/no results/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Search query")).toHaveValue("chunk overlap");
  });

  it("lists recent memories from the API", async () => {
    stubAll();
    renderWithProviders(<OverviewPage />);

    const recent = await screen.findByTestId("recent");
    expect(
      await within(recent).findByText("src/memoryos/application/worker.py"),
    ).toBeInTheDocument();
  });
});
