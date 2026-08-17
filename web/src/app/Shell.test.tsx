/**
 * The shell: that navigation lands somewhere, and that the keyboard model is
 * bound where it claims to be.
 *
 * The route assertions are the ones worth having. Phases 3 to 8 each shipped a
 * view and a masthead that named some of them, and the result was six working
 * pages nothing linked to — a failure no component test can see, because every
 * one of those pages rendered perfectly in isolation. What was missing was a
 * test that clicked the nav.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../test/harness";

/** Everything any of these routes might ask for, so no view 404s the harness. */
function stubEverything() {
  return stubFetch([
    ...SHELL_ROUTES,
    // M10.0: `/` is the chat, so every route test loads a transcript on the way
    // in. Listed before `/sources` for no reason other than reading order.
    { match: "/chat", body: [] },
    { match: "/sources", body: [] },
    { match: "/memories", body: [] },
    { match: "/decisions", body: [] },
    { match: "/outcomes", body: { worked: 0, failed: 0, mixed: 0, too_early: 0, undecided: 0, resolved: 0, rate: null } },
    { match: "/timeline", body: { start: "2026-01-01T00:00:00Z", end: "2026-08-01T00:00:00Z", buckets: [], provenance: [], total: 0 } },
    { match: "/gaps", body: [] },
    { match: "/judgements", body: [] },
    { match: "/surfacing", body: [] },
    { match: "/model", body: { facets: {}, assessments: [], dismissed: [] } },
    { match: "/search", body: { query: "", timing: { embed_ms: 0, search_ms: 0, total_ms: 0 }, hits: [] } },
  ]);
}

afterEach(() => vi.unstubAllGlobals());

describe("navigation", () => {
  it("renders the chat at the root, not the overview and not search", async () => {
    // The M10.0 swap, pinned. `/` is the front door: the thing you land on is
    // the box that both keeps what you type and answers what you ask.
    stubEverything();
    renderWithProviders(<App />, { route: "/" });

    expect(
      await screen.findByRole("heading", { name: /say it here/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Message")).toBeInTheDocument();
  });

  it("keeps the overview, one route along", async () => {
    stubEverything();
    renderWithProviders(<App />, { route: "/" });

    const nav = screen.getByRole("navigation");
    await userEvent.click(within(nav).getByRole("link", { name: "overview" }));

    expect(
      await screen.findByRole("heading", { name: /a corpus that remembers why/i }),
    ).toBeInTheDocument();
  });

  // One distinctive thing per destination — text that appears on that view and
  // on no other, so a passing row means the router landed there rather than
  // merely re-rendering the shell.
  // One distinctive heading per destination. Matched as a heading rather than
  // as loose text because several of these words also appear in the sidebar or
  // in a page's own body copy, and a text match would pass without the route
  // having changed at all.
  it.each([
    ["timeline", /date provenance/i],
    ["judgements", /golden set/i],
    ["surfacing", /^surfacing$/i],
    ["model", /user model/i],
    ["corpus", /health checks/i],
    ["sources", /where else this reads from/i],
  ])("navigating to %s renders that route", async (label, heading) => {
    stubEverything();
    renderWithProviders(<App />, { route: "/" });

    const nav = screen.getByRole("navigation");
    await userEvent.click(within(nav).getByRole("link", { name: label }));

    expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
  });

  it("navigating to search renders the search view", async () => {
    // Asserted on the control rather than on prose: "search" is a word the
    // sidebar also says, and a text match would pass without the route
    // changing at all.
    stubEverything();
    renderWithProviders(<App />, { route: "/" });

    const nav = screen.getByRole("navigation");
    await userEvent.click(within(nav).getByRole("link", { name: "search" }));

    expect(await screen.findByText(/nothing searched yet/i)).toBeInTheDocument();
  });

  it("sends an unbuilt route to a page that says so, rather than a blank one", async () => {
    stubEverything();
    renderWithProviders(<App />, { route: "/graph" });

    expect(await screen.findByText(/what is missing/i)).toBeInTheDocument();
    // The distinguishing property: it explains rather than pretending.
    expect(screen.getByText(/what is already there/i)).toBeInTheDocument();
  });

  it("forwards a search bookmarked at the old root path", async () => {
    // `/?q=…` was a real, linkable URL for eight milestones. Losing those to a
    // 404 would undo the entire argument for keeping search state in the URL.
    stubEverything();
    renderWithProviders(<App />, { route: "/?q=leases" });

    expect(await screen.findByLabelText("Search query")).toHaveValue("leases");
  });

  it("keeps the old agent path working", async () => {
    stubEverything();
    renderWithProviders(<App />, { route: "/ask" });

    expect(await screen.findByLabelText(/ask/i)).toBeInTheDocument();
  });
});

describe("keyboard", () => {
  it("focuses the search box when / is pressed, from any route", async () => {
    stubEverything();
    renderWithProviders(<App />, { route: "/search" });

    const input = await screen.findByLabelText("Search query");
    input.blur();
    expect(input).not.toHaveFocus();

    await userEvent.keyboard("/");
    expect(input).toHaveFocus();
  });

  it("does not steal / while something is being typed into", async () => {
    stubEverything();
    renderWithProviders(<App />, { route: "/search" });

    const input = await screen.findByLabelText("Search query");
    await userEvent.type(input, "src/memoryos");
    // A search box that swallowed `/` would be unusable for path queries.
    expect(input).toHaveValue("src/memoryos");
  });
});
