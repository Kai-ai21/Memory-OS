/**
 * The empty states, which on this corpus are the whole page.
 *
 * The thing worth protecting is that an absence renders as a *stated* absence.
 * A dimension with nothing above the evidence bar has to say so and say why —
 * a blank row reads as a view that failed to load, and an omitted row reads as
 * a model that is complete. Both are worse than the gap sentence the API
 * already sends.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";

import { InsightsPage } from "./InsightsPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

/** Shaped like the live response: every dimension empty, each for its own reason. */
const MODEL = {
  facets: {},
  dismissed: [],
  assessments: [
    {
      dimension: "goals",
      facets: 0,
      gap: "goals are stated, never inferred — use `model assert --dimension goals`",
      best_support: 0,
    },
    {
      dimension: "habits",
      facets: 0,
      gap: "nothing reached 3 distinct observations",
      best_support: 1,
    },
    {
      dimension: "workflows",
      facets: 0,
      gap: "entity extraction has reached 8 of 321 memories (2%)",
      best_support: 0,
    },
    // The one populated row, so the empty treatment is asserted against a
    // contrast rather than against a page where everything looks the same.
    { dimension: "decision_patterns", facets: 2, gap: "", best_support: 5 },
  ],
};

function routes(overrides: { decisions?: unknown[] } = {}) {
  return [
    { match: "/assumptions/stats", body: { total: 37, evaluated: 25, unevaluated: 12, held: 18, failed: 6, partially: 1, hold_rate: 0.72, groups: [] } },
    { match: "/patterns", body: [] },
    { match: "/reflections", body: [] },
    { match: "/model", body: MODEL },
    { match: "/decisions", body: overrides.decisions ?? new Array(12).fill({}) },
  ];
}

afterEach(() => vi.unstubAllGlobals());

describe("an empty dimension", () => {
  it("renders its gap message rather than blank space", async () => {
    stubFetch(routes());
    renderWithProviders(<InsightsPage />, { route: "/insights" });

    const rows = await screen.findAllByTestId("dimension");
    const goals = rows.find((row) => row.dataset.dimension === "goals")!;

    // Marked as empty, and carrying the API's own explanation of *why* — which
    // differs per dimension and is the reason this page is worth reading.
    expect(within(goals).getByTestId("insufficient")).toBeInTheDocument();
    expect(within(goals).getByTestId("gap")).toHaveTextContent(
      /goals are stated, never inferred/i,
    );
  });

  it("gives each dimension its own cause rather than one label repeated", async () => {
    // The reference reads INSUFFICIENT EVIDENCE on every empty row, which says
    // the same thing six times. The real causes are different problems with
    // different answers, and flattening them is what this asserts against.
    stubFetch(routes());
    renderWithProviders(<InsightsPage />, { route: "/insights" });

    const rows = await screen.findAllByTestId("dimension");
    const gaps = rows
      .filter((row) => row.dataset.empty === "true")
      .map((row) => within(row).getByTestId("gap").textContent);

    expect(gaps).toHaveLength(3);
    expect(new Set(gaps).size).toBe(3);
  });

  it("names how close the closest candidate got, when it got anywhere", async () => {
    stubFetch(routes());
    renderWithProviders(<InsightsPage />, { route: "/insights" });

    const rows = await screen.findAllByTestId("dimension");
    const habits = rows.find((row) => row.dataset.dimension === "habits")!;

    expect(within(habits).getByTestId("gap")).toHaveTextContent(
      /closest candidate reached 1 distinct observation/i,
    );
  });

  it("does not mark a populated dimension as insufficient", async () => {
    stubFetch(routes());
    renderWithProviders(<InsightsPage />, { route: "/insights" });

    const rows = await screen.findAllByTestId("dimension");
    const derived = rows.find((row) => row.dataset.dimension === "decision_patterns")!;

    expect(within(derived).queryByTestId("insufficient")).not.toBeInTheDocument();
    expect(derived).toHaveTextContent(/2 facets/);
  });
});

describe("the patterns empty state", () => {
  it("counts the real decisions rather than asserting a figure", async () => {
    // The number in this sentence is the point of it: "a pattern needs 3" means
    // nothing without "and you have 12". A hardcoded 12 would go stale on the
    // thirteenth decision and quietly become a lie.
    stubFetch(routes({ decisions: new Array(7).fill({}) }));
    renderWithProviders(<InsightsPage />, { route: "/insights" });

    const empty = await screen.findAllByTestId("empty");
    expect(empty[0]).toHaveTextContent(/7 decisions recorded/i);
    expect(empty[0]).toHaveTextContent(/at least 3 supporting decisions/i);
  });
});
