/**
 * What the assumptions page must not round away.
 *
 * Three numbers, three meanings: held, failed, and never looked at. The third
 * is in neither half of any rate on this page, and a corpus where most
 * assumptions are unevaluated — which is every young corpus — would otherwise
 * report a hold rate over whatever happened to get attention.
 *
 * And `partially` is its own state on screen, not a rounding towards either
 * neighbour, because the cases it catches are the interesting ones.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";

import { AssumptionsPage } from "./AssumptionsPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

const STATS = {
  total: 37,
  evaluated: 25,
  unevaluated: 12,
  held: 18,
  failed: 6,
  partially: 1,
  hold_rate: 0.72,
  groups: [
    {
      id: "55555555-5555-7555-8555-555555555555",
      label: "Chunking stays deterministic, so an ordinal identifies the same span",
      strategy: "manual",
      members: 2,
      evaluated: 2,
      held: 2,
      failed: 0,
      partially: 0,
      hold_rate: 1.0,
      failure_rate: 0.0,
      statements: [
        "Chunking stays deterministic, so an ordinal identifies the same span after a rebuild.",
        "Chunking stays deterministic, so an ordinal identifies the same span after a replay.",
      ],
    },
  ],
};

const UNEVALUATED = [
  {
    id: "66666666-6666-7666-8666-666666666666",
    decision_id: "77777777-7777-7777-8777-777777777777",
    decision_question: "What runs background work?",
    statement: "The JobQueue port stays thin enough that swapping it is a day",
    confidence: 0.7,
    held: null,
    evaluated_at: null,
    note: null,
    group_id: null,
    group_label: null,
    outcome_verdict: "worked",
    evidence: [],
  },
];

afterEach(() => vi.unstubAllGlobals());

describe("the assumptions page", () => {
  it("reports unevaluated beside the rate rather than inside it", async () => {
    stubFetch([
      { match: "/assumptions/stats", body: STATS },
      { match: "/assumptions", body: UNEVALUATED },
    ]);
    renderWithProviders(<AssumptionsPage />);

    expect(await screen.findByText("12 unevaluated")).toBeInTheDocument();
    // 18 of 25, not 18 of 37. The twelve nobody has looked at are in neither
    // half — counting them as failures would punish writing assumptions down.
    expect(screen.getByText("72.0% of 25 evaluated")).toBeInTheDocument();
  });

  it("shows partially as its own count", async () => {
    stubFetch([
      { match: "/assumptions/stats", body: STATS },
      { match: "/assumptions", body: UNEVALUATED },
    ]);
    renderWithProviders(<AssumptionsPage />);

    expect(await screen.findByText("1 partially")).toBeInTheDocument();
    expect(screen.getByText("18 held")).toBeInTheDocument();
    expect(screen.getByText("6 failed")).toBeInTheDocument();
  });

  it("lists every member of a recurring group", async () => {
    stubFetch([
      { match: "/assumptions/stats", body: STATS },
      { match: "/assumptions", body: UNEVALUATED },
    ]);
    renderWithProviders(<AssumptionsPage />);

    expect(await screen.findByText(/2 members · 2 evaluated/)).toBeInTheDocument();
    // Both statements, because the point of a group is that two different
    // sentences were the same belief — a label alone hides that.
    expect(screen.getByText(/after a rebuild\./)).toBeInTheDocument();
    expect(screen.getByText(/after a replay\./)).toBeInTheDocument();
  });

  it("says so plainly when nothing recurs", async () => {
    stubFetch([
      { match: "/assumptions/stats", body: { ...STATS, groups: [] } },
      { match: "/assumptions", body: UNEVALUATED },
    ]);
    renderWithProviders(<AssumptionsPage />);

    // The honest empty state: no recurrence means no pattern is available,
    // which is a finding rather than a missing feature.
    expect(await screen.findByText(/nothing recurs yet/)).toBeInTheDocument();
    expect(screen.getByText(/held once/)).toBeInTheDocument();
  });
});
