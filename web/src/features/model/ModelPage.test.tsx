/**
 * Three properties, and all three are about the page not flattering itself.
 *
 * **An empty dimension is on screen with its reason.** This is the majority of
 * the page on the corpus that exists, and a view that skipped empty sections
 * would render four headings implying the model was complete.
 *
 * **A stated goal does not look like a computed finding.** Showing somebody
 * their own words back as a discovery is the cheapest way for a page like this
 * to lose their trust.
 *
 * **Contradicting evidence is visible.** A facet with support and counter-
 * evidence is not a strong claim, and a page showing only the support would
 * present it as one.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ModelPage } from "./ModelPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

const DERIVED = {
  id: "11111111-1111-7111-8111-111111111111",
  dimension: "weaknesses",
  statement:
    "A belief you keep returning to has failed more often than it has held: 'extraction will cover enough of the corpus' held 1 of 4 times it was checked.",
  confidence: 0.5,
  support_count: 3,
  contradiction_count: 1,
  origin: "derived",
  detector: "assumption_group_weak",
  superseded_by: null,
  dismissed_at: null,
  dismissed_reason: null,
  evidence: [
    { kind: "decision", ref_id: "22222222-2222-7222-8222-222222222222", relation: "supports" },
    { kind: "decision", ref_id: "33333333-3333-7333-8333-333333333333", relation: "contradicts" },
  ],
};

const STATED = {
  ...DERIVED,
  id: "44444444-4444-7444-8444-444444444444",
  dimension: "goals",
  statement: "Grow the corpus until these dimensions can be derived.",
  confidence: null,
  support_count: 0,
  contradiction_count: 0,
  origin: "asserted",
  detector: null,
  evidence: [],
};

const EMPTY_DIMENSIONS = [
  {
    dimension: "habits",
    facets: 0,
    gap: "every one of 253 dated memories carries a filesystem mtime, which records when a file was last written rather than when work happened",
    best_support: 0,
  },
  {
    dimension: "strengths",
    facets: 0,
    gap: "nothing reached 3 distinct observations",
    best_support: 2,
  },
  {
    dimension: "learning_style",
    facets: 0,
    gap: "no deriver exists: this needs the outcomes of learning attempts",
    best_support: 0,
  },
];

const MODEL = {
  facets: { weaknesses: [DERIVED], goals: [STATED] },
  assessments: [
    { dimension: "goals", facets: 1, gap: "", best_support: 0 },
    { dimension: "weaknesses", facets: 1, gap: "", best_support: 0 },
    ...EMPTY_DIMENSIONS,
  ],
  dismissed: [
    {
      ...DERIVED,
      id: "55555555-5555-7555-8555-555555555555",
      statement: "You avoid writing tests.",
      dismissed_at: "2026-08-14T10:00:00Z",
      dismissed_reason: "that is backwards",
    },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe("the model page", () => {
  it("renders every empty dimension as a gap with its cause", async () => {
    stubFetch([{ match: "/model", body: MODEL }]);
    renderWithProviders(<ModelPage />);

    // Present as headings, not omitted.
    expect(await screen.findByText("habits")).toBeInTheDocument();
    expect(screen.getByText("strengths")).toBeInTheDocument();
    expect(screen.getByText("learning style")).toBeInTheDocument();
    // And each says what would fill it, in words a person can act on.
    expect(screen.getByText(/filesystem mtime/)).toBeInTheDocument();
    expect(screen.getByText(/no deriver exists/)).toBeInTheDocument();
    expect(
      screen.getByText(/closest candidate reached 2 distinct observations/),
    ).toBeInTheDocument();
    expect(screen.getAllByText("insufficient evidence")).toHaveLength(3);
  });

  it("distinguishes a stated goal from a computed finding", async () => {
    stubFetch([{ match: "/model", body: MODEL }]);
    renderWithProviders(<ModelPage />);

    expect(await screen.findByText("stated")).toBeInTheDocument();
    expect(screen.getByText("assumption_group_weak")).toBeInTheDocument();
    // A stated facet carries no confidence, because a goal somebody wrote is not
    // a claim with a probability attached.
    expect(screen.getAllByText("confidence")).toHaveLength(1);
  });

  it("shows contradicting evidence beside the support", async () => {
    stubFetch([{ match: "/model", body: MODEL }]);
    renderWithProviders(<ModelPage />);

    // Two facets are on the page, so two "against" labels — the point is that
    // the label exists at all, beside the support rather than hidden behind it.
    expect(await screen.findAllByText("against")).toHaveLength(2);
    expect(screen.getByText(/contradicts/)).toBeInTheDocument();
    // And a rejected claim stays on screen rather than vanishing.
    expect(screen.getByText("You avoid writing tests.")).toBeInTheDocument();
    expect(screen.getByText("that is backwards")).toBeInTheDocument();
  });

  it("shows the history a facet has been through", async () => {
    const older = { ...DERIVED, id: "66666666-6666-7666-8666-666666666666", statement: "An earlier wording." };
    stubFetch([
      // `/history` first and matched on its own: every model URL contains
      // "/model", so a broader pattern here would answer the history request
      // with the whole model and the component would render an object as a list.
      { match: "/history", body: [older, DERIVED] },
      { match: "/model", body: MODEL },
    ]);
    renderWithProviders(<ModelPage />);

    // Dimensions render in assessment order — goals, then weaknesses — so the
    // second button belongs to the derived facet the stub returns a chain for.
    const buttons = await screen.findAllByRole("button", { name: "history" });
    await userEvent.click(buttons[1]);

    // The superseded wording is still readable — the whole reason the column
    // exists rather than an UPDATE.
    expect(await screen.findByText(/An earlier wording/)).toBeInTheDocument();
    expect(screen.getByText("current")).toBeInTheDocument();
  });
});
