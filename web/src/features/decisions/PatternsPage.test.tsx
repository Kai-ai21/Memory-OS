/**
 * The property this page exists to hold: counter-evidence is not hidden.
 *
 * A patterns view that showed five agreeing decisions prominently and three
 * disagreeing ones behind a toggle would be worse than no view at all — the
 * reader comes away believing a claim the corpus half contradicts. So the test
 * asserts both lists render, both are labelled with their counts, and the
 * contradicting decisions are as clickable as the supporting ones.
 *
 * And the empty state, which on any young corpus is the one that shows: "no
 * patterns" must read as a result rather than as a broken detector, and the
 * calibration table has to be there to say what the data actually supports.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { PatternsPage } from "./PatternsPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

const CALIBRATION = {
  decisions: [
    {
      low: 0.75,
      high: 1.0,
      stated: 0.892,
      observed: 1.0,
      interval_low: 0.61,
      interval_high: 1.0,
      n: 6,
      miscalibrated: false,
    },
  ],
  assumptions: [
    {
      low: 0.75,
      high: 1.0,
      stated: 0.846,
      observed: 1.0,
      interval_low: 0.785,
      interval_high: 1.0,
      n: 14,
      miscalibrated: false,
    },
  ],
};

const PATTERN = {
  id: "11111111-1111-7111-8111-111111111111",
  statement: "A recurring assumption breaks more often than it holds.",
  kind: "assumption",
  detector: "assumption_group",
  support_count: 4,
  contradiction_count: 2,
  confidence: 0.53,
  first_observed: "2026-01-01T12:00:00Z",
  last_observed: "2026-03-01T12:00:00Z",
  span_days: 59,
  discovered_at: "2026-08-13T12:00:00Z",
  dismissed_at: null,
  dismissed_reason: null,
  supporting: [
    {
      decision_id: "22222222-2222-7222-8222-222222222222",
      decision_question: "Which deploy path?",
      decided_at: "2026-01-01T12:00:00Z",
      relation: "supports",
      note: "failed: the deploy is straightforward",
    },
  ],
  contradicting: [
    {
      decision_id: "33333333-3333-7333-8333-333333333333",
      decision_question: "Which migration path?",
      decided_at: "2026-02-01T12:00:00Z",
      relation: "contradicts",
      note: "held: the migration is a morning's work",
    },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe("the patterns page", () => {
  it("shows supporting and contradicting evidence at the same weight", async () => {
    stubFetch([
      { match: "/patterns/calibration", body: CALIBRATION },
      { match: "/patterns", body: [PATTERN] },
    ]);
    renderWithProviders(<PatternsPage />);

    // Both counts on the summary line, neither behind a disclosure.
    expect(await screen.findByText("4 supporting")).toBeInTheDocument();
    expect(screen.getByText("2 contradicting")).toBeInTheDocument();

    // Both columns rendered, both labelled with their length.
    expect(screen.getByText("supports (1)")).toBeInTheDocument();
    expect(screen.getByText("contradicts (1)")).toBeInTheDocument();

    // And the contradicting decision is as reachable as the supporting one:
    // clicking any piece of evidence opens the decision it came from.
    const supporting = screen.getByRole("link", { name: "Which deploy path?" });
    const contradicting = screen.getByRole("link", { name: "Which migration path?" });
    expect(supporting).toHaveAttribute(
      "href",
      "/decisions/22222222-2222-7222-8222-222222222222",
    );
    expect(contradicting).toHaveAttribute(
      "href",
      "/decisions/33333333-3333-7333-8333-333333333333",
    );
  });

  it("shows the detector's note beside each decision", async () => {
    stubFetch([
      { match: "/patterns/calibration", body: CALIBRATION },
      { match: "/patterns", body: [PATTERN] },
    ]);
    renderWithProviders(<PatternsPage />);

    // "supports" beside a title is not something a reader can check; the note
    // says what actually happened in that decision.
    expect(
      await screen.findByText(/failed: the deploy is straightforward/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/held: the migration is a morning's work/),
    ).toBeInTheDocument();
  });

  it("reads the empty state as a result, with the calibration table", async () => {
    stubFetch([
      { match: "/patterns/calibration", body: CALIBRATION },
      { match: "/patterns", body: [] },
    ]);
    renderWithProviders(<PatternsPage />);

    expect(
      await screen.findByText(/no patterns with sufficient support/),
    ).toBeInTheDocument();
    // Not a dead end: the bands say what the corpus does support.
    expect(screen.getByText(/decisions by stated confidence/)).toBeInTheDocument();
    const rows = screen.getAllByText(/within what the sample supports/);
    expect(rows).toHaveLength(2);
  });

  it("refuses to dismiss a pattern without a reason", async () => {
    const calls = stubFetch([
      { match: "/patterns/calibration", body: CALIBRATION },
      { match: "/patterns", body: [PATTERN] },
      { match: "/dismiss", status: 204 },
    ]);
    renderWithProviders(<PatternsPage />);
    await screen.findByText("4 supporting");

    const dismiss = screen.getByRole("button", { name: "dismiss" });
    expect(dismiss).toBeDisabled();

    await userEvent.type(
      screen.getByLabelText(`dismiss reason for ${PATTERN.id}`),
      "one belief, three phrasings",
    );
    expect(dismiss).toBeEnabled();
    await userEvent.click(dismiss);

    const posted = calls.find((call) => call.url.includes("/dismiss"));
    expect(posted?.method).toBe("POST");
    expect(posted?.body).toEqual({ reason: "one belief, three phrasings" });
  });

  it("marks a band that falls outside its interval", async () => {
    stubFetch([
      {
        match: "/patterns/calibration",
        body: {
          decisions: [{ ...CALIBRATION.decisions[0], miscalibrated: true }],
          assumptions: [],
        },
      },
      { match: "/patterns", body: [] },
    ]);
    renderWithProviders(<PatternsPage />);

    const table = await screen.findByText(/decisions by stated confidence/);
    expect(
      within(table.parentElement as HTMLElement).getByText("outside the interval"),
    ).toBeInTheDocument();
  });
});
