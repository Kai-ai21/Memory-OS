/**
 * What the outcome queue has to put on screen before anybody can judge it.
 *
 * Every candidate here is in the queue because one thing occurred after
 * another, which is not evidence of anything on its own. So the temporal gap is
 * asserted to be visible, and so is `entity_filter` when it says `unavailable`
 * — a candidate found by time alone is much weaker than one sharing a resolved
 * entity, and a queue that hid the difference would change meaning silently
 * depending on whether anybody had run extraction lately.
 *
 * The third assertion is the one the milestone turns on: accepting says
 * `inferred`, on the button, because accepting a reading is not the same as
 * having watched something happen.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { OutcomeQueue } from "./OutcomeQueue";
import { renderWithProviders, stubFetch } from "../../test/harness";

const RATE = {
  worked: 1,
  failed: 0,
  mixed: 0,
  too_early: 3,
  undecided: 12,
  resolved: 1,
  rate: 1.0,
};

function candidate(overrides: Record<string, unknown> = {}) {
  return {
    id: "77777777-7777-7777-8777-777777777777",
    decision_id: "88888888-8888-7888-8888-888888888888",
    decision_question: "What runs background work?",
    decision_decided_at: "2026-08-08T12:00:00Z",
    draft: {
      description: "The queue drained without a broker.",
      verdict: "worked",
      rationale: "the worker claims a task and holds a lease",
      judged_confidence: 0.8,
    },
    source_text: "The worker claims a task from the queue and holds a lease on it.",
    source_name: "self",
    external_key: "src/memoryos/application/worker.py",
    candidate_occurred_at: "2026-08-10T12:00:00Z",
    gap_days: 2.0,
    window_days: 165,
    shared_entities: ["postgres", "sqlalchemy"],
    entity_filter: "applied",
    status: "pending",
    model_id: "llama-3.3-70b-versatile",
    suggested_at: "2026-08-13T10:00:00Z",
    reviewed_at: null,
    outcome_id: null,
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("the outcome review queue", () => {
  it("states the temporal gap and the window that admitted it", async () => {
    stubFetch([
      { match: "/outcomes/rate", body: RATE },
      { match: "/outcomes/suggestions", body: [candidate()] },
    ]);
    renderWithProviders(<OutcomeQueue />);

    // The gap is the entire claim. It belongs on screen rather than folded
    // into a confidence the reviewer would have to trust.
    expect(await screen.findByText(/2\.0 days later/)).toBeInTheDocument();
    expect(screen.getByText(/window 165d/)).toBeInTheDocument();
    expect(screen.getByText(/postgres, sqlalchemy/)).toBeInTheDocument();
  });

  it("does not round a sub-day gap to zero days", async () => {
    // The condition this corpus is actually in: every mtime falls inside a
    // 2-day-18-hour window, so files written in one batch are minutes apart.
    // "0.0 days" would read as a very tight correlation; it is the temporal
    // signal saying it has nothing to offer.
    stubFetch([
      { match: "/outcomes/rate", body: RATE },
      { match: "/outcomes/suggestions", body: [candidate({ gap_days: 0.02 })] },
    ]);
    renderWithProviders(<OutcomeQueue />);

    expect(await screen.findByText(/29 minutes later/)).toBeInTheDocument();
    expect(screen.queryByText(/0\.0 days/)).not.toBeInTheDocument();
  });

  it("shows the decision beside the candidate", async () => {
    stubFetch([
      { match: "/outcomes/rate", body: RATE },
      { match: "/outcomes/suggestions", body: [candidate()] },
    ]);
    renderWithProviders(<OutcomeQueue />);

    expect(await screen.findByText(/What runs background work/)).toBeInTheDocument();
    expect(
      screen.getByText(/The queue drained without a broker/),
    ).toBeInTheDocument();
  });

  it("says when the entity filter could not be applied at all", async () => {
    stubFetch([
      { match: "/outcomes/rate", body: RATE },
      {
        match: "/outcomes/suggestions",
        body: [candidate({ entity_filter: "unavailable", shared_entities: [] })],
      },
    ]);
    renderWithProviders(<OutcomeQueue />);

    // Twice, and both are wanted: once in the header as a count of how much of
    // the queue is weaker than it looks, once on the candidate itself. Not "no
    // shared entities" — the test could not be run at all, and conflating the
    // two is what this column exists to prevent.
    const said = await screen.findAllByText(/found by time alone/);
    expect(said).toHaveLength(2);
    expect(
      screen.getByText(/entity filter unavailable: found by time alone/),
    ).toBeInTheDocument();
  });

  it("keeps too_early and undecided outside the success rate", async () => {
    stubFetch([
      { match: "/outcomes/rate", body: RATE },
      { match: "/outcomes/suggestions", body: [candidate()] },
    ]);
    renderWithProviders(<OutcomeQueue />);

    expect(await screen.findByText("3 too early")).toBeInTheDocument();
    expect(screen.getByText("12 not looked at")).toBeInTheDocument();
    // One worked over one resolved. The fifteen unresolved are not in it.
    expect(screen.getByText("100% of 1 resolved")).toBeInTheDocument();
  });

  it("accepts as inferred, and says so on the button", async () => {
    const calls = stubFetch([
      { match: "/outcomes/rate", body: RATE },
      { match: "/outcomes/suggestions?status", body: [candidate()] },
      { match: "/accept", body: { id: "99999999" } },
    ]);
    renderWithProviders(<OutcomeQueue />);
    await screen.findByText(/2\.0 days later/);

    // Rendering the queue is a read. Nothing here may write.
    expect(calls.every((call) => call.method === "GET")).toBe(true);

    const accept = screen.getByRole("button", { name: "accept as inferred" });
    await userEvent.click(accept);

    expect(calls.find((call) => call.url.includes("/accept"))?.method).toBe("POST");
  });
});
