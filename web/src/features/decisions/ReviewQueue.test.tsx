/**
 * The one property of the review queue that a UI test can hold.
 *
 * Not "does it render" — the useful assertions are that the passage is on
 * screen beside every draft, and that neither verdict fires without a click.
 * The queue's whole safety value is that accepting is a considered act about
 * evidence; a screen that showed the draft alone, or that had a default action,
 * would be a queue in name only.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ReviewQueue } from "./ReviewQueue";
import { renderWithProviders, stubFetch } from "../../test/harness";

const PASSAGE =
  "RRF discards the scores and keeps the ordering, which is the part both " +
  "retrievers mean the same thing by. A weighted sum does not work because the " +
  "two numbers are not on comparable scales.";

function suggestion(overrides: Record<string, unknown> = {}) {
  return {
    id: "55555555-5555-7555-8555-555555555555",
    draft: {
      question: "How are two retrievers combined?",
      chosen: "Reciprocal rank fusion",
      reasoning: "The two score scales are not comparable.",
      // The three fields the prompt is told to leave empty. The queue reports
      // them as unstated rather than rendering blanks.
      confidence: null,
      expected_outcome: null,
      options: [
        { description: "A weighted sum of normalised scores", rejected_because: "Not comparable" },
      ],
      assumptions: [],
    },
    source_text: PASSAGE,
    source_name: "self",
    external_key: "README.md",
    chunk_ordinal: 12,
    status: "pending",
    model_id: "llama-3.3-70b-versatile",
    suggested_at: "2026-08-13T10:00:00Z",
    reviewed_at: null,
    decision_id: null,
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("the review queue", () => {
  it("shows the source passage beside the draft", async () => {
    stubFetch([{ match: "/decisions/suggestions", body: [suggestion()] }]);
    renderWithProviders(<ReviewQueue />);

    expect(await screen.findByText(/How are two retrievers combined/)).toBeInTheDocument();
    // The evidence, not a summary of it. Accepting has to be a judgement about
    // what the passage says.
    expect(screen.getByText(new RegExp("RRF discards the scores"))).toBeInTheDocument();
    expect(screen.getByText(/self:README.md#12/)).toBeInTheDocument();
  });

  it("says outright which fields the model did not fill in", async () => {
    stubFetch([{ match: "/decisions/suggestions", body: [suggestion()] }]);
    renderWithProviders(<ReviewQueue />);

    // Blank fields would read as a rendering bug. An empty confidence is a
    // claim — that the passage did not state one — and it is written down.
    expect(await screen.findByText(/confidence not stated/)).toBeInTheDocument();
  });

  it("commits nothing until a verdict is clicked", async () => {
    const calls = stubFetch([
      { match: "/decisions/suggestions?status", body: [suggestion()] },
      { match: "/accept", body: { id: "66666666" } },
    ]);
    renderWithProviders(<ReviewQueue />);
    await screen.findByText(/How are two retrievers combined/);

    // Rendering the queue is a read. Nothing here may write.
    expect(calls.every((call) => call.method === "GET")).toBe(true);

    await userEvent.click(screen.getByRole("button", { name: "accept" }));

    const accepted = calls.find((call) => call.url.includes("/accept"));
    expect(accepted?.method).toBe("POST");
  });

  it("rejects through its own endpoint rather than a verdict flag", async () => {
    const calls = stubFetch([
      { match: "/decisions/suggestions?status", body: [suggestion()] },
      { match: "/reject", status: 204 },
    ]);
    renderWithProviders(<ReviewQueue />);
    await screen.findByText(/How are two retrievers combined/);

    await userEvent.click(screen.getByRole("button", { name: "reject" }));

    const rejected = calls.find((call) => call.url.includes("/reject"));
    expect(rejected?.method).toBe("POST");
    // Accept writes a decision and reject does not, so they are separate
    // routes. A single endpoint taking a verdict would hide that.
    expect(calls.some((call) => call.url.includes("/accept"))).toBe(false);
  });
});
