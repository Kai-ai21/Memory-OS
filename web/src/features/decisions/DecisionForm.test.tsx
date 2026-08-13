/**
 * What the capture form refuses, and what it declines to invent.
 *
 * Two properties. A decision with no alternative never reaches the API, and the
 * refusal explains itself rather than greying out a button. And an empty
 * confidence stays null in the payload — a form that helpfully defaulted it to
 * 0.5 would put a number nobody held into the table M5.2 measures calibration
 * against.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { DecisionForm } from "./DecisionForm";
import { renderWithProviders, stubFetch } from "../../test/harness";

afterEach(() => vi.unstubAllGlobals());

describe("recording a decision", () => {
  it("refuses one with no alternative, and says why", async () => {
    const calls = stubFetch([{ match: "/decisions", body: { id: "1" } }]);
    renderWithProviders(<DecisionForm />);

    await userEvent.type(screen.getByLabelText("question"), "Which queue?");
    await userEvent.type(screen.getByLabelText("chosen"), "Postgres");
    await userEvent.click(screen.getByRole("button", { name: "record" }));

    expect(await screen.findByText(/at least one alternative/)).toBeInTheDocument();
    // Nothing was sent. The rule is the point of the form, so learning it should
    // not cost a round trip.
    expect(calls).toHaveLength(0);
  });

  it("sends a null confidence when the box is empty", async () => {
    const calls = stubFetch([{ match: "/decisions", body: { id: "1" } }]);
    renderWithProviders(<DecisionForm />);

    await userEvent.type(screen.getByLabelText("question"), "Which queue?");
    await userEvent.type(screen.getByLabelText("chosen"), "Postgres");
    await userEvent.type(screen.getByLabelText("option 1"), "Celery");
    await userEvent.type(
      screen.getByLabelText("rejected because 1"),
      "cannot share the transaction",
    );
    await userEvent.click(screen.getByRole("button", { name: "record" }));

    const posted = calls.find((call) => call.method === "POST");
    expect(posted?.body).toMatchObject({
      question: "Which queue?",
      chosen: "Postgres",
      // Null, not 0.5 and not 0. An invented number would be measured as
      // though somebody had held it.
      confidence: null,
      options: [{ description: "Celery", rejected_because: "cannot share the transaction" }],
      assumptions: [],
    });
  });

  it("keeps the assumptions the reviewer typed, one per row", async () => {
    const calls = stubFetch([{ match: "/decisions", body: { id: "1" } }]);
    renderWithProviders(<DecisionForm />);

    await userEvent.type(screen.getByLabelText("question"), "Which queue?");
    await userEvent.type(screen.getByLabelText("chosen"), "Postgres");
    await userEvent.type(screen.getByLabelText("option 1"), "Celery");
    await userEvent.type(screen.getByLabelText("assumption 1"), "throughput stays low");
    await userEvent.click(screen.getByRole("button", { name: "another assumption" }));
    await userEvent.type(screen.getByLabelText("assumption 2"), "Postgres is already required");
    await userEvent.click(screen.getByRole("button", { name: "record" }));

    const posted = calls.find((call) => call.method === "POST");
    expect(posted?.body).toMatchObject({
      assumptions: [
        { statement: "throughput stays low", confidence: null },
        { statement: "Postgres is already required", confidence: null },
      ],
    });
  });
});
