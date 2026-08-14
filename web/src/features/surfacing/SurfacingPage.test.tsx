/**
 * Three properties, and every one of them is about not flattering the feature.
 *
 * **A refusal is on the page.** The gate stays quiet far more often than it
 * speaks, and a screen showing only the interruptions cannot distinguish a
 * careful system from one that never ran.
 *
 * **The dismissal rate says "noise" when it is above half.** In those words. A
 * proactive tool reports its own failure or nobody does.
 *
 * **Feedback reaches the API and the buttons go away.** Asking "was this
 * useful?" about something already judged is asking a question you have the
 * answer to.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SurfacingPage } from "./SurfacingPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

const SURFACED = {
  id: "11111111-1111-7111-8111-111111111111",
  focus: "src/memoryos/application/events.py",
  reason: "cleared",
  explanation: "two independent routes agreed on something you do not already have open",
  score: 0.0331,
  threshold: 0.0295,
  top_key: "memory:abc",
  top_title: "self::tests/integration/test_events.py",
  item_count: 9,
  trigger_kind: "file_focused",
  decided_at: "2026-08-14T10:00:00Z",
  surfaced: true,
  verdict: null,
};

const REFUSED = {
  ...SURFACED,
  id: "22222222-2222-7222-8222-222222222222",
  focus: "src/memoryos/domain/patterns.py",
  top_title: "self::scripts/seed_decisions.py",
  reason: "below_threshold",
  explanation:
    "the best item did not clear this focus's bar — one route found it, or two found it well down their rankings",
  score: 0.0164,
  threshold: 0.0295,
  surfaced: false,
  verdict: null,
};

afterEach(() => vi.unstubAllGlobals());

describe("the surfacing page", () => {
  it("shows what it refused beside what it surfaced", async () => {
    stubFetch([{ match: "/surfacing", body: [SURFACED, REFUSED] }]);
    renderWithProviders(<SurfacingPage />);

    expect(await screen.findByText(/test_events\.py/)).toBeInTheDocument();
    expect(screen.getByText(/patterns\.py/)).toBeInTheDocument();
    // And the refusal says how close it came, which is the first thing anybody
    // asks about silence.
    expect(screen.getByText(/scored 0\.0164 against 0\.0295/)).toBeInTheDocument();
  });

  it("calls exactly half exactly half, rather than rounding it kindly", async () => {
    const dismissed = { ...SURFACED, id: "33333333-3333-7333-8333-333333333333", verdict: "dismissed" };
    stubFetch([{ match: "/surfacing", body: [SURFACED, dismissed] }]);
    renderWithProviders(<SurfacingPage />);

    // One of two dismissed is exactly half, which is neither above the line
    // nor below it — and is the number the first real run produced, so it gets
    // its own sentence rather than being rounded into a flattering one.
    expect(await screen.findByText(/1 of 2 surfaced/)).toBeInTheDocument();
    expect(screen.queryByText(/means this is noise/)).not.toBeInTheDocument();
    expect(screen.getByText(/exactly half/)).toBeInTheDocument();
    expect(screen.queryByText(/Below half/)).not.toBeInTheDocument();
  });

  it("says so when more than half of what it volunteered was refused", async () => {
    const dismissed = { ...SURFACED, id: "33333333-3333-7333-8333-333333333333", verdict: "dismissed" };
    const alsoDismissed = { ...dismissed, id: "44444444-4444-7444-8444-444444444444" };
    stubFetch([{ match: "/surfacing", body: [SURFACED, dismissed, alsoDismissed] }]);
    renderWithProviders(<SurfacingPage />);

    expect(await screen.findByText(/means this is noise/)).toBeInTheDocument();
  });

  it("posts a verdict and offers no buttons on a judged row", async () => {
    const calls = stubFetch([
      { match: "/surfacing/11111111-1111-7111-8111-111111111111/dismiss", status: 204 },
      { match: "/surfacing", body: [SURFACED, { ...REFUSED, verdict: "useful", surfaced: true }] },
    ]);
    renderWithProviders(<SurfacingPage />);

    // One pair of buttons: the surfaced-and-unjudged row. The other row was
    // already rated, so its question has been answered.
    expect(await screen.findAllByRole("button", { name: "dismiss" })).toHaveLength(1);

    await userEvent.click(screen.getByRole("button", { name: "dismiss" }));

    const posted = calls.find((call) => call.method === "POST");
    expect(posted?.url).toContain(`/surfacing/${SURFACED.id}/dismiss`);
  });

  it("explains what the gate does rather than looking broken when empty", async () => {
    stubFetch([{ match: "/surfacing", body: [] }]);
    renderWithProviders(<SurfacingPage />);

    expect(await screen.findByTestId("empty")).toHaveTextContent(/defaults to silence/);
  });
});
