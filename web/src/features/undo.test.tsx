/**
 * Undo: that it restores the previous state, and does it inside the window.
 *
 * Pinning is the case worth testing end to end. It is the only reversible
 * action whose state lives entirely in the browser, so the assertion can be
 * about the *state* rather than about a request having been sent — and the
 * subtle half is position: a list that silently reorders when you take an
 * action back has not undone anything.
 *
 * The two undo paths that go through the API — archiving a session and
 * changing a judgement — are the same call with an inverted argument and a
 * re-post of the previous verdict respectively. See `SessionRail` and
 * `JudgementButtons`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ToastProvider } from "../app/Toaster";
import { PinButton } from "../components/PinButton";
import { TOAST_UNDO_MS } from "../lib/toast";
import { clearPins, getPins, togglePin } from "../lib/pins";

const A = { id: "aaa", label: "src/a.py" };
const B = { id: "bbb", label: "src/b.py" };
const C = { id: "ccc", label: "src/c.py" };

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  window.localStorage.clear();
  clearPins();
});
afterEach(() => {
  vi.useRealTimers();
  clearPins();
});

function setup(pin: { id: string; label: string }) {
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(
    <ToastProvider>
      <PinButton memoryId={pin.id} label={pin.label} />
    </ToastProvider>,
  );
  return user;
}

describe("undo", () => {
  it("puts an unpinned memory back where it was, not on the front", async () => {
    // Pinned newest-first, so the order is C, B, A and B sits in the middle.
    togglePin(A);
    togglePin(B);
    togglePin(C);
    expect(getPins().map((p) => p.id)).toEqual(["ccc", "bbb", "aaa"]);

    const user = setup(B);
    await user.click(screen.getByTestId("pin"));
    expect(getPins().map((p) => p.id)).toEqual(["ccc", "aaa"]);

    await user.click(screen.getByTestId("toast-undo"));

    // Back in the middle. Restoring to the front would look like an undo and
    // would have quietly reordered the list.
    expect(getPins().map((p) => p.id)).toEqual(["ccc", "bbb", "aaa"]);
  });

  it("stays available for the full five seconds", async () => {
    togglePin(A);
    const user = setup(A);
    await user.click(screen.getByTestId("pin"));

    // Just inside the window: still offered.
    await act(async () => {
      vi.advanceTimersByTime(TOAST_UNDO_MS - 200);
    });
    expect(screen.getByTestId("toast-undo")).toBeInTheDocument();

    await user.click(screen.getByTestId("toast-undo"));
    expect(getPins().map((p) => p.id)).toEqual(["aaa"]);
  });

  it("is gone once the window closes, and the action stands", async () => {
    togglePin(A);
    const user = setup(A);
    await user.click(screen.getByTestId("pin"));

    await act(async () => {
      vi.advanceTimersByTime(TOAST_UNDO_MS + 100);
    });

    expect(screen.queryByTestId("toast-undo")).not.toBeInTheDocument();
    expect(getPins()).toHaveLength(0);
  });

  it("offers no undo for pinning, which the same button already reverses", async () => {
    const user = setup(A);
    await user.click(screen.getByTestId("pin"));

    expect(screen.getByTestId("toast")).toHaveTextContent("Pinned");
    // Two ways to do one thing is worse than one, and the button is right there.
    expect(screen.queryByTestId("toast-undo")).not.toBeInTheDocument();
  });
});
