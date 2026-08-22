/**
 * The toast rules, as assertions.
 *
 * Two of the three rules in `lib/toast` are the kind that decay silently — a
 * second toast that stacks instead of replacing looks fine in the one case a
 * developer tries, and only becomes a notification centre in the hands of
 * somebody pinning five things in a row. So the count is pinned here.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ToastProvider } from "./Toaster";
import { TOAST_MS, useToast } from "../lib/toast";

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
afterEach(() => vi.useRealTimers());

/** A harness that can fire toasts on demand. */
function Harness() {
  const toast = useToast();
  return (
    <>
      <button type="button" onClick={() => toast.show("Pinned")}>
        first
      </button>
      <button type="button" onClick={() => toast.show("Unpinned")}>
        second
      </button>
    </>
  );
}

function setup() {
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(
    <ToastProvider>
      <Harness />
    </ToastProvider>,
  );
  return user;
}

describe("toasts", () => {
  it("renders exactly one at a time, the newest replacing the last", async () => {
    const user = setup();

    await user.click(screen.getByRole("button", { name: "first" }));
    expect(screen.getAllByTestId("toast")).toHaveLength(1);
    expect(screen.getByTestId("toast")).toHaveTextContent("Pinned");

    await user.click(screen.getByRole("button", { name: "second" }));

    // Still one. Two would already be a list, and three a notification centre.
    const toasts = screen.getAllByTestId("toast");
    expect(toasts).toHaveLength(1);
    expect(toasts[0]).toHaveTextContent("Unpinned");
    expect(screen.queryByText("Pinned")).not.toBeInTheDocument();
  });

  it("does not let a replaced toast's timer take the replacement down", async () => {
    const user = setup();

    await user.click(screen.getByRole("button", { name: "first" }));
    // Almost the full life of the first one.
    await act(async () => {
      vi.advanceTimersByTime(TOAST_MS - 100);
    });

    await user.click(screen.getByRole("button", { name: "second" }));
    // The first toast's timer would fire about here. The replacement must
    // survive it — this is the bug that makes a burst of actions leave nothing
    // on screen.
    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(screen.getByTestId("toast")).toHaveTextContent("Unpinned");
  });

  it("goes away on its own, and can be dismissed sooner", async () => {
    const user = setup();

    await user.click(screen.getByRole("button", { name: "first" }));
    await act(async () => {
      vi.advanceTimersByTime(TOAST_MS + 50);
    });
    expect(screen.queryByTestId("toast")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "first" }));
    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByTestId("toast")).not.toBeInTheDocument();
  });

  it("keeps the live region mounted even with nothing to say", () => {
    setup();
    // A live region created at the same moment its content arrives is one most
    // assistive technology does not read.
    expect(screen.getByTestId("toast-region")).toBeInTheDocument();
    expect(screen.getByTestId("toast-region")).toHaveAttribute("aria-live", "polite");
  });
});
