/**
 * The four things about the landing page that looking at it will not tell you.
 *
 * Three of them are the defects the reference implementation shipped with, in
 * the form they would come back: a canvas that ignores the accessibility
 * setting it exists to respect, a frame loop that outlives the page that
 * started it, and a shell that follows a route it has no business on. The
 * fourth is the only control on the screen that does anything.
 *
 * **No real frame loop runs here.** `requestAnimationFrame` is replaced with a
 * recording stub. The cancellation test in particular has to be asserted on
 * the call rather than on a timer, because a leaked loop's whole symptom is
 * that nothing observable happens — it just keeps costing.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../../App";
import { WelcomePage } from "./WelcomePage";
import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../../test/harness";

/** Which media queries answer true. jsdom ships no `matchMedia` at all. */
function stubMedia(truthy: string[] = []) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: truthy.some((needle) => query.includes(needle)),
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  );
}

/**
 * A 2D context that records nothing and draws nothing.
 *
 * jsdom's `getContext` returns null without the `canvas` package, and the
 * component correctly bails out on a null context — which would make the
 * cancellation test pass because no loop was ever started.
 *
 * Wide enough to satisfy the application background as well as this page: the
 * `Open MEMO` test navigates into the shell, which mounts `BackgroundLayer`,
 * which builds a gradient sprite on the way up.
 */
function stubCanvas() {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform: vi.fn(),
    fillRect: vi.fn(),
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    drawImage: vi.fn(),
    createRadialGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
    globalAlpha: 1,
    globalCompositeOperation: "source-over",
    fillStyle: "",
  } as unknown as CanvasRenderingContext2D);
}

/** A recording `requestAnimationFrame` that never actually runs a callback. */
function stubFrames() {
  const requested = vi.fn(() => 77);
  const cancelled = vi.fn();
  vi.stubGlobal("requestAnimationFrame", requested);
  vi.stubGlobal("cancelAnimationFrame", cancelled);
  return { requested, cancelled };
}

/** Everything any route reached from this page might ask for. */
function stubEverything() {
  return stubFetch([
    ...SHELL_ROUTES,
    { match: "/chat", body: [] },
    { match: "/sources", body: [] },
  ]);
}

beforeEach(() => stubCanvas());

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("the particle canvas", () => {
  it("is not created at all under prefers-reduced-motion: reduce", () => {
    // **Defect 4.** Not a paused canvas and not a slower one. Eight hundred
    // drifting specks across a full viewport is a vestibular trigger, and this
    // is the setting somebody turns on to be spared exactly this — so the
    // correct amount is none, and the page still has to be a page without it.
    stubMedia(["prefers-reduced-motion"]);
    stubFrames();

    renderWithProviders(<WelcomePage />, { route: "/welcome" });

    expect(screen.queryByTestId("fluid-particles")).not.toBeInTheDocument();
    // The content is the page. It does not depend on the effect existing.
    expect(screen.getByRole("heading", { name: "MEMO" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open memo/i })).toBeInTheDocument();
  });

  it("cancels its animation frame on unmount", () => {
    // **Defect 2.** The reference's cleanup removed the resize listener and
    // left the loop running — for the life of the tab, at sixty frames a
    // second, drawing into a canvas detached from the document. Nothing about
    // that is visible; it is purely a bill.
    stubMedia();
    const frames = stubFrames();

    const view = renderWithProviders(<WelcomePage />, { route: "/welcome" });
    expect(screen.getByTestId("fluid-particles")).toBeInTheDocument();
    expect(frames.requested).toHaveBeenCalled();
    expect(frames.cancelled).not.toHaveBeenCalled();

    view.unmount();

    expect(frames.cancelled).toHaveBeenCalledWith(77);
  });
});

describe("the route", () => {
  it("renders /welcome without the app shell", () => {
    // The landing page is outside the shell, which is a routing fact rather
    // than a styling one: `Shell` mounts the sidebar, the command palette and
    // the global keyboard model, and none of the three mean anything on a
    // front door. Asserted on the nav landmark, which is what the shell
    // contributes and what nothing else on this page renders.
    stubMedia();
    stubFrames();
    stubEverything();

    renderWithProviders(<App />, { route: "/welcome" });

    expect(screen.queryByRole("navigation")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "MEMO" })).toBeInTheDocument();
  });

  it("takes Open MEMO to the chat, and leaves the form inert", async () => {
    // The one control on the page that does anything, and the reason the
    // accent is spent on it rather than on the button above it.
    stubMedia();
    stubFrames();
    stubEverything();

    renderWithProviders(<App />, { route: "/welcome" });

    await userEvent.click(screen.getByRole("link", { name: /open memo/i }));

    expect(
      await screen.findByRole("heading", { name: /say it here/i }),
    ).toBeInTheDocument();
    // And the shell is back, because that route is inside it.
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
  });
});

describe("the sign-in shell", () => {
  it("renders the form and disables every control in it", () => {
    // There is no user system in this project. A form that looked operable
    // would be an interface telling a lie, and the worst version of that lie
    // is the one that takes a password first. `<fieldset disabled>` makes
    // "inert" a fact about the document rather than a styling choice.
    stubMedia();
    stubFrames();

    renderWithProviders(<WelcomePage />, { route: "/welcome" });

    expect(screen.getByLabelText(/email/i)).toBeDisabled();
    expect(screen.getByLabelText(/password/i)).toBeDisabled();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    expect(screen.getByText(/sign-in isn’t active yet/i)).toBeInTheDocument();
  });
});
