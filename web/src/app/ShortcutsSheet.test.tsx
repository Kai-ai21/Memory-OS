/**
 * `?` opens the sheet and `Esc` closes it.
 *
 * Driven through `App` rather than by rendering `ShortcutsSheet` with `open`
 * flipped by hand, because the thing worth testing is the *binding*. A sheet
 * that renders correctly when told to is not the failure mode here — a sheet
 * nothing opens is, and that is exactly the state the other four shortcuts
 * were in before this milestone, working and undiscoverable.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../test/harness";

afterEach(() => vi.unstubAllGlobals());

function stubShell() {
  return stubFetch([...SHELL_ROUTES, { match: "/chat", body: [] }]);
}

describe("the shortcuts sheet", () => {
  it("opens on ? and closes on Esc", async () => {
    stubShell();
    renderWithProviders(<App />, { route: "/" });

    // Not there until asked for. The sheet is mounted only while open — see
    // the note in `ShortcutsSheet` — so absence is absence from the document.
    expect(screen.queryByTestId("shortcuts-sheet")).not.toBeInTheDocument();

    /* Focus has to leave the composer first, and that is not a workaround for
       the test — it is the binding behaving correctly. `/` is autofocused into
       the chat box on this route, `?` is a character, and a shortcut that fired
       from inside a textarea would make the question mark untypeable. The next
       test pins that guard directly; the sidebar's `Shortcuts` row is what
       covers the person sitting in the composer. */
    (document.activeElement as HTMLElement | null)?.blur();

    await userEvent.keyboard("?");

    const sheet = await screen.findByTestId("shortcuts-sheet");
    expect(sheet).toBeInTheDocument();

    // It lists the bindings it claims to. ⌘K is the one the sidebar already
    // advertised; `/` is one of the four that were documented nowhere.
    expect(screen.getByRole("heading", { name: /keyboard shortcuts/i })).toBeInTheDocument();
    expect(screen.getByText(/command palette/i)).toBeInTheDocument();
    expect(screen.getByText(/focus the search box/i)).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");

    await waitFor(() =>
      expect(screen.queryByTestId("shortcuts-sheet")).not.toBeInTheDocument(),
    );
  });

  it("does not open while something is being typed into", async () => {
    // `?` is a character. The chat composer is focused on load, and a shortcut
    // that fired from inside a textarea would make the question mark
    // untypeable — which is the bug this guard exists to prevent.
    stubShell();
    renderWithProviders(<App />, { route: "/" });

    const box = await screen.findByLabelText("Message");
    await userEvent.click(box);
    await userEvent.keyboard("does this work?");

    expect(screen.queryByTestId("shortcuts-sheet")).not.toBeInTheDocument();
    expect(box).toHaveValue("does this work?");
  });

  it("is reachable from More in the sidebar, for anyone who does not know ?", async () => {
    // The row exists precisely because a shortcut that can only be discovered
    // by pressing it is not discoverable.
    stubShell();
    renderWithProviders(<App />, { route: "/" });

    await userEvent.click(await screen.findByRole("button", { name: /more/i }));
    await userEvent.click(await screen.findByRole("menuitem", { name: /shortcuts/i }));

    expect(await screen.findByTestId("shortcuts-sheet")).toBeInTheDocument();
  });
});
