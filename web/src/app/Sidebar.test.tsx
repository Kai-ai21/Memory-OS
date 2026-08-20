/**
 * The panel, at the three points M9.8 could quietly break it.
 *
 * The redesign removed the group headings, the wordmark, the filled button and
 * the two pinned rows. Every one of those removals is invisible to a route
 * test — the nav still navigates — and two of them are exactly how a nav loses
 * an item without anybody noticing: a view that drops out of the table renders
 * nothing, and a heading that stops rendering takes the group boundary with it.
 * So the first test counts, and it counts *per group*.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../test/harness";
import { ALL_ROUTES, inGroup } from "./routes";
import { Sidebar } from "./Sidebar";

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

/** The nav's own group blocks, in order. */
function groups(nav: HTMLElement): HTMLElement[] {
  return Array.from(nav.querySelectorAll<HTMLElement>(":scope > div"));
}

describe("the thirteen views", () => {
  it("renders every one of them, in its group, with no headings left", () => {
    // Thirteen is not a number this file gets to choose: it is the length of
    // the route table, which is the one place a view is declared. Comparing
    // against `ALL_ROUTES` rather than a literal is what makes this test fail
    // when somebody adds a page and forgets the sidebar, instead of when
    // somebody adds a page.
    stubFetch(SHELL_ROUTES);
    renderWithProviders(<Sidebar />);

    const nav = screen.getByRole("navigation", { name: "Views" });
    const labels = within(nav)
      .getAllByRole("link")
      .map((link) => link.textContent);

    expect(labels).toEqual(ALL_ROUTES.map((route) => route.label));
    expect(labels).toHaveLength(13);

    // **The groups survive the headings.** They are 20px of space now, so the
    // only thing left that says a group exists is the DOM structure — and a
    // flattened nav would still render all thirteen rows and still pass every
    // assertion above.
    const blocks = groups(nav);
    expect(blocks).toHaveLength(2);
    expect(within(blocks[0]).getAllByRole("link")).toHaveLength(
      inGroup("primary").length + 1, // + chat, which sits above the groups
    );
    expect(within(blocks[1]).getAllByRole("link")).toHaveLength(
      inGroup("secondary").length,
    );

    // No heading, and specifically not the one that used to be here.
    expect(within(nav).queryAllByRole("heading")).toHaveLength(0);
    expect(within(nav).queryByText("more")).not.toBeInTheDocument();
  });

  it("gives every row a glyph, and starts the list with new chat", () => {
    // The icons are what make thirteen rows read as navigation rather than as
    // a list of words — the argument for the whole change. A row that lost its
    // glyph would look like a mistake and break nothing.
    stubFetch(SHELL_ROUTES);
    renderWithProviders(<Sidebar />);

    const nav = screen.getByRole("navigation", { name: "Views" });
    for (const link of within(nav).getAllByRole("link")) {
      expect(link.querySelector("svg")).toBeTruthy();
    }

    // A row, not a filled button: it wears the same class every nav item does.
    const start = within(nav).getByRole("button", { name: "new chat" });
    expect(start.className).toMatch(/nav-item/);
    expect(start.querySelector("svg")).toBeTruthy();
  });
});

describe("where you are", () => {
  it("marks the active row without spending the accent on it", () => {
    // **Rule 1, pinned at the point it is most tempting to break.** The active
    // item is a tint and a step up in weight. If the accent comes back, the
    // one blue thing on the screen is the one thing you cannot click.
    //
    // Asserted on the class contract because jsdom applies no stylesheet; the
    // colour behind `nav-item-on` is checked in `styles/theme.test.ts`.
    stubFetch(SHELL_ROUTES);
    renderWithProviders(<Sidebar />, { route: "/timeline" });

    const nav = screen.getByRole("navigation", { name: "Views" });
    const current = within(nav)
      .getAllByRole("link")
      .filter((link) => link.getAttribute("aria-current") === "page");

    expect(current).toHaveLength(1);
    expect(current[0]).toHaveTextContent("timeline");
    expect(current[0].className).toMatch(/nav-item-on/);
    expect(current[0].className).not.toMatch(/accent/);
  });
});

describe("the details block", () => {
  it("stays as you left it across a reload", async () => {
    // The disclosure is the whole argument for moving the corpus figures out
    // of sight: they cost six permanent rows and they are worth one click —
    // but only if the click is not charged again on every load.
    stubFetch(SHELL_ROUTES);
    const first = renderWithProviders(<Sidebar />);

    expect(screen.queryByTestId("sidebar-figures")).not.toBeInTheDocument();
    await userEvent.click(screen.getByTestId("details-toggle"));
    expect(await screen.findByTestId("sidebar-figures")).toBeInTheDocument();

    // The reload: every piece of React state goes, and a fresh tree reads the
    // preference back off `localStorage` the way a new tab would.
    first.unmount();
    renderWithProviders(<Sidebar />);

    expect(await screen.findByTestId("sidebar-figures")).toBeInTheDocument();
    expect(screen.getByTestId("details-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });
});
