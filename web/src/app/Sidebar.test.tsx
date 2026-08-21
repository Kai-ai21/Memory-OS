/**
 * The panel, at the points a refinement can quietly undo itself.
 *
 * M9.8's tests counted the rows, because that milestone removed the headings
 * and a group boundary is invisible to a route test. M9.9's are about *how the
 * rows are set*, which is even more invisible: a label that drifts back to
 * lowercase, a row that loses its glyph, or a `.meta` class copied into the nav
 * from anywhere else in the application would all render, navigate, and pass
 * every test this suite had before this file.
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

/** The panel's root, which is what every structural query starts from. */
function panel(): HTMLElement {
  return screen.getByRole("navigation", { name: "Views" }).parentElement!;
}

/** Every row in the panel: the nav, plus `Details` and `More` in the footer. */
function rows(): HTMLElement[] {
  return Array.from(panel().querySelectorAll<HTMLElement>("a.nav-item, button.nav-item"));
}

/**
 * Every class in this application that resolves to JetBrains Mono.
 *
 * Listed rather than detected, because jsdom applies no stylesheet: the class
 * is the contract, and `index.css` is where each of these sets `--font-mono`.
 */
const MONO = ["meta", "meta-label", "meta-label-on", "kbd", "code-content", "font-mono"];

describe("every row", () => {
  it("renders with a glyph and a sentence-case label", () => {
    // **The inventory, counted rather than trusted.** The nav's share of it
    // comes from `ALL_ROUTES`, which is the one place a view is declared, so
    // this fails when somebody adds a page and forgets the sidebar rather than
    // when somebody adds a page. `New chat`, `Details` and `More` are the three
    // rows that are not routes and are named here because nothing else names
    // them.
    stubFetch(SHELL_ROUTES);
    renderWithProviders(<Sidebar />);

    const labels = rows().map((row) => row.textContent?.trim() ?? "");

    expect(labels).toEqual([
      "New chat",
      ...ALL_ROUTES.map(
        (route) => route.label.charAt(0).toUpperCase() + route.label.slice(1),
      ),
      "Details",
      "More",
    ]);

    for (const row of rows()) {
      // One glyph, and the label beside it capitalised. A row that lost its
      // icon would look like a mistake and break nothing.
      expect(row.querySelector("svg")).toBeTruthy();
      const label = row.querySelector("span")?.textContent ?? "";
      expect(label).toMatch(/^[A-Z]/);
      expect(label).not.toBe(label.toUpperCase());
    }
  });

  it("keeps the two groups, separated by a rule rather than a heading", () => {
    // The rule is a `::before` on `.nav-group + .nav-group`, so the DOM
    // evidence for a group boundary is the class — and a nav flattened into one
    // block would still render every row and still pass the count above.
    stubFetch(SHELL_ROUTES);
    renderWithProviders(<Sidebar />);

    const nav = screen.getByRole("navigation", { name: "Views" });
    const blocks = Array.from(nav.querySelectorAll<HTMLElement>(".nav-group"));

    expect(blocks).toHaveLength(2);
    expect(within(blocks[0]).getAllByRole("link")).toHaveLength(
      inGroup("primary").length + 1, // + chat, which sits above the groups
    );
    expect(within(blocks[1]).getAllByRole("link")).toHaveLength(
      inGroup("secondary").length,
    );
    expect(within(nav).queryAllByRole("heading")).toHaveLength(0);
  });
});

describe("where you are", () => {
  it("marks the active row without spending the accent on it", () => {
    // **Rule 1, pinned at the point it is most tempting to break.** The active
    // item is a `surface-tint` fill and a jump to 600. If the accent comes
    // back, the one blue thing on the screen is the one thing you cannot click.
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
    expect(current[0]).toHaveTextContent("Timeline");
    expect(current[0].className).toMatch(/nav-item-on/);
    expect(current[0].className).not.toMatch(/accent/);
  });
});

describe("the mono", () => {
  it("appears nowhere in the panel except inside the details block", async () => {
    // **The whole argument of M9.9 in one assertion.** JetBrains Mono is this
    // application's register for paths, scores, offsets and hashes; navigation
    // is prose. The exception is four counts that are read against each other
    // and need a column that lines up, and it is an exception precisely
    // because it is bounded — so this opens the disclosure first, which is the
    // only state in which any mono is allowed to exist here at all.
    stubFetch(SHELL_ROUTES);
    renderWithProviders(<Sidebar />);

    await userEvent.click(screen.getByTestId("details-toggle"));
    await screen.findByTestId("sidebar-figures");

    const details = document.getElementById("sidebar-details");
    expect(details).toBeTruthy();

    const monospaced = Array.from(
      panel().querySelectorAll<HTMLElement>(MONO.map((name) => `.${name}`).join(",")),
    );

    // There is some — the figures — and every last piece of it is in the block.
    expect(monospaced.length).toBeGreaterThan(0);
    for (const element of monospaced) {
      expect(details!.contains(element)).toBe(true);
    }
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
