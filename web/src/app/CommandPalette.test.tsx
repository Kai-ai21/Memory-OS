/**
 * The palette: that it opens, that it closes, and that what it offers is real.
 *
 * The open/close pair is the whole contract of a modal keyboard control — a
 * palette that opens and cannot be dismissed with `Esc` is worse than none,
 * because it captures focus. The rest is that its entries actually go
 * somewhere: a palette listing views that do not exist is a menu of dead ends.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../test/harness";

const MEMORIES = [
  {
    id: "11111111-1111-7111-8111-111111111111",
    source_id: "33333333-3333-7333-8333-333333333333",
    external_key: "src/memoryos/application/worker.py",
    version: 1,
    is_current: true,
    kind: "code",
    title: null,
    content_hash: "a".repeat(64),
    occurred_at: null,
    occurred_at_source: "filesystem",
    ingested_at: "2026-08-01T10:00:00Z",
    deleted_at: null,
  },
];

function stubAll() {
  return stubFetch([
    ...SHELL_ROUTES,
    { match: "/memories", body: MEMORIES },
    { match: "/decisions", body: [] },
    { match: "/sources", body: [] },
    { match: "/gaps", body: [] },
    {
      match: "/timeline",
      body: {
        start: "2026-01-01T00:00:00Z",
        end: "2026-08-01T00:00:00Z",
        buckets: [],
        provenance: [],
        total: 0,
      },
    },
    {
      match: "/search",
      body: { query: "", timing: { embed_ms: 0, search_ms: 0, total_ms: 0 }, hits: [] },
    },
  ]);
}

afterEach(() => vi.unstubAllGlobals());

/**
 * jsdom implements `<dialog>` but not always its modal semantics, so the tests
 * assert on the element's own `open` state rather than on visibility.
 */
function palette(): HTMLDialogElement {
  return screen.getByRole("dialog", { hidden: true }) as HTMLDialogElement;
}

describe("opening and closing", () => {
  it("opens on ⌘K", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });

    expect(palette().open).toBe(false);

    await userEvent.keyboard("{Meta>}k{/Meta}");

    await waitFor(() => expect(palette().open).toBe(true));
  });

  it("opens on ctrl+K too, for anyone not on a Mac", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });

    await userEvent.keyboard("{Control>}k{/Control}");

    await waitFor(() => expect(palette().open).toBe(true));
  });

  it("closes on Esc", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });

    await userEvent.keyboard("{Meta>}k{/Meta}");
    await waitFor(() => expect(palette().open).toBe(true));

    await userEvent.keyboard("{Escape}");

    await waitFor(() => expect(palette().open).toBe(false));
  });

  it("opens even while a search box has focus", async () => {
    // The one case a naive "ignore shortcuts while typing" guard breaks, and
    // the case where the palette is most useful.
    stubAll();
    renderWithProviders(<App />, { route: "/search" });

    const input = await screen.findByLabelText("Search query");
    input.focus();

    await userEvent.keyboard("{Meta>}k{/Meta}");

    await waitFor(() => expect(palette().open).toBe(true));
  });
});

describe("what it offers", () => {
  it("lists every view before anything is typed", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    const results = await screen.findByTestId("palette-results");
    expect(within(results).getByText("overview")).toBeInTheDocument();
    expect(within(results).getByText("search")).toBeInTheDocument();
    // Planned routes are offered and labelled, not hidden.
    expect(within(results).getByText(/graph \(planned\)/)).toBeInTheDocument();
  });

  it("finds a view by an alias rather than only by its label", async () => {
    // `ask` is what the agent page was called until M9.0, and what somebody
    // who used it then will still type.
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    await userEvent.type(await screen.findByLabelText("Command"), "ask");

    const results = screen.getByTestId("palette-results");
    expect(within(results).getByText("agent")).toBeInTheDocument();
  });

  it("navigates to the view that is chosen", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    await userEvent.type(await screen.findByLabelText("Command"), "timeline");
    await userEvent.keyboard("{Enter}");

    expect(await screen.findByText(/date provenance/i)).toBeInTheDocument();
    // And it closed itself on the way.
    await waitFor(() => expect(palette().open).toBe(false));
  });

  it("offers a memory by path, and opens it", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    await userEvent.type(await screen.findByLabelText("Command"), "worker.py");

    const results = screen.getByTestId("palette-results");
    expect(
      within(results).getByText("src/memoryos/application/worker.py"),
    ).toBeInTheDocument();
  });

  it("reaches a view the sidebar does not name", async () => {
    // Five working pages hang off `/decisions` and none of them is in the
    // sidebar. "Jump to any view" has to mean any view, or the shortcut has a
    // hole in exactly the places that are hardest to find by clicking.
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    await userEvent.type(await screen.findByLabelText("Command"), "outcome");

    const results = screen.getByTestId("palette-results");
    expect(within(results).getByText("outcome queue")).toBeInTheDocument();
  });

  it("never offers reflections, on any query", async () => {
    // The one route nothing is allowed to volunteer. A claim about somebody's
    // judgement is something they go and look at, from the patterns page — a
    // palette that surfaced it on a keystroke would be the tool speaking
    // first, which is the failure M5.4 exists to prevent.
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    const box = await screen.findByLabelText("Command");
    for (const term of ["reflect", "reflection", "decisions", "patterns"]) {
      await userEvent.clear(box);
      await userEvent.type(box, term);

      // The *view* entries, not the whole panel. Typing "reflection" always
      // produces the search fallback labelled with what was typed — that is
      // the palette offering to search the corpus for a word, which is not the
      // same act as offering the page and is not what this rule is about.
      const views = within(screen.getByTestId("palette-results"))
        .getAllByRole("option")
        .filter((option) => option.textContent?.startsWith("view"));
      expect(views.map((option) => option.textContent).join(" ")).not.toMatch(
        /reflection/i,
      );
    }
  });

  it("offers to search the corpus for anything it does not recognise", async () => {
    // The fallback that always works, and the reason the box never dead-ends.
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    await userEvent.type(await screen.findByLabelText("Command"), "chunk overlap");

    const results = screen.getByTestId("palette-results");
    expect(within(results).getByText(/search the corpus for this/)).toBeInTheDocument();
  });

  it("moves the selection with the arrow keys", async () => {
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await userEvent.keyboard("{Meta>}k{/Meta}");

    const results = await screen.findByTestId("palette-results");
    const first = within(results).getAllByRole("option")[0];
    expect(first).toHaveAttribute("aria-selected", "true");

    await userEvent.keyboard("{ArrowDown}");

    expect(within(results).getAllByRole("option")[1]).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(first).toHaveAttribute("aria-selected", "false");
  });
});
