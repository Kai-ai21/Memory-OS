/**
 * The split panel: that `⌘\` opens it, and that the width survives a reload.
 *
 * The width is the assertion worth having. A draggable divider that resets to
 * its default on every load is a control people drag once and then stop using,
 * and the persistence is easy to break in a way nothing else notices — the
 * state lives in React, the storage write is a side effect, and the two can
 * silently disagree.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { SHELL_ROUTES, renderWithProviders, stubFetch } from "../test/harness";
import { clampWidth, DEFAULT_WIDTH } from "../lib/split";

const MEMORY_ID = "11111111-1111-7111-8111-111111111111";

const MEMORY = {
  id: MEMORY_ID,
  external_key: "src/memoryos/application/worker.py",
  source_name: "self",
  title: null,
  kind: "code",
  content: "a worker claims a job and holds a lease on it",
  content_hash: "abc123def456abc123",
  normalized_hash: "def456abc123def456",
  version: 1,
  is_current: true,
  deleted_at: null,
  occurred_at: null,
  occurred_at_source: "unknown",
  ingested_at: "2026-08-20T10:00:00Z",
  chunks: [],
  versions: [],
};

function stubAll() {
  return stubFetch([
    ...SHELL_ROUTES,
    { match: "/chat", body: [] },
    /* Order matters: the harness takes the first route whose string the URL
       contains, and the detail endpoint is `/memories/{id}` — so a bare
       `/memories` listed first swallows it and returns a list where the page
       expects an object. */
    { match: `/memories/${MEMORY_ID}`, body: MEMORY },
    { match: "/memories", body: [] },
  ]);
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

/** A memory in `recents` is what `⌘\` falls back to opening. */
function seedRecent() {
  window.localStorage.setItem(
    "memo:recents",
    JSON.stringify([{ to: `/memory/${MEMORY_ID}`, label: MEMORY.external_key, kind: "memory" }]),
  );
}

describe("the split panel", () => {
  it("opens on ⌘\\ and closes on it again", async () => {
    seedRecent();
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await screen.findByLabelText("Message");

    expect(screen.queryByTestId("split-panel")).not.toBeInTheDocument();

    await userEvent.keyboard("{Meta>}\\{/Meta}");
    expect(await screen.findByTestId("split-panel")).toBeInTheDocument();

    await userEvent.keyboard("{Meta>}\\{/Meta}");
    await waitFor(() => expect(screen.queryByTestId("split-panel")).not.toBeInTheDocument());
  });

  it("closes on Esc", async () => {
    seedRecent();
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await screen.findByLabelText("Message");

    await userEvent.keyboard("{Meta>}\\{/Meta}");
    await screen.findByTestId("split-panel");

    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByTestId("split-panel")).not.toBeInTheDocument());
  });

  it("persists its width across a reload", async () => {
    seedRecent();
    stubAll();
    const first = renderWithProviders(<App />, { route: "/" });
    await screen.findByLabelText("Message");

    await userEvent.keyboard("{Meta>}\\{/Meta}");
    const divider = await screen.findByTestId("split-divider");

    // Drive it from the keyboard rather than faking a pointer drag: the
    // divider is a real `separator` and the arrow keys are the same setter the
    // drag calls, so this tests the persistence rather than jsdom's geometry.
    divider.focus();
    await userEvent.keyboard("{ArrowLeft}{ArrowLeft}");

    const widened = Number(screen.getByTestId("split-panel").dataset.splitWidth);
    expect(widened).toBe(DEFAULT_WIDTH + 4);
    expect(JSON.parse(window.localStorage.getItem("memo:split-width")!)).toBe(widened);

    // A "reload": tear the tree down and mount a fresh one against the same
    // storage, which is what a refresh actually is.
    first.unmount();
    stubAll();
    renderWithProviders(<App />, { route: "/" });
    await screen.findByLabelText("Message");
    await userEvent.keyboard("{Meta>}\\{/Meta}");

    const reopened = await screen.findByTestId("split-panel");
    expect(Number(reopened.dataset.splitWidth)).toBe(widened);
  });

  it("refuses a width that would collapse either pane", () => {
    // The stored value is user-writable and survives version changes, so the
    // clamp is the guard against `width: NaN%` as much as against a bad drag.
    expect(clampWidth(5)).toBe(22);
    expect(clampWidth(95)).toBe(68);
    expect(clampWidth(Number.NaN)).toBe(DEFAULT_WIDTH);
  });
});
