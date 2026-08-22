/**
 * The preview: that it waits, and that it leaves immediately.
 *
 * Both halves are the feature. A preview with no delay flickers across a list
 * as the cursor crosses it on the way somewhere else; one that lingers after
 * mouse-out covers whatever you moved to. Fake timers rather than waiting 400ms
 * of real time — the delay is the thing under test, so it has to be controlled
 * rather than tolerated.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { MemoryPreview, PREVIEW_DELAY_MS } from "./MemoryPreview";
import { renderWithProviders, stubFetch } from "../test/harness";

const MEMORY_ID = "11111111-1111-7111-8111-111111111111";

const MEMORY = {
  id: MEMORY_ID,
  external_key: "src/memoryos/application/worker.py",
  source_name: "self",
  title: "how leases work",
  kind: "code",
  content: "a worker claims a job and holds a lease on it",
  content_hash: "abc",
  normalized_hash: "def",
  version: 1,
  is_current: true,
  deleted_at: null,
  occurred_at: null,
  occurred_at_source: "unknown",
  ingested_at: "2026-08-20T10:00:00Z",
  chunks: [],
  versions: [],
};

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function renderPreview() {
  stubFetch([{ match: `/memories/${MEMORY_ID}`, body: MEMORY }]);
  return renderWithProviders(
    <MemoryPreview memoryId={MEMORY_ID}>
      <a href="/memory/x">worker.py</a>
    </MemoryPreview>,
  );
}

describe("the hover preview", () => {
  it("appears only after the delay, and not before it", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPreview();

    await user.hover(screen.getByTestId("memory-preview-anchor"));

    // Just short of the threshold: still nothing. This is the assertion that
    // catches a delay accidentally set to zero, which every other test passes.
    await act(async () => {
      vi.advanceTimersByTime(PREVIEW_DELAY_MS - 50);
    });
    expect(screen.queryByTestId("memory-preview")).not.toBeInTheDocument();

    await act(async () => {
      vi.advanceTimersByTime(60);
    });
    expect(screen.getByTestId("memory-preview")).toBeInTheDocument();
  });

  it("dismisses on mouse-out with no delay at all", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPreview();

    const anchor = screen.getByTestId("memory-preview-anchor");
    await user.hover(anchor);
    await act(async () => {
      vi.advanceTimersByTime(PREVIEW_DELAY_MS + 20);
    });
    expect(screen.getByTestId("memory-preview")).toBeInTheDocument();

    await user.unhover(anchor);
    // Synchronous — no timer is advanced between the unhover and this check.
    expect(screen.queryByTestId("memory-preview")).not.toBeInTheDocument();
  });

  it("never opens for a cursor that merely passed over", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderPreview();

    const anchor = screen.getByTestId("memory-preview-anchor");
    await user.hover(anchor);
    await act(async () => {
      vi.advanceTimersByTime(120);
    });
    await user.unhover(anchor);

    // Past the full delay from the original hover: the timer must have been
    // cancelled, not merely hidden behind a flag.
    await act(async () => {
      vi.advanceTimersByTime(PREVIEW_DELAY_MS);
    });
    expect(screen.queryByTestId("memory-preview")).not.toBeInTheDocument();
  });
});
