/**
 * Three things about the timeline that would be quietly wrong.
 *
 * Quietly is the operative word. A chart with the wrong number of bars still
 * looks like a chart; a click that fetches the wrong range still returns a list
 * of plausible memories; a date rendered without its provenance still reads as a
 * date. None of these fail loudly, and all three change what a reader concludes
 * from the screen.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { TimelinePage } from "./TimelinePage";
import { renderWithProviders, stubFetch } from "../../test/harness";

afterEach(() => vi.unstubAllGlobals());

const SOURCES = [
  {
    id: "33333333-3333-7333-8333-333333333333",
    kind: "filesystem",
    name: "self",
    config: {},
    last_sync_at: null,
    last_full_sync_at: null,
    created_at: "2026-06-01T10:00:00Z",
    memories: 6,
    chunks: 40,
  },
];

/**
 * Four monthly buckets with an empty one in the middle.
 *
 * The hole is the point: it is what the hatch and the gap marker are for, and a
 * fixture with activity in every period could not tell a chart that draws
 * absence from one that skips it.
 */
function timelineResponse() {
  return {
    start: "2026-01-01T00:00:00Z",
    end: "2026-05-01T00:00:00Z",
    period: "month",
    total: 9,
    buckets: [
      {
        start: "2026-01-01T00:00:00Z",
        end: "2026-02-01T00:00:00Z",
        count: 4,
        by_kind: { code: 3, note: 1 },
      },
      { start: "2026-02-01T00:00:00Z", end: "2026-03-01T00:00:00Z", count: 0, by_kind: {} },
      {
        start: "2026-03-01T00:00:00Z",
        end: "2026-04-01T00:00:00Z",
        count: 2,
        by_kind: { code: 2 },
      },
      {
        start: "2026-04-01T00:00:00Z",
        end: "2026-05-01T00:00:00Z",
        count: 3,
        by_kind: { note: 3 },
      },
    ],
    provenance: [
      {
        provenance: "filesystem",
        count: 8,
        earliest: "2026-01-04T09:00:00Z",
        latest: "2026-04-20T18:00:00Z",
      },
      { provenance: "declared", count: 1, earliest: null, latest: null },
      { provenance: "unknown", count: 0, earliest: null, latest: null },
    ],
  };
}

function memoriesAtResponse() {
  return {
    start: "2026-01-01T00:00:00Z",
    end: "2026-02-01T00:00:00Z",
    total: 2,
    memories: [
      {
        id: "11111111-1111-7111-8111-111111111111",
        external_key: "src/worker.py",
        source_name: "self",
        kind: "code",
        title: null,
        // The low-confidence case: an mtime. Nobody said this is when the work
        // happened; it is when the bytes were last written to a disk.
        occurred_at: "2026-01-04T09:00:00Z",
        occurred_at_source: "filesystem",
        ingested_at: "2026-05-02T12:00:00Z",
      },
      {
        id: "22222222-2222-7222-8222-222222222222",
        external_key: "inbox/kickoff.eml",
        source_name: "self",
        kind: "note",
        title: null,
        // The high-confidence case: the source stated the date itself.
        occurred_at: "2026-01-11T14:30:00Z",
        occurred_at_source: "declared",
        ingested_at: "2026-05-02T12:00:00Z",
      },
    ],
  };
}

const GAPS = [
  {
    start: "2026-01-20T00:00:00Z",
    end: "2026-03-02T00:00:00Z",
    days: 41,
    source_name: "self",
    before: {
      id: "11111111-1111-7111-8111-111111111111",
      external_key: "src/worker.py",
      kind: "code",
      occurred_at: "2026-01-20T00:00:00Z",
      occurred_at_source: "filesystem",
    },
    after: {
      id: "22222222-2222-7222-8222-222222222222",
      external_key: "src/queue.py",
      kind: "code",
      occurred_at: "2026-03-02T00:00:00Z",
      occurred_at_source: "filesystem",
    },
  },
];

function stubTimeline() {
  return stubFetch([
    { match: "/sources", body: SOURCES },
    { match: "/timeline", body: timelineResponse() },
    { match: "/gaps", body: GAPS },
    { match: "/memories/at", body: memoriesAtResponse() },
  ]);
}

describe("bucketing", () => {
  it("draws one bar per period, including the empty ones", async () => {
    stubTimeline();
    renderWithProviders(<TimelinePage />);

    await waitFor(() => expect(screen.getByTestId("activity-chart")).toBeInTheDocument());

    // Four buckets in the response, four bars — the empty February included.
    // A chart built from only the periods that have rows would draw three, and
    // three bars of a four-month window is a chart of a corpus that never
    // stopped.
    const bars = screen.getAllByTestId("bar");
    expect(bars).toHaveLength(4);
    expect(bars.map((bar) => bar.dataset.count)).toEqual(["4", "0", "2", "3"]);

    // And the empty one is *drawn*, not left blank. This is the whole argument
    // of M4.0's gap detection reaching the surface: absence rendered as absence
    // is indistinguishable from the edge of the data.
    expect(screen.getAllByTestId("empty-period")).toHaveLength(1);
    expect(within(bars[1]).getByTestId("empty-period")).toBeInTheDocument();

    // The gap itself is an object with boundaries, in its own lane.
    expect(screen.getAllByTestId("gap-marker")).toHaveLength(1);
  });
});

describe("selecting a period", () => {
  it("fetches exactly the clicked period's range, not a guessed width", async () => {
    const calls = stubTimeline();
    renderWithProviders(<TimelinePage />);
    await waitFor(() => expect(screen.getByTestId("activity-chart")).toBeInTheDocument());

    await userEvent.click(screen.getAllByTestId("bar")[0]);

    await waitFor(() =>
      expect(calls.some((call) => call.url.includes("/memories/at"))).toBe(true),
    );
    const request = calls.find((call) => call.url.includes("/memories/at"));
    const params = new URL(request!.url).searchParams;

    // January's start, and January's *own* length. A fixed 30 would spill into
    // February and a fixed 31 would be wrong for the next bar the reader
    // clicks — the bucket carries its boundaries and this uses them.
    expect(params.get("date")).toBe("2026-01-01T00:00:00Z");
    expect(params.get("window_days")).toBe("31");

    await waitFor(() =>
      expect(screen.getByTestId("period-memories")).toHaveTextContent("src/worker.py"),
    );
  });
});

describe("provenance", () => {
  it("marks a low-confidence date and leaves a stated one unmarked", async () => {
    stubTimeline();
    renderWithProviders(<TimelinePage />);
    await waitFor(() => expect(screen.getByTestId("activity-chart")).toBeInTheDocument());
    await userEvent.click(screen.getAllByTestId("bar")[0]);
    await waitFor(() =>
      expect(screen.getByTestId("period-memories")).toHaveTextContent("src/worker.py"),
    );

    const rows = within(screen.getByTestId("period-memories")).getAllByRole("listitem");

    // The mtime carries the approximation mark and says why on hover. An
    // interface that rendered this identically to the line below it would be
    // asserting the two dates are the same kind of claim.
    //
    // Scoped to the date stamp by test id rather than found by its title.
    // M9.1 added the reference's provenance chip to the same row, which also
    // carries the provenance and the same explanatory title — so the title
    // alone now matches two elements. Both assertions below are unchanged.
    const inferred = within(rows[0]).getByTestId("date-stamp");
    expect(inferred).toHaveAttribute("data-provenance", "filesystem");
    expect(inferred).toHaveTextContent("~");

    // The declared one is unmarked, and the absence of the mark is the signal:
    // marking every date would make the mark invisible.
    const stated = within(rows[1]).getByTestId("date-stamp");
    expect(stated).toHaveAttribute("data-provenance", "declared");
    expect(stated).not.toHaveTextContent("~");

    // And the new chip states the same claim in a word, which is the reference's
    // treatment: a stated date is chipped in cyan, an inferred one is not.
    expect(within(rows[0]).getByTestId("provenance-chip")).toHaveTextContent("filesystem");
    expect(within(rows[1]).getByTestId("provenance-chip")).toHaveTextContent("declared");
  });
});
