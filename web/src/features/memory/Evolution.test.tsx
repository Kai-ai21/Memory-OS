/**
 * What the evolution view must not get wrong.
 *
 * The two that matter: an empty diff has to read as "nothing changed" rather
 * than as a diff that failed to render, and a summary the backend flagged as
 * ungrounded has to arrive on screen flagged. Both failures look like a working
 * feature — one shows an empty box, the other shows fluent prose.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Evolution } from "./Evolution";
import { renderWithProviders, stubFetch } from "../../test/harness";

afterEach(() => vi.unstubAllGlobals());

const MEMORY_ID = "11111111-1111-7111-8111-111111111111";
const V1 = "aaaaaaaa-1111-7111-8111-111111111111";
const V2 = "bbbbbbbb-2222-7222-8222-222222222222";

function version(id: string, number: number, overrides: Record<string, unknown> = {}) {
  return {
    id,
    version: number,
    is_current: number === 2,
    kind: "code",
    title: null,
    content_hash: "a".repeat(64),
    normalized_hash: "b".repeat(64),
    occurred_at: "2026-08-10T15:38:55Z",
    occurred_at_source: "filesystem",
    ingested_at: "2026-08-10T16:15:01Z",
    deleted_at: null,
    characters: 5624,
    chunks: number === 2 ? 5 : 0,
    holds_chunks: number === 2,
    chunker_versions: number === 2 ? ["structural-v3"] : [],
    adopted: number === 1 ? null : false,
    text_changed: number === 1 ? null : true,
    bytes_changed: number === 1 ? null : true,
    change: number === 1 ? "first version" : "rechunked",
    ...overrides,
  };
}

function evolutionResponse(diff: Record<string, unknown>) {
  return {
    memory_id: MEMORY_ID,
    source_id: "33333333-3333-7333-8333-333333333333",
    source_name: "self",
    external_key: "src/memoryos/config.py",
    versions: [version(V1, 1), version(V2, 2)],
    diffs: [
      {
        from_id: V1,
        to_id: V2,
        from_version: 1,
        to_version: 2,
        added_chars: 652,
        removed_chars: 0,
        chunk_delta: null,
        is_empty: false,
        span_count: 2,
        spans: [
          {
            kind: "added",
            a_start: 300,
            a_end: 300,
            b_start: 300,
            b_end: 340,
            a_text: "",
            b_text: "gemini_api_key: str | None = None\n",
            truncated: false,
          },
        ],
        affected_chunks: [
          {
            id: "44444444-4444-7444-8444-444444444444",
            ordinal: 3,
            char_start: 200,
            char_end: 900,
            definition: "Settings",
            spans: 1,
          },
        ],
        unified: "--- a\n+++ b\n",
        summary: null,
        ...diff,
      },
    ],
  };
}

describe("rendering a diff", () => {
  it("shows the added text marked, and the chunks it landed in", async () => {
    stubFetch([{ match: "/evolution", body: evolutionResponse({}) }]);
    renderWithProviders(<Evolution memoryId={MEMORY_ID} kind="code" />);

    await waitFor(() => expect(screen.getByTestId("span-table")).toBeInTheDocument());

    const added = screen.getByTestId("added");
    expect(added).toHaveTextContent("gemini_api_key");
    expect(added).toHaveClass("mark");

    // The left side of a pure insertion is empty, and says so rather than
    // rendering a blank cell that reads as a loading failure.
    expect(screen.getByTestId("diff-row")).toHaveTextContent(/nothing here before/i);

    expect(screen.getByTestId("affected-chunks")).toHaveTextContent("#3 (Settings)");
    // The chunk delta is unavailable rather than zero, because the older version
    // no longer holds its chunks.
    expect(screen.getByTestId("diff-panel")).toHaveTextContent(/chunk delta\s*n\/a/i);
  });

  it("says an empty diff is an unchanged text, not a failed render", async () => {
    stubFetch([
      {
        match: "/evolution",
        body: evolutionResponse({
          is_empty: true,
          spans: [],
          span_count: 0,
          added_chars: 0,
          affected_chunks: [],
        }),
      },
    ]);
    renderWithProviders(<Evolution memoryId={MEMORY_ID} kind="code" />);

    await waitFor(() => expect(screen.getByTestId("empty-diff")).toBeInTheDocument());
    expect(screen.getByTestId("empty-diff")).toHaveTextContent(/normalized text is identical/i);
    expect(screen.queryByTestId("span-table")).not.toBeInTheDocument();
  });
});

describe("summaries", () => {
  it("only asks for them when told to, and marks an ungrounded one", async () => {
    const calls = stubFetch([
      {
        match: "/evolution",
        body: evolutionResponse({
          summary: {
            text: "Adds a retry_budget field to Settings.",
            model_id: "llama-3.3-70b-versatile",
            summarizer_version: "change-v1",
            grounded: false,
            unsupported: ["retry_budget"],
            context_only: ["Settings"],
            trivial: false,
            cached: false,
          },
        }),
      },
    ]);
    renderWithProviders(<Evolution memoryId={MEMORY_ID} kind="code" />);
    await waitFor(() => expect(screen.getByTestId("span-table")).toBeInTheDocument());

    // Nothing has asked for a summary yet, so the first request must not have.
    expect(calls.every((call) => !call.url.includes("summarize=true"))).toBe(true);

    await userEvent.click(screen.getByRole("button", { name: /describe changes/i }));

    await waitFor(() =>
      expect(calls.some((call) => call.url.includes("summarize=true"))).toBe(true),
    );
    const panel = await screen.findByTestId("change-summary");
    // The text is shown — never suppressed — and so is the verdict on it.
    expect(panel).toHaveTextContent("Adds a retry_budget field to Settings.");
    expect(panel).toHaveTextContent(/ungrounded/i);
    expect(within(panel).getByText(/named but not in the diff/i)).toHaveTextContent(
      "retry_budget",
    );
    expect(within(panel).getByText(/from context, not from the change/i)).toHaveTextContent(
      "Settings",
    );
  });
});
