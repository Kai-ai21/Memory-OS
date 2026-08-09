/**
 * What a judgement actually sends.
 *
 * The payload is the milestone's output, so it is asserted field by field. Two
 * things in particular: the item is identified by `(source_name, external_key)`
 * rather than by a memory id that a rebuild would invalidate, and a `missing`
 * verdict carries no rank — the API rejects one that does, and the point of the
 * verdict is that the item was never ranked.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { JudgementButtons, type JudgementTarget } from "./JudgementButtons";
import { renderWithProviders, stubFetch } from "../../test/harness";

const TARGET: JudgementTarget = {
  queryText: "how does the job queue claim work",
  sourceName: "self",
  externalKey: "src/memoryos/application/worker.py",
  memoryId: "11111111-1111-7111-8111-111111111111",
  chunkId: "22222222-2222-7222-8222-222222222222",
  rank: 2,
  score: 0.7911,
  filters: { k: 10, sources: ["self"], kind: null, exact: false },
};

afterEach(() => vi.unstubAllGlobals());

describe("judgement submission", () => {
  it("posts the full payload to /judgements", async () => {
    const calls = stubFetch([{ match: "/judgements", body: { id: "44444444" } }]);
    renderWithProviders(<JudgementButtons target={TARGET} />);

    await userEvent.click(screen.getByRole("button", { name: "relevant" }));

    const call = calls.find((candidate) => candidate.url.includes("/judgements"));
    expect(call?.method).toBe("POST");
    expect(call?.body).toEqual({
      query_text: "how does the job queue claim work",
      // The durable identity. A memory id would be stale after a replay.
      source_name: "self",
      external_key: "src/memoryos/application/worker.py",
      verdict: "relevant",
      // Snapshots of what the system said at the moment of judgement.
      memory_id: "11111111-1111-7111-8111-111111111111",
      chunk_id: "22222222-2222-7222-8222-222222222222",
      rank_at_judgement: 2,
      score_at_judgement: 0.7911,
      filters: { k: 10, sources: ["self"], kind: null, exact: false },
    });
  });

  it("sends not_relevant when that button is used", async () => {
    const calls = stubFetch([{ match: "/judgements", body: { id: "x" } }]);
    renderWithProviders(<JudgementButtons target={TARGET} />);

    await userEvent.click(screen.getByRole("button", { name: "not relevant" }));

    expect(calls.at(-1)?.body).toMatchObject({ verdict: "not_relevant" });
  });

  it("strips rank and score from a missing verdict", async () => {
    // The API refuses a rank on `missing`, because the item was not in the
    // ranking. This is the client agreeing rather than the client deciding.
    const calls = stubFetch([{ match: "/judgements", body: { id: "x" } }]);
    renderWithProviders(
      <JudgementButtons target={TARGET} verdicts={["missing"]} />,
    );

    await userEvent.click(screen.getByRole("button", { name: "missing" }));

    expect(calls.at(-1)?.body).toMatchObject({
      verdict: "missing",
      rank_at_judgement: null,
      score_at_judgement: null,
    });
  });

  it("shows the recorded verdict as active, so re-judging reads as a change", () => {
    stubFetch([{ match: "/judgements", body: { id: "x" } }]);
    renderWithProviders(<JudgementButtons target={TARGET} current="relevant" />);

    expect(screen.getByRole("button", { name: "relevant" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "not relevant" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("reports a failure instead of appearing to have saved", async () => {
    stubFetch([{ match: "/judgements", status: 500, body: { detail: "boom" } }]);
    renderWithProviders(<JudgementButtons target={TARGET} />);

    await userEvent.click(screen.getByRole("button", { name: "relevant" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/not saved/i);
  });

  it("tells the caller which verdict was recorded", async () => {
    stubFetch([{ match: "/judgements", body: { id: "x" } }]);
    const onRecorded = vi.fn();
    renderWithProviders(<JudgementButtons target={TARGET} onRecorded={onRecorded} />);

    await userEvent.click(screen.getByRole("button", { name: "relevant" }));

    expect(onRecorded).toHaveBeenCalledWith("relevant");
  });
});
