/**
 * The four-signal panel: that it draws four, and that it does not invent two.
 *
 * The reference shows SEMANTIC, KEYWORD, RECENCY and GRAPH all carrying a
 * percentage. Against this backend only the first two ever can — recency,
 * importance and graph all have a fusion weight of 0.0 in `config.py`, each set
 * by a measurement written above it — so the interesting assertion is not that
 * four rows appear. It is that the two with nothing behind them say so instead
 * of showing 0%, which would assert that the signal ran and found nothing.
 */

import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ExplanationPanel } from "./Explanation";

/** Shaped exactly like a live response: two contributions, no graph path. */
const LIVE = {
  final_rank: 1,
  fused_score: 0.030282,
  rerank_score: 5.719,
  graph_path: null,
  why: "Ranked 1st: strong semantic match (rank 1), weak keyword match (rank 12), unchanged by reranking.",
  contributions: [
    { name: "semantic", rank: 1, score: 0.8185, weight: 1.0, contribution: 0.016393, share: 0.5414 },
    { name: "keyword", rank: 12, score: 0.026, weight: 1.0, contribution: 0.013889, share: 0.4586 },
  ],
};

function row(signal: string) {
  return screen
    .getAllByTestId("contribution")
    .find((element) => element.dataset.signal === signal)!;
}

async function open(explanation: unknown) {
  render(
    <ExplanationPanel
      explanation={explanation as never}
      citations={[]}
      code={false}
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: /why this ranked/i }));
}

describe("the explanation panel", () => {
  it("draws all four signals, in the reference's order", async () => {
    await open(LIVE);

    expect(
      screen.getAllByTestId("contribution").map((element) => element.dataset.signal),
    ).toEqual(["semantic", "keyword", "recency", "graph"]);
  });

  it("shows the live percentages for the signals that contributed", async () => {
    await open(LIVE);

    // Rounded from the shares the API sent, never recomputed from the scores.
    // The API assembles these from the same 1/(k+rank) terms that produced the
    // fused score, precisely so the UI cannot disagree with the ranker.
    expect(within(row("semantic")).getByText("54%")).toBeInTheDocument();
    expect(within(row("keyword")).getByText("46%")).toBeInTheDocument();

    // And each names where its own retriever placed the result.
    expect(within(row("semantic")).getByText("#1")).toBeInTheDocument();
    expect(within(row("keyword")).getByText("#12")).toBeInTheDocument();
  });

  it("says a missing signal contributed nothing rather than showing it at 0%", async () => {
    await open(LIVE);

    // The distinction this whole panel turns on. A row reading 0% claims the
    // signal ran and found nothing; these two did not run at all.
    expect(within(row("recency")).getByText(/no contribution/i)).toBeInTheDocument();
    expect(within(row("graph")).getByText(/no contribution/i)).toBeInTheDocument();
    expect(within(row("recency")).queryByText("0%")).not.toBeInTheDocument();
  });

  it("does not claim to know why a signal is absent", async () => {
    // `build_explanation` drops a ranking either because its weight is zero or
    // because it never returned this chunk, and the serialised form keeps no
    // trace of which. Picking one would be a guess presented as a fact.
    await open(LIVE);

    const note = screen.getByTestId("absent-note");
    expect(note).toHaveTextContent(/fusion weight of zero/i);
    expect(note).toHaveTextContent(/did not return this result/i);
    expect(note).toHaveTextContent(/does not distinguish/i);
  });

  it("appends a live signal the reference has no row for", async () => {
    // `importance` has a weight of 0.0 today, so it never appears — but if it
    // is switched on, a panel built to the mockup's four rows would silently
    // drop it and misreport the ranking it exists to explain.
    await open({
      ...LIVE,
      contributions: [
        ...LIVE.contributions,
        { name: "importance", rank: 4, score: 0.4, weight: 0.1, contribution: 0.0015, share: 0.05 },
      ],
    });

    expect(
      screen.getAllByTestId("contribution").map((element) => element.dataset.signal),
    ).toEqual(["semantic", "keyword", "recency", "graph", "importance"]);
    expect(within(row("importance")).getByText("5%")).toBeInTheDocument();
  });

  it("names the entity route when the graph is what introduced the result", async () => {
    await open({
      ...LIVE,
      graph_path: "job queue -> SKIP LOCKED",
      contributions: [
        { name: "graph", rank: 1, score: 0.5, weight: 0.5, contribution: 0.008, share: 1.0 },
      ],
    });

    // The one contribution a reader cannot check against the text in front of
    // them, so the route is stated rather than left to a percentage.
    expect(screen.getByTestId("graph-path")).toHaveTextContent(/job queue -> SKIP LOCKED/);
    expect(within(row("graph")).getByText("100%")).toBeInTheDocument();
  });
});
