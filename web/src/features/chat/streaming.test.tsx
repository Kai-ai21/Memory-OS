/**
 * The three ways a streamed answer ends, and the one that must not be silent.
 *
 * The reducer is tested directly rather than through a rendered page, because
 * what it decides is precisely what an interface must not get wrong: a draft is
 * not an answer, a withdrawal replaces the draft, and an interruption leaves the
 * text on screen with a mark rather than looking finished. A component test would
 * assert on the words; this asserts on the state those words come from.
 */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { StreamingAnswer } from "./StreamingAnswer";
import type { AnswerStream } from "./useAnswerStream";
import { frames } from "../../lib/stream";

const BASE: AnswerStream = {
  question: "what did I say about postgres?",
  running: false,
  searching: false,
  retrieval: null,
  text: "",
  citations: [],
  done: null,
  interrupted: null,
};

const DONE = {
  answer: "Postgres full-text search is fast [1].",
  replacement: null,
  marked_answer: "Postgres full-text search is fast [1].",
  model_id: "openai/gpt-oss-120b",
  refused: false,
  grounded: true,
  citation_rate: 1,
  hallucinated_indices: [],
  total_ms: 2400,
};

function draw(state: Partial<AnswerStream>) {
  return render(
    <MemoryRouter>
      <StreamingAnswer state={{ ...BASE, ...state }} />
    </MemoryRouter>,
  );
}

describe("the wait", () => {
  it("says what it is doing before any token arrives", () => {
    // The reason this milestone exists. Retrieval and reranking is seven to
    // eleven seconds and generation is under two, so streaming only the tokens
    // would replace a ten-second blank screen with an eight-second one.
    draw({ running: true, searching: true });
    expect(screen.getByTestId("retrieval-status")).toHaveTextContent(/searching/i);
  });

  it("replaces the status with counts once retrieval finishes", () => {
    draw({
      running: true,
      retrieval: {
        hits: 10,
        chunks: 23,
        passages: 10,
        dropped: 0,
        retrieve_ms: 8100,
        rerank_ms: 9300,
      },
    });
    const status = screen.getByTestId("retrieval-status");
    // Numbers, not a spinner: a system working versus a system that might be
    // broken.
    expect(status).toHaveTextContent("10 memories");
    expect(status).toHaveTextContent("23 matched chunks");
  });
});

describe("how it ends", () => {
  it("labels a draft as writing, never as an answer", () => {
    draw({ running: true, text: "Postgres full-text sea" });
    // Tokens arrive before verification has run, so nothing may present them as
    // checked.
    expect(screen.getByText("writing…")).toBeInTheDocument();
    expect(screen.queryByText("answer")).not.toBeInTheDocument();
  });

  it("shows a withdrawal instead of the draft it withdrew", () => {
    draw({
      text: "This answer cited passages that were never retrieved (9), so it has been withdrawn.",
      done: { ...DONE, replacement: "…withdrawn.", grounded: false, hallucinated_indices: [9] },
    });
    expect(screen.getByText("withdrawn")).toBeInTheDocument();
    // Both halves: the replacement text is what the answer box now holds, and
    // the note beside it says the draft was withdrawn rather than finished.
    expect(screen.getByTestId("answer-text")).toHaveTextContent(/withdrawn/i);
    expect(
      screen.getByText(/The draft that was streaming cited passages/i),
    ).toBeInTheDocument();
  });

  it("marks an interrupted answer rather than leaving it looking finished", () => {
    draw({
      text: "Postgres full-text search is faster than",
      interrupted: "the connection dropped",
    });
    // The text stays — throwing it away would lose what did arrive — and the
    // mark is what a truncated sentence cannot say for itself.
    expect(screen.getByTestId("answer-text")).toHaveTextContent("faster than");
    expect(screen.getByTestId("interrupted")).toHaveTextContent(/incomplete/i);
    expect(screen.getByText("interrupted")).toBeInTheDocument();
    expect(screen.queryByText("answer")).not.toBeInTheDocument();
  });

  it("shows citations as they arrive, linked to their memories", () => {
    draw({
      running: true,
      text: "Postgres is fast [1].",
      citations: [
        {
          memory_id: "11111111-1111-7111-8111-111111111111",
          locator: "chat::2026-08-17/a.md#0",
          excerpt: "postgres full-text search is faster than I expected",
        },
      ],
    });
    // Beside the sentence they support, while it is still being written. The same
    // list rendered at the end is a bibliography.
    expect(
      screen.getByRole("link", { name: "chat::2026-08-17/a.md#0" }),
    ).toHaveAttribute("href", "/memory/11111111-1111-7111-8111-111111111111");
  });
});

describe("frame parsing", () => {
  async function collect(chunks: string[]) {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
        controller.close();
      },
    });
    const out = [];
    for await (const frame of frames(stream)) out.push(frame);
    return out;
  }

  it("reassembles a frame split across chunk boundaries", async () => {
    // A chunk boundary can fall anywhere — mid-frame, mid-field, mid-word. A
    // reader that assumed one chunk was one frame would drop text at random.
    const parsed = await collect([
      'event: token\ndata: {"te',
      'xt": "Postgres "}\n\nevent: to',
      'ken\ndata: {"text": "is fast"}\n\n',
    ]);
    expect(parsed.map((frame) => (frame.data as { text: string }).text)).toEqual([
      "Postgres ",
      "is fast",
    ]);
  });

  it("ignores heartbeats and unreadable frames without ending the stream", async () => {
    const parsed = await collect([
      ": keep-alive\n\n",
      "event: token\ndata: not json\n\n",
      'event: done\ndata: {"answer": "ok"}\n\n',
    ]);
    // One bad frame must not end a stream that is otherwise working.
    expect(parsed).toHaveLength(1);
    expect(parsed[0].event).toBe("done");
  });
});
