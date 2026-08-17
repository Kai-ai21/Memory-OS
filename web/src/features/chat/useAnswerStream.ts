/**
 * A streamed answer, as state a component can draw.
 *
 * **The draft is never presented as final.** Tokens arrive before anything has
 * checked them — verification runs on the joined text after the stream ends,
 * because a citation marker can arrive split across two chunks — so this keeps
 * `settled` false until `done`, and the interface says "writing…" rather than
 * drawing a finished answer.
 *
 * Three terminal states, and conflating any two of them is the failure this hook
 * exists to prevent:
 *
 * * `done` with no replacement — an answer, checked.
 * * `done` with a replacement — verification rejected the draft, and the text is
 *   swapped for the withdrawal. Visibly.
 * * `error` — the stream died. Whatever arrived stays on screen and is *marked*
 *   as interrupted, because a partial answer that looks complete is worse than no
 *   answer at all.
 */

import { useCallback, useRef, useState } from "react";

import { ask, type Frame } from "../../lib/stream";

export interface Retrieval {
  hits: number;
  chunks: number;
  passages: number;
  dropped: number;
  retrieve_ms: number;
  rerank_ms: number;
}

export interface StreamedCitation {
  memory_id: string;
  locator: string;
  excerpt: string;
}

export interface AnswerStream {
  question: string | null;
  /** True from the request until a terminal event. */
  running: boolean;
  /** Set once retrieval starts, before any token — this is what fills the wait. */
  searching: boolean;
  retrieval: Retrieval | null;
  /** The draft, growing. Never final until `done`. */
  text: string;
  citations: StreamedCitation[];
  /** Set on `done`. Null while streaming and after an interruption. */
  done: DonePayload | null;
  /** Set on `error`. The answer is interrupted, not finished. */
  interrupted: string | null;
}

export interface DonePayload {
  answer: string;
  /** Non-null when verification withdrew the draft. The text to draw instead. */
  replacement: string | null;
  marked_answer: string;
  model_id: string;
  refused: boolean;
  grounded: boolean;
  citation_rate: number;
  hallucinated_indices: number[];
  total_ms: number;
}

const EMPTY: AnswerStream = {
  question: null,
  running: false,
  searching: false,
  retrieval: null,
  text: "",
  citations: [],
  done: null,
  interrupted: null,
};

export function useAnswerStream() {
  const [state, setState] = useState<AnswerStream>(EMPTY);
  const abort = useRef<AbortController | null>(null);

  const run = useCallback(async (question: string, sessionId: string | null) => {
    // One answer at a time. A second question while the first is streaming
    // abandons the first, rather than interleaving two drafts into one box.
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;

    setState({ ...EMPTY, question, running: true, searching: true });

    try {
      for await (const frame of ask({ question, sessionId, signal: controller.signal })) {
        setState((was) => reduce(was, frame));
      }
      // **A stream can end without saying anything**, and that is what killing
      // the API mid-answer actually looks like: the socket closes, the body is
      // incomplete, and the reader reports a clean end-of-stream with no error to
      // catch. Measured — SIGKILL during retrieval produced zero tokens, no
      // `done`, and no exception.
      //
      // So the absence of `done` is itself the signal. Without this the state
      // would sit at `running: true` forever with no mark on it, which is
      // precisely the "left looking complete" failure this milestone forbids.
      setState((was) =>
        was.done || was.interrupted
          ? was
          : {
              ...was,
              running: false,
              searching: false,
              interrupted: "the connection to the server closed before the answer finished",
            },
      );
    } catch (error) {
      if (controller.signal.aborted) return;
      // The fetch itself failed — the API is gone, or the socket dropped before
      // a frame arrived. Same treatment as an in-band error: whatever is on
      // screen stays, marked.
      setState((was) => ({
        ...was,
        running: false,
        searching: false,
        interrupted: error instanceof Error ? error.message : String(error),
      }));
    }
  }, []);

  const reset = useCallback(() => {
    abort.current?.abort();
    setState(EMPTY);
  }, []);

  return { state, run, reset };
}

function reduce(state: AnswerStream, frame: Frame): AnswerStream {
  const data = frame.data as Record<string, never>;
  switch (frame.event) {
    case "retrieval_started":
      return { ...state, searching: true };
    case "retrieval_done":
      return {
        ...state,
        searching: false,
        retrieval: frame.data as unknown as Retrieval,
      };
    case "token":
      return { ...state, text: state.text + String(data.text ?? "") };
    case "citation":
      // Appended as they are emitted rather than collected for the end, which is
      // the point: a citation arriving beside the sentence it supports is
      // evidence, and the same list rendered after the fact is a bibliography.
      return {
        ...state,
        citations: [...state.citations, frame.data as unknown as StreamedCitation],
      };
    case "done": {
      const done = frame.data as unknown as DonePayload;
      return {
        ...state,
        running: false,
        searching: false,
        done,
        // The replacement wins the text. Keeping the draft here and rendering the
        // withdrawal beside it would leave the withdrawn claim on screen, which
        // is exactly what withdrawing it is for.
        text: done.replacement ?? state.text,
      };
    }
    case "error":
      return {
        ...state,
        running: false,
        searching: false,
        // Whatever the server managed to send, so an interface that lost the
        // tokens still has something to mark.
        text: state.text || String(data.partial ?? ""),
        interrupted: String(data.message ?? "the answer stream stopped"),
      };
    default:
      return state;
  }
}
