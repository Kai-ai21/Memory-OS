/**
 * An answer while it is still arriving, and after it stops.
 *
 * **The retrieval line is the reason this is worth building.** M10.0 measured the
 * wait: embedding, searching and reranking is seven to eleven seconds, and
 * generation — the only part token streaming makes visible — is under two.
 * Streaming just the tokens would replace a ten-second blank screen with an
 * eight-second one. So the first thing on screen is what it is doing, in
 * milliseconds, and the counts replace it when they are known.
 *
 * **Nothing here is presented as final until `done`.** Tokens arrive before
 * verification has run — it cannot run per chunk, because a citation marker can
 * arrive split across two — so the draft is labelled `writing…` and the verdict
 * replaces the label when it lands. An interruption is labelled too, at full
 * weight, because a partial answer that looks complete is the worst of the three
 * outcomes.
 */

import { Link } from "react-router-dom";

import { count } from "../../lib/format";
import { Refusal } from "./Refusal";
import type { AnswerStream } from "./useAnswerStream";

export function StreamingAnswer({ state }: { state: AnswerStream }) {
  const withdrawn = state.done?.replacement != null;

  return (
    <div className="flex flex-col gap-4" data-testid="streaming-answer">
      <div className="flex items-baseline gap-3">
        <span className="meta-label text-ink-2">you</span>
      </div>
      {/* The reader's own words, in the display face at reading size. The
          reference sets these larger than the answer, and that is the right
          ranking: the question is what the screen is about. */}
      <p className="display text-xl whitespace-pre-wrap">{state.question}</p>

      {/* A refusal that has finished streaming is a refusal, not a quiet
          answer. Handed to the same component the stored transcript uses, so
          the two cannot drift apart. */}
      {state.done?.refused && !state.interrupted && !withdrawn ? (
        <Refusal
          footnote={
            <p className="meta text-ink-3">
              nothing was cited because nothing was used
              {state.done.model_id ? ` · ${state.done.model_id}` : ""}
            </p>
          }
        >
          {state.text}
        </Refusal>
      ) : (
      <div
        className={`relative pl-6 ${
          state.interrupted || withdrawn ? "border-l-2 border-deny" : ""
        }`}
      >
        {!state.interrupted && !withdrawn ? (
          <span
            className="absolute inset-y-0 left-0 w-0.5 bg-rule-strong"
            aria-hidden
          />
        ) : null}
        <div className="flex flex-wrap items-baseline gap-x-3">
          <span className="meta-label-on">{label(state)}</span>
          {state.done?.model_id ? (
            <span className="meta text-ink-3">{state.done.model_id}</span>
          ) : null}
          {state.done?.grounded === false && !withdrawn ? (
            <span className="meta text-deny">not fully cited</span>
          ) : null}
        </div>

        <Progress state={state} />

        {state.text ? (
          <p
            className={`prose-content mt-2 text-base ${
              state.interrupted ? "border-b border-dashed border-deny pb-1" : ""
            }`}
            data-testid="answer-text"
          >
            {state.text}
          </p>
        ) : null}

        {state.interrupted ? (
          // Said plainly, beside the text rather than instead of it. Somebody
          // reading half an answer needs to know it is half — the text itself
          // cannot tell them, because a truncated sentence reads like a short one.
          <p className="meta mt-1 text-deny" data-testid="interrupted">
            The answer stopped partway through and is incomplete: {state.interrupted}.
            Nothing here has been checked or kept.
          </p>
        ) : null}

        {withdrawn ? (
          <p className="meta mt-1 text-deny">
            The draft that was streaming cited passages that were never retrieved,
            so it has been withdrawn rather than shown.
          </p>
        ) : null}

        <Citations state={state} />
      </div>
      )}
    </div>
  );
}

function label(state: AnswerStream): string {
  if (state.interrupted) return "interrupted";
  if (state.done?.replacement) return "withdrawn";
  if (state.done?.refused) return "declined";
  if (state.done) return "answer";
  if (state.searching) return "searching";
  return "writing…";
}

/**
 * What it is doing, while it does it.
 *
 * Disappears once the first token arrives — a status line that stayed would
 * compete with the text it was covering for.
 */
function Progress({ state }: { state: AnswerStream }) {
  if (state.done || state.interrupted) {
    return state.done ? (
      <p className="meta text-ink-3">
        {count(state.retrieval?.passages ?? 0)} passages ·{" "}
        {Math.round(state.done.citation_rate * 100)}% of factual sentences cited ·{" "}
        {(state.done.total_ms / 1000).toFixed(1)}s
      </p>
    ) : null;
  }
  if (state.searching) {
    // Before any number is known. The first thing on screen, in milliseconds.
    return (
      <p className="meta text-ink-3" data-testid="retrieval-status">
        searching the corpus…
      </p>
    );
  }
  if (state.retrieval) {
    return (
      <p className="meta text-ink-3" data-testid="retrieval-status">
        found {count(state.retrieval.hits)} memories ·{" "}
        {count(state.retrieval.chunks)} matched chunks ·{" "}
        {count(state.retrieval.passages)} passages sent · reading them…
      </p>
    );
  }
  return null;
}

/**
 * Citations as they are emitted, not collected for the end.
 *
 * A citation arriving beside the sentence it supports is evidence; the same list
 * rendered after the fact is a bibliography.
 */
function Citations({ state }: { state: AnswerStream }) {
  if (state.citations.length === 0) {
    if (!state.done) return null;
    return (
      <p className="meta mt-2 text-ink-3">
        {state.done.refused
          ? "nothing was cited because nothing was used"
          : "the answer cited no passage"}
      </p>
    );
  }
  return (
    <ul className="mt-2 flex flex-col gap-1" data-testid="streaming-citations">
      {state.citations.map((citation) => (
        <li key={`${citation.locator}-${citation.excerpt.slice(0, 24)}`}>
          <Link
            to={`/memory/${citation.memory_id}`}
            className="meta font-mono text-ink hover:text-accent hover:underline"
          >
            {citation.locator}
          </Link>
          <span className="meta ml-2 text-ink-3">{citation.excerpt.slice(0, 140)}</span>
        </li>
      ))}
    </ul>
  );
}
