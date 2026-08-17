/**
 * The front door: one box that keeps what you type and answers what you ask.
 *
 * **The connection line is the product.** "Stored. Connects to 3 earlier
 * memories via `postgres`, `indexing`." is the sentence that separates this from
 * a notes app — it is the only visible evidence that anything was connected to
 * anything — and it is also the one piece of information that cannot be shown at
 * send time. Entity extraction is a background job and a model call, so the line
 * arrives a second or two later, and the page polls for it rather than making
 * the send wait. Until it comes back the message says `indexing…`, which is
 * true, rather than `stored and searchable`, which would not be yet.
 *
 * **Nothing here classifies.** The server decides whether a message is a
 * statement, a question or both, and this page renders the decision. A second
 * classifier in the client would eventually disagree with the first, and the
 * symptom would be a message stored by one and answered by the other.
 *
 * The classification is shown on every message and is togglable, because a
 * rules-based classifier will misread things and a misreading nobody can see is
 * a misreading nobody can correct. It is a small control on the message rather
 * than a modal: an interruption asking "did you mean to store that?" on every
 * ambiguous line would make the box slower to use than a text file.
 */

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type ChatTurn, type MessageIntent } from "../../api/client";
import { Empty, Failure } from "../../components/primitives";
import { describeConnections } from "../../lib/connections";
import { timestamp } from "../../lib/format";

/**
 * How often a message that is still indexing is asked about.
 *
 * Two seconds, and only for messages that have not finished. A page that polled
 * everything forever would be a page that makes a request per second per message
 * on screen; a page that polled once would show `indexing…` for as long as the
 * tab stayed open on any message the worker was slow to reach.
 */
const POLL_MS = 2_000;

export function ChatPage() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const bottom = useRef<HTMLDivElement>(null);

  // Search lived at `/` two milestones ago and an overview lived here until this
  // one, so a bookmarked `/?q=…` is a real link in somebody's notes. Forwarded
  // rather than swallowed: the argument for keeping search state in the URL was
  // that those links are worth something, and they do not stop being worth
  // something because the route under them changed again.
  const carried = params.get("q");
  useEffect(() => {
    if (carried) navigate(`/search?q=${encodeURIComponent(carried)}`, { replace: true });
  }, [carried, navigate]);

  const turns = useQuery({ queryKey: ["chat"], queryFn: api.chat });

  const send = useMutation({
    mutationFn: (text: string) => api.send(text),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["chat"] }),
  });

  useEffect(() => {
    // Optional-called, because jsdom has no layout and therefore no
    // `scrollIntoView`. Guarding here rather than shimming it in the test setup:
    // this is a nicety of a conversation view, and a missing scroll must never
    // be the reason a test about what was stored fails.
    bottom.current?.scrollIntoView?.({ block: "end" });
  }, [turns.data?.length, send.isPending]);

  return (
    <div className="flex max-w-(--width-reading) flex-col gap-5">
      <Preamble />

      {turns.isError ? <Failure error={turns.error} /> : null}
      {turns.data?.length === 0 && !send.isPending ? (
        <Empty title="Nothing typed yet">
          Anything you write here is kept as a memory and connected to what is
          already in the corpus through the things it talks about. Anything you
          ask is answered from all of it, with citations — or declined, if
          nothing here covers it.
        </Empty>
      ) : null}

      <ol className="flex flex-col gap-5" data-testid="messages">
        {(turns.data ?? []).map((turn) => (
          <Turn key={turn.id} turn={turn} />
        ))}
      </ol>

      {/* The optimistic half. A message must appear the instant it is sent, and
          a question takes seconds to come back — so the typed text is drawn
          immediately and the answer replaces it when it arrives. */}
      {send.isPending ? <Pending text={send.variables} /> : null}
      {send.isError ? <Failure error={send.error} /> : null}

      <div ref={bottom} />
      <Composer onSend={(text) => send.mutate(text)} busy={send.isPending} />
    </div>
  );
}

function Preamble() {
  return (
    <header className="flex flex-col gap-2">
      <h1 className="display-page">Say it here.</h1>
      <p className="prose-lead">
        Type a thought and it is kept — hashed, chunked, embedded, and linked to
        everything else that talks about the same things. Ask a question and it
        is answered from all of it, citing the passages it used. This is
        something you talk to, which can also{" "}
        <Link to="/sources" className="text-amber underline">
          read your files
        </Link>
        .
      </p>
    </header>
  );
}

/**
 * One turn: what was typed, and what became of it.
 *
 * The typed text and the answer are the same size and the same weight. Making
 * the answer larger would say it matters more than what you wrote, which is
 * backwards — the answer is derived and the message is the corpus.
 */
function Turn({ turn }: { turn: ChatTurn }) {
  return (
    <li className="flex flex-col gap-2" data-testid="message">
      <div className="flex items-baseline gap-3">
        <span className="meta-label text-muted">you</span>
        <span className="meta text-faint">{timestamp(turn.created_at)}</span>
        <IntentMark turn={turn} />
      </div>
      <p className="whitespace-pre-wrap text-ink">{turn.text}</p>
      {turn.memory_id ? <StoredLine memoryId={turn.memory_id} /> : null}
      {turn.answer !== null ? <Answer turn={turn} /> : null}
    </li>
  );
}

/**
 * How the message was read, and the toggle that argues with it.
 *
 * A button rather than a badge, because the classification is a decision the
 * system made about somebody's words and a decision you cannot answer back to is
 * one you stop trusting. The toggle is deliberately not a re-classify: it
 * explains what happened and links to the memory, which is where a wrongly
 * stored message is dealt with.
 */
function IntentMark({ turn }: { turn: ChatTurn }) {
  const [open, setOpen] = useState(false);
  const label: Record<MessageIntent, string> = {
    statement: "stored",
    question: "asked",
    both: "stored + asked",
  };
  const why: Record<MessageIntent, string> = {
    statement:
      "Read as a claim, so it was stored as a memory. Nothing was asked of the corpus.",
    question:
      "Read as a question, so it was answered and not stored. An answer is derived from what is already here; storing it would let generated text become evidence.",
    both: "Read as a claim with a question attached, so it was stored and answered.",
  };

  return (
    <span className="relative">
      <button
        type="button"
        className="meta text-faint underline decoration-dotted underline-offset-2 hover:text-amber"
        aria-expanded={open}
        onClick={() => setOpen((was) => !was)}
      >
        {label[turn.intent]}
      </button>
      {open ? (
        <span className="meta mt-1 block max-w-prose text-muted">
          {why[turn.intent]}{" "}
          {turn.memory_id ? (
            <Link to={`/memory/${turn.memory_id}`} className="text-amber underline">
              open the memory
            </Link>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}

/**
 * The connection line, once there is one.
 *
 * Three states and all three are said in words. `indexing…` while chunks and
 * vectors are being written; the connections once extraction has run; and
 * "connects to nothing yet" when it has run and found no shared entity — which
 * is a fact about the corpus, not a loading state, and a line that silently
 * disappeared in that case would leave the reader unable to tell the two apart.
 */
function StoredLine({ memoryId }: { memoryId: string }) {
  const status = useQuery({
    queryKey: ["message-status", memoryId],
    queryFn: () => api.messageStatus(memoryId),
    // Stops as soon as extraction has *run*, whatever it found. `extracted`
    // records the attempt, so a one-line thought that mentions nothing settles
    // rather than polling for an entity that is never coming.
    refetchInterval: (query) => (query.state.data?.extracted ? false : POLL_MS),
  });

  if (status.isError) return null;
  const data = status.data;

  return (
    <p className="meta text-faint" data-testid="connection-line">
      <Link to={`/memory/${memoryId}`} className="hover:text-amber hover:underline">
        stored
      </Link>
      {" · "}
      {!data ? "checking…" : describeConnections(data)}
    </p>
  );
}

/**
 * An answer, visually distinct and never softened.
 *
 * A refusal renders as a refusal: the same words the API produced, labelled, at
 * full weight. The temptation in a chat interface is to wrap "the passages do
 * not cover this" in something conversational — "Hmm, I'm not sure, but maybe…"
 * — because that is what a chat interface sounds like. That softening is how the
 * guardrail dies: a hedged refusal reads as a weak answer, and a weak answer is
 * something a reader will act on.
 */
function Answer({ turn }: { turn: ChatTurn }) {
  return (
    <div className="border-l-2 border-rule-strong pl-4" data-testid="answer">
      <div className="flex items-baseline gap-3">
        <span className="meta-label text-muted">
          {turn.refused ? "declined" : "answer"}
        </span>
        {turn.answer_model ? (
          <span className="meta text-faint">{turn.answer_model}</span>
        ) : null}
        {turn.grounded === false ? (
          <span className="meta text-deny">not fully cited</span>
        ) : null}
      </div>
      <p className="mt-1 leading-relaxed text-ink">{turn.answer}</p>

      {turn.citations.length > 0 ? (
        <ul className="mt-2 flex flex-col gap-1" data-testid="citations">
          {turn.citations.map((citation) => (
            <li key={`${citation.locator}-${citation.excerpt.slice(0, 24)}`}>
              {citation.memory_id ? (
                <Link
                  to={`/memory/${citation.memory_id}`}
                  className="meta font-mono text-ink hover:text-amber hover:underline"
                >
                  {citation.locator}
                </Link>
              ) : (
                <span className="meta font-mono text-ink">{citation.locator}</span>
              )}
              <span className="meta ml-2 text-faint">
                {citation.excerpt.slice(0, 140)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        // Said rather than omitted. An answer with no citation list looks
        // exactly like an answer whose citations have not rendered.
        <p className="meta mt-2 text-faint">
          {turn.refused
            ? "nothing was cited because nothing was used"
            : "the answer cited no passage"}
        </p>
      )}
    </div>
  );
}

/** The message that has been sent and not yet come back. */
function Pending({ text }: { text: string }) {
  return (
    <div className="flex flex-col gap-2 opacity-60" data-testid="pending">
      <span className="meta-label text-muted">you</span>
      <p className="whitespace-pre-wrap text-ink">{text}</p>
      <p className="meta text-faint">sending…</p>
    </div>
  );
}

/**
 * The box.
 *
 * Enter sends, shift-enter is a newline. A textarea rather than an input because
 * a pasted block is a legitimate message — it becomes a longer memory, chunked
 * the way any long document is — and a single-line field would silently flatten
 * it.
 */
function Composer({
  onSend,
  busy,
}: {
  onSend: (text: string) => void;
  busy: boolean;
}) {
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const typed = text.trim();
    if (!typed || busy) return;
    onSend(typed);
    setText("");
  }

  return (
    <form
      className="sticky bottom-0 flex flex-col gap-1 bg-page pt-2"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <label htmlFor="chat-input" className="meta-label">
        say something
      </label>
      <textarea
        id="chat-input"
        ref={box}
        // The shell's `/` shortcut focuses whatever carries this, so one key
        // reaches the box on whichever page has one.
        data-search-input
        rows={2}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
        placeholder="postgres full-text search is faster than I expected"
        className="field resize-y"
        aria-label="Message"
        spellCheck={false}
        autoFocus
      />
      <p className="meta text-faint">
        <span className="kbd">enter</span> sends, <span className="kbd">shift</span>
        <span className="kbd">enter</span> starts a line. A question is answered
        and not stored; anything else is kept.
      </p>
    </form>
  );
}
