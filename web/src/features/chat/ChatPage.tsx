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
 *
 * **A session is not the memory, and this page has to make that legible.** The
 * rail on the left is navigation; the `stored` link under every message goes to
 * its memory detail view, where it sits beside everything it connects to
 * regardless of which conversation it was typed in. Those are two different
 * groupings of the same message and the page shows both, because a reader who
 * only ever saw the conversation would conclude that is where meaning lives.
 *
 * The selected session lives in the URL as `?session=`, so a conversation is a
 * link somebody can paste. Search within it lives in `?q=` for the same reason —
 * and it is a substring filter over rows already on screen, not corpus search,
 * which has its own page and answers a different question.
 */

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type ChatMessage, type MessageIntent } from "../../api/client";
import { Empty, Failure } from "../../components/primitives";
import { describeConnections } from "../../lib/connections";
import { fileSize, timestamp } from "../../lib/format";
import { AttachmentList } from "./Attachments";
import { SessionRail } from "./SessionRail";
import { StreamingAnswer } from "./StreamingAnswer";
import { useAnswerStream } from "./useAnswerStream";
import { useLiveEvents } from "./useLiveEvents";

/**
 * How often a message that is still indexing is asked about.
 *
 * Two seconds, and only for messages that have not finished. A page that polled
 * everything forever would be a page that makes a request per second per message
 * on screen; a page that polled once would show `indexing…` for as long as the
 * tab stayed open on any message the worker was slow to reach.
 */
const POLL_MS = 2_000;

/**
 * `?session=new`: the button was pressed and no conversation exists yet.
 *
 * A real value in the URL rather than the absence of one, so that "I want a fresh
 * conversation" and "I have not said which conversation" stay distinguishable —
 * and so the state is linkable like every other piece of view state in this
 * application. Cannot collide with a session id, which is always a uuid.
 */
const NEW_SESSION = "new";

export function ChatPage() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const bottom = useRef<HTMLDivElement>(null);

  // Search lived at `/` two milestones ago and an overview lived here until the
  // last one, so a bookmarked `/?q=…` is a real link in somebody's notes — but
  // `?q=` now means "filter this conversation", so only a `?q=` with no session
  // is treated as the old search link. Forwarded rather than swallowed: the
  // argument for keeping search state in the URL was that those links are worth
  // something.
  const carried = params.get("q");
  const selected = params.get("session");
  useEffect(() => {
    if (carried && !selected) {
      navigate(`/search?q=${encodeURIComponent(carried)}`, { replace: true });
    }
  }, [carried, selected, navigate]);

  const sessions = useQuery({
    queryKey: ["chat-sessions", false],
    queryFn: () => api.chatSessions(false),
  });

  // Three states, not two, and the third is why `session=new` is in the URL as a
  // literal rather than being represented by its absence.
  //
  // *Absent* means "no opinion": draw the most recent conversation, which is what
  // somebody opening a fresh tab is in the middle of. *A uuid* means a
  // conversation they clicked. *`new`* means they pressed the button and want an
  // empty one — and collapsing that into absence would make the button snap
  // straight back to the latest conversation, which is what it did before this
  // distinction existed.
  const starting = selected === NEW_SESSION;
  const active = starting ? null : (selected ?? sessions.data?.[0]?.id ?? null);
  const filter = active && selected && !starting ? (carried ?? "") : "";

  const messages = useQuery({
    queryKey: ["chat-messages", active, filter],
    queryFn: () => api.chatMessages(active!, filter || undefined),
    enabled: active !== null,
  });

  const [pending, setPending] = useState<File[]>([]);
  // Connection lines arrive here, pushed, without a refresh. One subscription for
  // the page rather than one per message: the server says which memory moved and
  // the client re-reads that one status.
  useLiveEvents();
  const answer = useAnswerStream();

  const send = useMutation({
    mutationFn: (text: string) =>
      // One box, two endpoints, and the client decides by whether files are
      // queued rather than by asking. A note typed beside a file is the *same*
      // act as typing a note — `attach` classifies it with the same rules — so a
      // separate "upload" flow would have been a second front door.
      pending.length > 0
        ? api.attach(pending, { note: text, sessionId: active, newSession: starting })
        // `defer_answer`: the server stores and classifies, and this page streams
        // the answer for anything it calls a question. One classifier, on the
        // server, and the several-second model call moved off the send.
        : api.send(text, active, starting, true),
    onSuccess: (exchange) => {
      // Pinned to the session the server actually used, which is not always the
      // one that was asked for: with no session named, the thirty-minute rule may
      // have opened a new one, and a page that kept showing the old one would draw
      // a message into a conversation it is not in.
      setParams(
        (was) => {
          const next = new URLSearchParams(was);
          next.set("session", exchange.session_id);
          next.delete("q");
          return next;
        },
        { replace: true },
      );
      setPending([]);
      void client.invalidateQueries({ queryKey: ["chat-sessions"] });
      void client.invalidateQueries({ queryKey: ["chat-messages"] });

      // The server classified. A question streams its answer from here; a
      // statement is already stored and there is nothing more to do.
      const asked = exchange.messages[0];
      if (asked.intent === "question" || asked.intent === "both") {
        void answer.run(asked.content, exchange.session_id).then(() => {
          // The server wrote the assistant turn when the stream ended, so the
          // transcript now has a row the streamed view is a duplicate of.
          // Refetching and clearing swaps one for the other; leaving both would
          // draw the same answer twice.
          void client.invalidateQueries({ queryKey: ["chat-messages"] });
          answer.reset();
        });
      }
    },
  });

  useEffect(() => {
    // Optional-called, because jsdom has no layout and therefore no
    // `scrollIntoView`. Guarding here rather than shimming it in the test setup:
    // this is a nicety of a conversation view, and a missing scroll must never
    // be the reason a test about what was stored fails.
    bottom.current?.scrollIntoView?.({ block: "end" });
  }, [messages.data?.length, send.isPending]);

  function open(id: string) {
    setParams((was) => {
      const next = new URLSearchParams(was);
      next.set("session", id);
      next.delete("q");
      return next;
    });
  }

  function start() {
    setParams((was) => {
      const next = new URLSearchParams(was);
      next.set("session", NEW_SESSION);
      next.delete("q");
      return next;
    });
    // Nothing is created here. A conversation begins on its first message, so an
    // empty session cannot exist — which is what keeps the rail free of rows
    // nobody typed into.
    send.reset();
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[16rem_minmax(0,1fr)]">
      <SessionRail current={active} onSelect={open} onNew={start} />

      <div className="flex min-w-0 max-w-(--width-reading) flex-col gap-5">
        <Preamble />
        {active ? <Filter sessionId={active} /> : null}

        {messages.isError ? <Failure error={messages.error} /> : null}
        {active === null && !send.isPending ? (
          <Empty title="Nothing typed yet">
            Anything you write here is kept as a memory and connected to what is
            already in the corpus through the things it talks about. Anything you
            ask is answered from all of it, with citations — or declined, if
            nothing here covers it.
          </Empty>
        ) : null}
        {active !== null && messages.data?.length === 0 && filter ? (
          <Empty title="Nothing in this conversation matches">
            This is a substring filter over the messages in front of you.{" "}
            <Link to={`/search?q=${encodeURIComponent(filter)}`} className="text-amber underline">
              Search the whole corpus
            </Link>{" "}
            instead — that one is semantic and reaches everything, not just this
            conversation.
          </Empty>
        ) : null}

        <ol className="flex flex-col gap-5" data-testid="messages">
          {(messages.data ?? []).map((message) => (
            <Turn key={message.id} message={message} />
          ))}
        </ol>

        {/* The optimistic half. A message must appear the instant it is sent, and
            a question takes seconds to come back — so the typed text is drawn
            immediately and the answer replaces it when it arrives. */}
        {send.isPending ? <Pending text={send.variables} /> : null}
        {send.isError ? <Failure error={send.error} /> : null}
        {answer.state.question !== null ? (
          <StreamingAnswer state={answer.state} />
        ) : null}

        <div ref={bottom} />
        <Composer
          onSend={(text) => {
            answer.reset();
            send.mutate(text);
          }}
          busy={send.isPending}
          queued={pending}
          onQueue={(files) => setPending((was) => [...was, ...files])}
          onRemove={(index) =>
            setPending((was) => was.filter((_, at) => at !== index))
          }
        />
      </div>
    </div>
  );
}

/**
 * Search within this conversation.
 *
 * **Distinct from corpus search, and the copy says so rather than leaving it to
 * be discovered.** This is a substring filter over rows the reader can already
 * see — a scroll replacement — and running it through the embedder would return
 * semantic neighbours from a conversation they can see all of. The link to
 * `/search` is right there for the question this one does not answer.
 */
function Filter({ sessionId }: { sessionId: string }) {
  const [params, setParams] = useSearchParams();
  const value = params.get("q") ?? "";

  return (
    <div className="flex items-baseline gap-3 border-b border-rule pb-1">
      <label htmlFor="session-filter" className="meta-label text-muted">
        in this conversation
      </label>
      <input
        id="session-filter"
        className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-faint"
        placeholder="find a line you typed here"
        value={value}
        aria-label="Filter this conversation"
        spellCheck={false}
        onChange={(event) => {
          const next = new URLSearchParams(params);
          next.set("session", sessionId);
          if (event.target.value) next.set("q", event.target.value);
          else next.delete("q");
          setParams(next, { replace: true });
        }}
      />
      <Link to="/search" className="meta shrink-0 text-faint hover:text-amber">
        search the corpus
      </Link>
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
function Turn({ message }: { message: ChatMessage }) {
  if (message.role === "assistant") {
    return (
      <li data-testid="message">
        <Answer message={message} />
      </li>
    );
  }
  return (
    <li className="flex flex-col gap-2" data-testid="message">
      <div className="flex items-baseline gap-3">
        <span className="meta-label text-muted">you</span>
        <span className="meta text-faint">{timestamp(message.created_at)}</span>
        <IntentMark message={message} />
      </div>
      <p className="whitespace-pre-wrap text-ink">{message.content}</p>
      <AttachmentList attachments={message.attachments} />
      {message.memory_id ? <StoredLine memoryId={message.memory_id} /> : null}
      {message.external_key && !message.memory_id ? (
        // The key outlived the memory: it was deleted, or a replay has not
        // rebuilt it yet. Said rather than rendered as an un-clickable "stored",
        // because a link that silently does nothing is worse than a sentence.
        <p className="meta text-faint">stored · its memory is not in the corpus right now</p>
      ) : null}
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
function IntentMark({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false);
  if (message.intent === null) return null;
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
        {label[message.intent]}
      </button>
      {open ? (
        <span className="meta mt-1 block max-w-prose text-muted">
          {why[message.intent]}{" "}
          {message.memory_id ? (
            <Link to={`/memory/${message.memory_id}`} className="text-amber underline">
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
    // And it keeps polling while the tab is in the background, which the
    // default does not. The connection line is the thing this milestone is for,
    // and it arrives seconds late by design; somebody who sends a thought and
    // switches away is the normal case rather than the edge one, and
    // `refetchOnWindowFocus` alone leaves the line stale until they come back.
    // Safe because the interval above already terminates itself — this widens
    // *when* it polls, never *for how long*.
    refetchIntervalInBackground: true,
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
function Answer({ message }: { message: ChatMessage }) {
  return (
    <div className="border-l-2 border-rule-strong pl-4" data-testid="answer">
      <div className="flex items-baseline gap-3">
        <span className="meta-label text-muted">
          {message.refused ? "declined" : "answer"}
        </span>
        {message.answer_model ? (
          <span className="meta text-faint">{message.answer_model}</span>
        ) : null}
        {message.grounded === false ? (
          <span className="meta text-deny">not fully cited</span>
        ) : null}
      </div>
      <p className="mt-1 leading-relaxed text-ink">{message.content}</p>

      {message.citations.length > 0 ? (
        <ul className="mt-2 flex flex-col gap-1" data-testid="citations">
          {message.citations.map((citation) => (
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
          {message.refused
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
  queued,
  onQueue,
  onRemove,
}: {
  onSend: (text: string) => void;
  busy: boolean;
  queued: File[];
  onQueue: (files: File[]) => void;
  onRemove: (index: number) => void;
}) {
  const [text, setText] = useState("");
  const [over, setOver] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const limits = useQuery({
    queryKey: ["attach-limits"],
    queryFn: api.attachLimits,
    staleTime: Infinity,
  });

  function submit() {
    const typed = text.trim();
    // Files alone are a message. A note is optional — "here is the proposal" is
    // frequently the whole of what somebody wants to say, and requiring text
    // would make dropping a file a two-step act.
    if ((!typed && queued.length === 0) || busy) return;
    onSend(typed);
    setText("");
  }

  return (
    <form
      // `bg-paper` and not `bg-page`: the page background token is `--color-paper`,
      // and a class naming a token that does not exist compiles to nothing — which
      // makes a sticky element transparent, so the messages scroll visibly through
      // the box you are typing into.
      className={`sticky bottom-0 flex flex-col gap-1 border-t bg-paper pt-2 pb-3 ${
        over ? "border-amber-bright" : "border-rule-strong"
      }`}
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      // The drop target is the composer rather than the whole page. A page-wide
      // target catches a file somebody meant to drop on another window, and the
      // highlight has to say *where* it will land.
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        onQueue(Array.from(event.dataTransfer.files));
      }}
    >
      {queued.length > 0 ? (
        <ul className="flex flex-col gap-0.5 pb-1" data-testid="queued">
          {queued.map((file, index) => (
            <li
              key={`${file.name}-${index}`}
              className="flex items-baseline gap-2 border-l-2 border-rule pl-2"
            >
              <span className="font-mono text-sm text-ink">{file.name}</span>
              <span className="meta text-faint">{fileSize(file.size)}</span>
              <button
                type="button"
                className="meta text-faint hover:text-deny"
                aria-label={`Remove ${file.name}`}
                onClick={() => onRemove(index)}
              >
                remove
              </button>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor="chat-input" className="meta-label">
          say something
        </label>
        <button
          type="button"
          className="meta text-faint hover:text-amber"
          onClick={() => picker.current?.click()}
          // The clip is the discoverable half; the drop zone is the fast half.
          // Both, because a person who has never dropped a file into a text box
          // will not discover that they can.
          title={
            limits.data?.suffixes
              ? `Attach a file — up to ${fileSize(limits.data.max_file_bytes)}, ${limits.data.suffixes.join(", ")}`
              : "Attach a file"
          }
        >
          📎 attach
        </button>
      </div>
      <input
        ref={picker}
        type="file"
        multiple
        className="hidden"
        aria-label="Attach files"
        // The accept list comes from the API, which composes it from the parsers
        // that handle each format. A hardcoded list here would eventually differ
        // from what the pipeline can read — and a file the picker hides that the
        // system could have read is the worse of the two failures.
        accept={limits.data?.suffixes?.join(",")}
        onChange={(event) => {
          onQueue(Array.from(event.target.files ?? []));
          event.target.value = "";
        }}
      />
      <textarea
        id="chat-input"
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
        placeholder={
          queued.length > 0
            ? "say something about these files, or just send them"
            : "postgres full-text search is faster than I expected"
        }
        className="field resize-y"
        aria-label="Message"
        spellCheck={false}
        autoFocus
      />
      <p className="meta text-faint">
        <span className="kbd">enter</span> sends, <span className="kbd">shift</span>
        <span className="kbd">enter</span> starts a line. A question is answered
        and not stored; anything else is kept. Drop a file here and it becomes a
        memory too — a note beside it is kept as its own.
      </p>
    </form>
  );
}
