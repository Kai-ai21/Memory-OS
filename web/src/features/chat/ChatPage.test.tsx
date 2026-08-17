/**
 * The front door, and the two things it must not get wrong.
 *
 * A refusal has to reach the screen as a refusal — the failure this milestone
 * warns about is the conversational softening that turns "the passages do not
 * cover this" into something a reader will act on — and the connection line has
 * to say which of its four states it is in rather than rendering nothing while
 * it waits.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChatPage } from "./ChatPage";
import { describeConnections } from "../../lib/connections";
import { ATTACH_LIMITS, renderWithProviders, stubFetch } from "../../test/harness";

const MEMORY_ID = "11111111-1111-7111-8111-111111111111";
const SESSION_ID = "99999999-9999-7999-8999-999999999999";

const SESSION = {
  id: SESSION_ID,
  title: "postgres full-text search is faster than I expected",
  started_at: "2026-08-17T10:00:00Z",
  last_activity: "2026-08-17T10:05:00Z",
  message_count: 2,
  archived_at: null,
};

const STATEMENT = {
  id: "aaaaaaaa-aaaa-7aaa-8aaa-aaaaaaaaaaaa",
  session_id: SESSION_ID,
  role: "user",
  content: "postgres full-text search is faster than I expected",
  ordinal: 0,
  intent: "statement",
  created_at: "2026-08-17T10:00:00Z",
  external_key: "2026-08-17/aaaa.md",
  memory_id: MEMORY_ID,
  answer_model: null,
  refused: null,
  grounded: null,
  citations: [],
  attachments: [],
};

const QUESTION = {
  id: "cccccccc-cccc-7ccc-8ccc-ccccccccccc1",
  session_id: SESSION_ID,
  role: "user",
  content: "what did I say about sourdough?",
  ordinal: 1,
  intent: "question",
  created_at: "2026-08-17T10:05:00Z",
  external_key: null,
  memory_id: null,
  answer_model: null,
  refused: null,
  grounded: null,
  citations: [],
  attachments: [],
};

const REFUSAL = {
  id: "bbbbbbbb-bbbb-7bbb-8bbb-bbbbbbbbbbbb",
  session_id: SESSION_ID,
  role: "assistant",
  content:
    "The retrieved passages do not contain anything about this, so there is nothing here to answer from.",
  ordinal: 2,
  intent: null,
  created_at: "2026-08-17T10:05:00Z",
  external_key: null,
  memory_id: null,
  answer_model: "groq/llama",
  refused: true,
  grounded: true,
  citations: [],
  attachments: [],
};

/** The requests the page makes on mount, plus whatever a test adds. */
function chatRoutes(messages: unknown[], sessions: unknown[] = [SESSION]) {
  return [
    // Before the session route, because the stub matcher takes the first entry
    // whose string the URL contains and `/chat/attach/limits` would otherwise
    // never be reached past a bare `/chat` match a caller adds.
    { match: "/chat/attach/limits", body: ATTACH_LIMITS },
    { match: `/chat/${SESSION_ID}`, body: messages },
    { match: "/chat/sessions", body: sessions },
    { match: `/chat/messages/${MEMORY_ID}/status`, body: CONNECTED },
  ];
}

const CONNECTED = {
  memory_id: MEMORY_ID,
  chunks: 1,
  embedded_chunks: 1,
  extracted: true,
  searchable: true,
  stage: "indexed" as const,
  failure: null,
  connections: [
    { entity_id: "cccccccc-cccc-7ccc-8ccc-cccccccccccc", name: "postgres", memories: 3 },
    { entity_id: "dddddddd-dddd-7ddd-8ddd-dddddddddddd", name: "indexing", memories: 1 },
  ],
  connected_memories: 3,
};

afterEach(() => vi.unstubAllGlobals());

describe("the connection line", () => {
  it("names the entities and counts the memories once", () => {
    // Three, not four. Two entities reaching three and one memory is not four
    // memories — a line that added the per-entity counts up would claim more
    // than the corpus holds, which is the one thing this sentence must not do.
    expect(describeConnections(CONNECTED)).toBe(
      "connects to 3 earlier memories via postgres, indexing",
    );
  });

  it.each([
    [
      "says which stage it is in while the vectors are still being written",
      {
        ...CONNECTED,
        chunks: 1,
        embedded_chunks: 0,
        searchable: false,
        extracted: false,
        stage: "chunking" as const,
      },
      /chunking and embedding/i,
    ],
    [
      "shows the parser's own sentence when a file could not be read",
      {
        ...CONNECTED,
        chunks: 0,
        embedded_chunks: 0,
        searchable: false,
        extracted: false,
        stage: "failed" as const,
        failure:
          "PDF 'scan.pdf' yielded 0 characters across 3 page(s); it is almost certainly scanned and needs OCR",
      },
      /almost certainly scanned and needs OCR/,
    ],
    [
      "distinguishes not-yet-looked from looked-and-found-nothing",
      {
        ...CONNECTED,
        extracted: false,
        connections: [],
        connected_memories: 0,
      },
      /looking for what it connects to/i,
    ],
    [
      "says so plainly when nothing connected",
      { ...CONNECTED, connections: [], connected_memories: 0 },
      /nothing here appears in an earlier memory/i,
    ],
  ])("%s", (_name, status, expected) => {
    expect(describeConnections(status)).toMatch(expected);
  });

  it("renders the line under a stored message", async () => {
    stubFetch(chatRoutes([STATEMENT]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    expect(
      await screen.findByText(/connects to 3 earlier memories via postgres/i),
    ).toBeInTheDocument();
  });
});

describe("an answer", () => {
  it("renders a refusal as a refusal, unsoftened", async () => {
    stubFetch(chatRoutes([QUESTION, REFUSAL]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const answer = await screen.findByTestId("answer");
    // Labelled `declined`, and carrying the API's own words. Nothing here may
    // wrap them in "hmm, I'm not sure, but maybe" — that softening is how the
    // guardrail stops being read as one.
    expect(within(answer).getByText(/declined/i)).toBeInTheDocument();
    expect(
      within(answer).getByText(/do not contain anything about this/i),
    ).toBeInTheDocument();
    expect(within(answer).getByText(/nothing was cited/i)).toBeInTheDocument();
  });

  it("shows no stored line for a question, because nothing was stored", async () => {
    stubFetch(chatRoutes([QUESTION, REFUSAL]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await screen.findByTestId("answer");
    expect(screen.queryByTestId("connection-line")).not.toBeInTheDocument();
  });
});

describe("sending", () => {
  it("posts the typed text and does not classify it here", async () => {
    // The client sends a string and renders whatever came back. A second
    // classifier in the browser would eventually disagree with the server's,
    // and the symptom would be a message stored by one and answered by the
    // other.
    const calls = stubFetch([
      ...chatRoutes([STATEMENT]),
      {
        match: "/chat",
        body: { session_id: SESSION_ID, messages: [STATEMENT] },
        status: 201,
      },
    ]);
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await userEvent.type(
      await screen.findByLabelText("Message"),
      "postgres full-text search is faster than I expected{Enter}",
    );

    const posted = calls.find((call) => call.method === "POST");
    // The session travels with the message. A page that omitted it would let the
    // thirty-minute rule reopen a conversation the reader has explicitly opened.
    //
    // `defer_answer` is the M10.3 half: the server still classifies — a second
    // classifier here would eventually disagree with it — but it stores without
    // answering, and this page opens `/chat/ask` for whatever comes back a
    // question.
    expect(posted?.body).toEqual({
      text: "postgres full-text search is faster than I expected",
      session_id: SESSION_ID,
      new_session: false,
      defer_answer: true,
    });
  });

  it("shift-enter writes a line rather than sending", async () => {
    const calls = stubFetch(chatRoutes([]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const box = await screen.findByLabelText("Message");
    await userEvent.type(box, "first{Shift>}{Enter}{/Shift}second");

    expect(box).toHaveValue("first\nsecond");
    expect(calls.some((call) => call.method === "POST")).toBe(false);
  });
});

describe("the classification", () => {
  it("is visible on the message and explains itself", async () => {
    stubFetch(chatRoutes([STATEMENT]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const mark = await screen.findByRole("button", { name: "stored" });
    await userEvent.click(mark);

    expect(await screen.findByText(/read as a claim/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open the memory/i })).toHaveAttribute(
      "href",
      `/memory/${MEMORY_ID}`,
    );
  });
});

describe("sessions", () => {
  it("lists conversations and loads the one that is clicked", async () => {
    const other = {
      ...SESSION,
      id: "88888888-8888-7888-8888-888888888888",
      title: "a different conversation",
      message_count: 1,
    };
    const calls = stubFetch([
      { match: `/chat/${other.id}`, body: [] },
      ...chatRoutes([STATEMENT], [SESSION, other]),
    ]);
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    // Awaited on the rows rather than on the list: the `ul` renders while the
    // request is still in flight, so `findByTestId("sessions")` resolves against
    // an empty rail and asserts nothing.
    expect(await screen.findAllByTestId("session-row")).toHaveLength(2);

    await userEvent.click(
      screen.getByRole("button", { name: /a different conversation/ }),
    );

    // The clicked session is fetched. Its id is in the URL too, so the
    // conversation is a link somebody can paste.
    expect(calls.some((call) => call.url.includes(`/chat/${other.id}`))).toBe(true);
  });

  it("filters within a session server-side, and says it is not corpus search", async () => {
    const calls = stubFetch(chatRoutes([STATEMENT]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await userEvent.type(
      await screen.findByLabelText("Filter this conversation"),
      "postgres",
    );

    // Sent as `q` on the session route rather than to `/search`: this is a
    // substring filter over rows already on screen, and the page offers the
    // corpus search as a separate, labelled way out.
    expect(
      calls.some((call) => call.url.includes(`/chat/${SESSION_ID}?q=postgres`)),
    ).toBe(true);
    expect(
      screen.getByRole("link", { name: /search the corpus/i }),
    ).toBeInTheDocument();
  });

  it("a new conversation creates nothing until something is typed", async () => {
    const calls = stubFetch(chatRoutes([STATEMENT]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await screen.findAllByTestId("session-row");
    await userEvent.click(screen.getByRole("button", { name: "new" }));

    // No POST. A conversation begins on its first message, which is what keeps
    // the rail free of empty rows nobody typed into.
    expect(calls.some((call) => call.method === "POST")).toBe(false);
    expect(await screen.findByText(/nothing typed yet/i)).toBeInTheDocument();
  });

  it("archives without deleting, and says so", async () => {
    const calls = stubFetch([
      { match: "/archive", body: null, status: 204 },
      ...chatRoutes([STATEMENT]),
    ]);
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await screen.findAllByTestId("session-row");
    const archive = screen.getByRole("button", { name: "archive" });
    // The promise is in the control's own tooltip, not buried in a help page.
    expect(archive).toHaveAttribute(
      "title",
      expect.stringContaining("stays a memory"),
    );

    await userEvent.click(archive);
    expect(
      calls.some(
        (call) => call.method === "POST" && call.url.includes(`${SESSION_ID}/archive`),
      ),
    ).toBe(true);
  });
});

describe("attachments", () => {
  const ATTACHED = {
    ...STATEMENT,
    id: "eeeeeeee-eeee-7eee-8eee-eeeeeeeeeee1",
    content: "this is the vendor's proposal",
    attachments: [
      {
        id: "ffffffff-ffff-7fff-8fff-fffffffffff1",
        ordinal: 0,
        filename: "proposal.pdf",
        byte_size: 2_411_724,
        media_type: "application/pdf",
        external_key: "2026-08-17/proposal.pdf#0198",
        content_hash: "a".repeat(64),
        deduplicated: false,
        memory_id: MEMORY_ID,
      },
    ],
  };

  it("shows filename, size, type, and the stage it is actually in", async () => {
    stubFetch([
      {
        match: `/chat/messages/${MEMORY_ID}/status`,
        body: { ...CONNECTED, chunks: 0, embedded_chunks: 0, searchable: false, extracted: false, stage: "parsing" },
      },
      ...chatRoutes([ATTACHED]),
    ]);
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const attachment = await screen.findByTestId("attachment");
    expect(within(attachment).getByText("proposal.pdf")).toBeInTheDocument();
    // Binary units, matching what a file manager shows.
    expect(within(attachment).getByText("2.3 MB")).toBeInTheDocument();
    expect(within(attachment).getByText("application/pdf")).toBeInTheDocument();
    // Not "done". A PDF is not searchable the moment upload finishes, and saying
    // so would make the system look broken thirty seconds later.
    expect(await within(attachment).findByText(/reading it/i)).toBeInTheDocument();
  });

  it("shows the parser's own sentence when a file cannot be read", async () => {
    const reason =
      "PDF 'scan.pdf' yielded 0 characters across 3 page(s); it is almost certainly scanned and needs OCR";
    stubFetch([
      {
        match: `/chat/messages/${MEMORY_ID}/status`,
        body: { ...CONNECTED, chunks: 0, embedded_chunks: 0, searchable: false, extracted: false, stage: "failed", failure: reason },
      },
      ...chatRoutes([ATTACHED]),
    ]);
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    // Verbatim, and at full weight. A dead-lettered attachment nobody hears about
    // is the worst outcome — they think it worked.
    expect(await screen.findByText(reason)).toBeInTheDocument();
  });

  it("says when an upload linked to bytes already in the corpus", async () => {
    stubFetch(
      chatRoutes([
        {
          ...ATTACHED,
          attachments: [{ ...ATTACHED.attachments[0], deduplicated: true }],
        },
      ]),
    );
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    // "already in memory, linked" rather than a silent success, which looks
    // identical to a re-upload that did nothing.
    expect(
      await screen.findByText(/already in memory, linked/i),
    ).toBeInTheDocument();
  });

  it("links a file to its memory, not to the conversation", async () => {
    stubFetch(chatRoutes([ATTACHED]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    // A session is a view; the memory is the thing. The file links out to where
    // it sits beside everything it connects to regardless of which conversation
    // it arrived in.
    expect(await screen.findByRole("link", { name: "proposal.pdf" })).toHaveAttribute(
      "href",
      `/memory/${MEMORY_ID}`,
    );
  });

  it("queues a dropped file and posts it as multipart with the note", async () => {
    const calls = stubFetch([
      // `/chat/attach/limits` first: the stub matcher takes the first entry whose
      // string the URL contains, and the limits URL contains `/chat/attach`.
      { match: "/chat/attach/limits", body: ATTACH_LIMITS },
      {
        match: "/chat/attach",
        body: { session_id: SESSION_ID, messages: [ATTACHED] },
        status: 201,
      },
      ...chatRoutes([STATEMENT]),
    ]);
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const file = new File(["# Proposal\n"], "proposal.md", { type: "text/markdown" });
    const box = await screen.findByLabelText("Message");
    // The composer is the drop target, not the page: a page-wide target catches a
    // file somebody meant to drop on another window.
    fireEvent.drop(box.closest("form")!, { dataTransfer: { files: [file] } });

    expect(await screen.findByTestId("queued")).toBeInTheDocument();
    await userEvent.type(box, "this is the vendor's proposal{Enter}");

    const posted = await waitFor(() => {
      const found = calls.find(
        (call) => call.method === "POST" && call.url.includes("/chat/attach"),
      );
      expect(found).toBeDefined();
      return found!;
    });

    // Multipart, with the file and the note in one request. The note is not a
    // separate send: it is a thought about these files and belongs to the same
    // turn, which is what makes it their context rather than an unrelated message
    // that happened to follow.
    const body = posted.body as FormData;
    expect((body.get("files") as File).name).toBe("proposal.md");
    expect(body.get("note")).toBe("this is the vendor's proposal");
    expect(body.get("session_id")).toBe(SESSION_ID);
  });

  it("offers a paperclip as well, because a drop zone is not discoverable", async () => {
    stubFetch(chatRoutes([STATEMENT]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    // Awaited, because the accepted formats come from the API rather than from a
    // constant here: the button renders with a plain title and gains the list
    // when `/chat/attach/limits` answers. A hardcoded list would eventually hide
    // a file the pipeline could read, which is the worse of the two failures.
    await waitFor(() =>
      expect(screen.getByLabelText("Attach files")).toHaveAttribute(
        "accept",
        ATTACH_LIMITS.suffixes.join(","),
      ),
    );
    expect(screen.getByRole("button", { name: /attach/i })).toHaveAttribute(
      "title",
      expect.stringContaining(".pdf"),
    );
  });
});
