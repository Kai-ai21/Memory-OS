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
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChatPage } from "./ChatPage";
import { describeConnections } from "../../lib/connections";
import { renderWithProviders, stubFetch } from "../../test/harness";

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
};

/** The requests the page makes on mount, plus whatever a test adds. */
function chatRoutes(messages: unknown[], sessions: unknown[] = [SESSION]) {
  return [
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
      "says indexing while the vectors are still being written",
      { ...CONNECTED, embedded_chunks: 0, searchable: false, extracted: false },
      /indexing/i,
    ],
    [
      "distinguishes not-yet-looked from looked-and-found-nothing",
      { ...CONNECTED, extracted: false, connections: [], connected_memories: 0 },
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
    expect(posted?.body).toEqual({
      text: "postgres full-text search is faster than I expected",
      session_id: SESSION_ID,
      new_session: false,
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
