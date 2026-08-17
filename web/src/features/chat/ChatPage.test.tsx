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

const STATEMENT = {
  id: "aaaaaaaa-aaaa-7aaa-8aaa-aaaaaaaaaaaa",
  text: "postgres full-text search is faster than I expected",
  intent: "statement",
  created_at: "2026-08-17T10:00:00Z",
  memory_id: MEMORY_ID,
  answer: null,
  answer_model: null,
  refused: null,
  grounded: null,
  citations: [],
};

const REFUSAL = {
  id: "bbbbbbbb-bbbb-7bbb-8bbb-bbbbbbbbbbbb",
  text: "what did I say about sourdough?",
  intent: "question",
  created_at: "2026-08-17T10:05:00Z",
  memory_id: null,
  answer:
    "The retrieved passages do not contain anything about this, so there is nothing here to answer from.",
  answer_model: "groq/llama",
  refused: true,
  grounded: true,
  citations: [],
};

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
    stubFetch([
      { match: `/chat/${MEMORY_ID}/status`, body: CONNECTED },
      { match: "/chat", body: [STATEMENT] },
    ]);
    renderWithProviders(<ChatPage />);

    expect(
      await screen.findByText(/connects to 3 earlier memories via postgres/i),
    ).toBeInTheDocument();
  });
});

describe("an answer", () => {
  it("renders a refusal as a refusal, unsoftened", async () => {
    stubFetch([{ match: "/chat", body: [REFUSAL] }]);
    renderWithProviders(<ChatPage />);

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
    stubFetch([{ match: "/chat", body: [REFUSAL] }]);
    renderWithProviders(<ChatPage />);

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
      { match: `/chat/${MEMORY_ID}/status`, body: CONNECTED },
      { match: "/chat", body: [] },
    ]);
    renderWithProviders(<ChatPage />);

    await userEvent.type(
      await screen.findByLabelText("Message"),
      "postgres full-text search is faster than I expected{Enter}",
    );

    const posted = calls.find((call) => call.method === "POST");
    expect(posted?.body).toEqual({
      text: "postgres full-text search is faster than I expected",
    });
  });

  it("shift-enter writes a line rather than sending", async () => {
    const calls = stubFetch([{ match: "/chat", body: [] }]);
    renderWithProviders(<ChatPage />);

    const box = await screen.findByLabelText("Message");
    await userEvent.type(box, "first{Shift>}{Enter}{/Shift}second");

    expect(box).toHaveValue("first\nsecond");
    expect(calls.some((call) => call.method === "POST")).toBe(false);
  });
});

describe("the classification", () => {
  it("is visible on the message and explains itself", async () => {
    stubFetch([
      { match: `/chat/${MEMORY_ID}/status`, body: CONNECTED },
      { match: "/chat", body: [STATEMENT] },
    ]);
    renderWithProviders(<ChatPage />);

    const mark = await screen.findByRole("button", { name: "stored" });
    await userEvent.click(mark);

    expect(await screen.findByText(/read as a claim/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open the memory/i })).toHaveAttribute(
      "href",
      `/memory/${MEMORY_ID}`,
    );
  });
});
