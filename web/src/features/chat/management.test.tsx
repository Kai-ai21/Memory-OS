/**
 * Correcting, deleting and tagging from the chat, and the one thing that must not
 * be possible.
 *
 * **The deletion guardrail is what these tests are for.** "Users can permanently
 * delete memories" has been a stated guarantee since Phase 1 and M10.4 is the first
 * milestone in which anything in the interface could exercise it. So the assertions
 * here are about the *shape of the consent*: that a permanent deletion cannot be
 * triggered without the word being typed, that the dialog names what will be lost,
 * and that it states what the append-only log keeps rather than promising more
 * erasure than is delivered.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChatPage } from "./ChatPage";
import { ATTACH_LIMITS, renderWithProviders, stubFetch } from "../../test/harness";

const MEMORY_ID = "11111111-1111-7111-8111-111111111111";
const SESSION_ID = "99999999-9999-7999-8999-999999999999";
const MESSAGE_ID = "aaaaaaaa-aaaa-7aaa-8aaa-aaaaaaaaaaaa";
const CORRECTION_ID = "bbbbbbbb-bbbb-7bbb-8bbb-bbbbbbbbbbbb";

const SESSION = {
  id: SESSION_ID,
  title: "postgres full-text search",
  started_at: "2026-08-17T10:00:00Z",
  last_activity: "2026-08-17T10:05:00Z",
  message_count: 1,
  archived_at: null,
};

const STORED = {
  id: MESSAGE_ID,
  session_id: SESSION_ID,
  role: "user" as const,
  content: "postgres full-text search is faster than I expected",
  ordinal: 0,
  intent: "statement" as const,
  created_at: "2026-08-17T10:00:00Z",
  external_key: "2026-08-17/aaaa.md",
  memory_id: MEMORY_ID,
  answer_model: null,
  refused: null,
  grounded: null,
  citations: [],
  attachments: [],
  corrects: null,
  superseded_by: null,
  tags: [] as string[],
};

const STATUS = {
  memory_id: MEMORY_ID,
  chunks: 2,
  embedded_chunks: 2,
  extracted: true,
  searchable: true,
  stage: "indexed" as const,
  failure: null,
  connections: [],
  connected_memories: 0,
};

// The API's own sentence. Copied into this fixture rather than paraphrased,
// because what is being asserted is that the *interface renders what the server
// says* about how much is erased — a paraphrase here would pass while the screen
// showed something else.
const LOG_NOTE =
  "The append-only ingestion log keeps its record that something was observed " +
  "here — a hash, a byte size and a date — because deleting from the log would " +
  "rewrite history rather than delete content, and the corpus could no longer be " +
  "rebuilt from it. Everything else goes: the memory and every earlier version of " +
  "it, its chunks and their vectors, its entity mentions, its graph nodes, its " +
  "tags, the conversation turns carrying its text, and the stored file itself. " +
  "This cannot be undone.";

const SCOPE = {
  memories: 2,
  chunks: 7,
  embedded_chunks: 7,
  mentions: 9,
  orphaned_entities: 2,
  tags: 1,
  turns: 3,
  attachments: 0,
  evidence: 1,
  blobs: 2,
  shared_blobs: 1,
  previews: ["postgres full-text search is faster than I expected"],
  log_note: LOG_NOTE,
};

function routes(messages: unknown[] = [STORED], extra: unknown[] = []) {
  return [
    { match: "/chat/attach/limits", body: ATTACH_LIMITS },
    ...(extra as { match: string; body?: unknown; status?: number }[]),
    { match: `/chat/messages/${MEMORY_ID}/deletion`, body: SCOPE },
    { match: `/chat/messages/${MEMORY_ID}/status`, body: STATUS },
    { match: "/tags", body: [{ tag: "#idea", items: 4 }] },
    { match: `/chat/${SESSION_ID}`, body: messages },
    { match: "/chat/sessions", body: [SESSION] },
  ];
}

afterEach(() => vi.unstubAllGlobals());

describe("managing a memory from the chat", () => {
  it("corrects a message, and says the original is kept", async () => {
    const calls = stubFetch(
      routes([STORED], [
        {
          match: `/chat/messages/${MESSAGE_ID}/correct`,
          body: { session_id: SESSION_ID, messages: [] },
        },
      ]),
    );
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await userEvent.click(await screen.findByRole("button", { name: "manage" }));
    await userEvent.click(screen.getByRole("button", { name: "correct" }));

    const box = screen.getByLabelText("the corrected text");
    // Pre-filled, because a correction is usually an edit of a sentence and
    // retyping a paragraph to fix a word is how somebody decides not to bother.
    expect(box).toHaveValue(STORED.content);
    await userEvent.clear(box);
    await userEvent.type(box, "postgres full-text search was slower than I expected");

    // The promise the interface makes before the click, not after it.
    expect(
      screen.getByText(/the original stays in this conversation, marked superseded/i),
    ).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: /save the correction/i }),
    );

    await waitFor(() =>
      expect(
        calls.some(
          (call) =>
            call.method === "POST" &&
            call.url.includes(`/chat/messages/${MESSAGE_ID}/correct`),
        ),
      ).toBe(true),
    );
    expect(
      await screen.findByText(/version 2 of the memory; the previous version is kept/i),
    ).toBeInTheDocument();
  });

  it("keeps both versions on screen, the original marked superseded", async () => {
    // What the transcript looks like after a correction: two rows, one pointing
    // at the other. Both readable — the previous text is not noise, it is what
    // somebody believed before they corrected it, which is what Phase 5 reasons
    // over.
    const original = { ...STORED, superseded_by: CORRECTION_ID };
    const correction = {
      ...STORED,
      id: CORRECTION_ID,
      ordinal: 1,
      content: "postgres full-text search was slower than I expected",
      corrects: MESSAGE_ID,
      superseded_by: null,
    };
    stubFetch(routes([original, correction]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const rows = await screen.findAllByTestId("message");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByTestId("superseded")).toBeInTheDocument();
    expect(within(rows[0]).getByText(STORED.content)).toBeInTheDocument();
    expect(within(rows[1]).getByTestId("correction")).toBeInTheDocument();
    expect(within(rows[1]).getByText(correction.content)).toBeInTheDocument();

    // The superseded row offers no management controls: correcting a version that
    // has already been replaced would branch the history.
    expect(within(rows[0]).queryByRole("button", { name: "manage" })).toBeNull();
  });

  it("removes from view on one click, and says it can be undone", async () => {
    const calls = stubFetch(
      routes([STORED], [
        {
          match: `/chat/messages/${MEMORY_ID}`,
          body: {
            permanent: false,
            recoverable: true,
            detail:
              "Removed from view. It is excluded from search, answers and the graph, and every version, chunk and byte is still stored — so this can be undone.",
          },
        },
      ]),
    );
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await userEvent.click(await screen.findByRole("button", { name: "manage" }));
    await userEvent.click(screen.getByRole("button", { name: "remove from view" }));

    // No confirmation, deliberately: it is reversible by design, and a dialog on
    // the reversible level is what teaches somebody to click through the one that
    // matters.
    await waitFor(() =>
      expect(
        calls.some(
          (call) =>
            call.method === "DELETE" && !call.url.includes("permanent=true"),
        ),
      ).toBe(true),
    );
    expect(await screen.findByText(/so this can be undone/i)).toBeInTheDocument();
  });

  it("will not delete permanently until the word is typed", async () => {
    const calls = stubFetch(
      routes([STORED], [
        {
          match: `/chat/messages/${MEMORY_ID}?permanent=true`,
          body: {
            permanent: true,
            recoverable: false,
            memories: 2,
            chunks: 7,
            detail: "Permanently deleted 2 version(s)…",
          },
        },
      ]),
    );
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await userEvent.click(await screen.findByRole("button", { name: "manage" }));
    await userEvent.click(
      screen.getByRole("button", { name: /delete permanently/i }),
    );

    const dialog = await screen.findByRole("alertdialog");

    // The counts, from the API, read when the dialog opened. A person cannot
    // consent to an unspecified amount of loss.
    const scope = await within(dialog).findByTestId("deletion-scope");
    expect(scope).toHaveTextContent("2 versions of this memory");
    expect(scope).toHaveTextContent("7 chunks, 7 with vectors");
    expect(scope).toHaveTextContent(
      "9 entity mentions, leaving 2 entities the corpus will no longer know about",
    );
    expect(scope).toHaveTextContent("3 conversation turns carrying its text");
    expect(scope).toHaveTextContent(
      "2 stored files, and 1 kept because something else uses them",
    );

    // **And what is *not* erased, in the server's own words.** The log keeps its
    // record that something was observed here. Promising more than that is the
    // one thing this dialog must not do.
    expect(within(dialog).getByTestId("log-note")).toHaveTextContent(
      /append-only ingestion log keeps its record/i,
    );
    expect(within(dialog).getByTestId("log-note")).toHaveTextContent(
      /this cannot be undone/i,
    );

    const button = within(dialog).getByRole("button", {
      name: /delete permanently/i,
    });
    expect(button).toBeDisabled();

    // The wrong word does not enable it either. A button that a second click
    // satisfies is a button somebody double-clicks.
    await userEvent.type(within(dialog).getByLabelText(/type .delete. to confirm/i), "yes");
    expect(button).toBeDisabled();
    expect(calls.some((call) => call.url.includes("permanent=true"))).toBe(false);

    await userEvent.clear(within(dialog).getByLabelText(/type .delete. to confirm/i));
    await userEvent.type(
      within(dialog).getByLabelText(/type .delete. to confirm/i),
      "delete",
    );
    expect(button).toBeEnabled();
    await userEvent.click(button);

    await waitFor(() =>
      expect(
        calls.some(
          (call) => call.method === "DELETE" && call.url.includes("permanent=true"),
        ),
      ).toBe(true),
    );
  });

  it("tags a message, and says what the tag connected to", async () => {
    const calls = stubFetch(
      routes([STORED], [
        {
          match: `/chat/messages/${MEMORY_ID}/tags`,
          body: {
            applied: ["#postgres", "#idea"],
            already: [],
            // One of the two was already a concept the corpus knew about, which
            // is the interesting half: that tag connected this memory to
            // everything mentioning it.
            entities_created: 1,
          },
        },
      ]),
    );
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await userEvent.click(await screen.findByRole("button", { name: "manage" }));
    await userEvent.click(screen.getByRole("button", { name: "tag" }));
    await userEvent.type(screen.getByLabelText("tags"), "#postgres #idea");
    await userEvent.click(screen.getByRole("button", { name: "apply" }));

    await waitFor(() =>
      expect(
        calls.some(
          (call) =>
            call.method === "POST" &&
            call.url.includes(`/chat/messages/${MEMORY_ID}/tags`),
        ),
      ).toBe(true),
    );
    expect(
      await screen.findByText(/joined a concept the corpus already knew about/i),
    ).toBeInTheDocument();
  });

  it("renders tags as links into a filtered corpus search", async () => {
    stubFetch(routes([{ ...STORED, tags: ["#idea"] }]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    const link = await screen.findByRole("link", { name: "#idea" });
    // Into `/search`, not a conversation-local filter: the point of a tag being a
    // concept in the shared vocabulary is that it reaches the whole corpus.
    expect(link).toHaveAttribute("href", "/search?tag=idea");
  });

  it("never sends a slash command to the classifier", async () => {
    const calls = stubFetch(routes([STORED]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await screen.findAllByTestId("message");
    await userEvent.type(
      screen.getByLabelText("say something"),
      "/help{Enter}",
    );

    // **The failure this prevents is the worst one available in this feature.**
    // `/delete` reads as a *statement* to a classifier biased towards storing, so
    // sending it to `/chat` would file the command as a memory and leave the
    // thing it named in place.
    await waitFor(() => expect(screen.getByTestId("command-note")).toBeInTheDocument());
    expect(
      calls.some((call) => call.method === "POST" && call.url.endsWith("/chat")),
    ).toBe(false);
  });

  it("a typed /delete --permanent asks for confirmation rather than deleting", async () => {
    const calls = stubFetch(routes([STORED]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await screen.findAllByTestId("message");
    await userEvent.type(
      screen.getByLabelText("say something"),
      "/delete --permanent{Enter}",
    );

    // A typed command is not a confirmation. The guardrail is that permanent
    // deletion names what will be lost and requires that to be acknowledged — a
    // client that destroyed a memory because a line ended in `--permanent` would
    // be the guardrail with a shortcut around it.
    expect(
      await screen.findByText(/permanent deletion needs confirmation/i),
    ).toBeInTheDocument();
    expect(calls.some((call) => call.method === "DELETE")).toBe(false);
  });

  it("filters a conversation by tag, in the URL", async () => {
    const calls = stubFetch(routes([{ ...STORED, tags: ["#idea"] }]));
    renderWithProviders(<ChatPage />, { route: `/?session=${SESSION_ID}` });

    await screen.findAllByTestId("message");
    await userEvent.type(
      screen.getByLabelText("say something"),
      "/filter #idea{Enter}",
    );

    await waitFor(() =>
      expect(
        calls.some((call) => call.url.includes(`/chat/${SESSION_ID}?tag=idea`)),
      ).toBe(true),
    );
  });
});
