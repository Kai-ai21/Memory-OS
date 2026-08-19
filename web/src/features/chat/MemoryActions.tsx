/**
 * Correcting, deleting and tagging a memory, from the message it was typed as.
 *
 * **The deletion guardrail's first appearance in the interface.** "Users can
 * permanently delete memories" has been a stated guarantee since Phase 1 and until
 * M10.4 there was no button for it anywhere — which made it a claim about the
 * schema rather than about the product.
 *
 * ## The two levels must not look alike
 *
 * They do different things and they are drawn differently. *Remove from view* is a
 * plain control that acts immediately: it is reversible, every byte is kept, and
 * putting a confirmation on it would train somebody to click through the one that
 * matters. *Delete permanently* is separated, coloured as a refusal, and opens a
 * dialog that names what will be lost and requires the word to be typed.
 *
 * ## The dialog states what is *not* erased
 *
 * The append-only log keeps its record that something was observed. That sentence
 * comes from the API — `log_note` on the scope response — rather than being written
 * here, because it is a claim about what the system does and the browser is the
 * wrong place to decide how much erasure to promise. A copy in this file is a copy
 * nobody reviewing the deletion path would read.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type ChatMessage, type DeletionScope } from "../../api/client";

/**
 * The controls on one of your own stored messages.
 *
 * Rendered only where they mean something: a message with a memory. An answer has
 * nothing to correct and a question stored nothing to delete, so neither gets a
 * row of controls that would fail on use.
 *
 * Collapsed behind one word by default. A conversation with `correct · remove ·
 * delete · tag` under every line reads as a list of dangerous options rather than
 * as something somebody wrote.
 */
export function MemoryActions({ message }: { message: ChatMessage }) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"none" | "correct" | "tag" | "delete">("none");
  const [note, setNote] = useState<string | null>(null);

  const memoryId = message.memory_id;
  if (!memoryId || message.role !== "user") return null;

  function done(message: string) {
    setNote(message);
    setMode("none");
    void client.invalidateQueries({ queryKey: ["chat-messages"] });
    void client.invalidateQueries({ queryKey: ["chat-sessions"] });
    void client.invalidateQueries({ queryKey: ["tags"] });
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline gap-3">
        <button
          type="button"
          className="meta text-ink-3 underline decoration-dotted underline-offset-2 hover:text-accent"
          aria-expanded={open}
          onClick={() => {
            setOpen((was) => !was);
            setMode("none");
          }}
        >
          {open ? "done" : "manage"}
        </button>
        {open ? (
          <>
            <button
              type="button"
              className="meta text-ink-2 hover:text-accent"
              onClick={() => setMode(mode === "correct" ? "none" : "correct")}
            >
              correct
            </button>
            <button
              type="button"
              className="meta text-ink-2 hover:text-accent"
              onClick={() => setMode(mode === "tag" ? "none" : "tag")}
            >
              tag
            </button>
            <RemoveFromView memoryId={memoryId} onDone={done} />
            {/* Set apart, and the separator is doing work: this is the only
                irreversible control in the product. */}
            <span className="meta text-ink-3">·</span>
            <button
              type="button"
              className="meta text-deny underline decoration-dotted underline-offset-2 hover:text-deny"
              onClick={() => setMode(mode === "delete" ? "none" : "delete")}
            >
              delete permanently
            </button>
          </>
        ) : null}
      </div>

      {mode === "correct" ? (
        <CorrectForm
          message={message}
          onDone={(text) => done(text)}
          onCancel={() => setMode("none")}
        />
      ) : null}
      {mode === "tag" ? (
        <TagForm
          memoryId={memoryId}
          onDone={done}
          onCancel={() => setMode("none")}
        />
      ) : null}
      {mode === "delete" ? (
        <DeleteDialog
          memoryId={memoryId}
          preview={message.content}
          onDone={done}
          onCancel={() => setMode("none")}
        />
      ) : null}
      {note ? (
        <p className="meta text-accent" role="status">
          {note}
        </p>
      ) : null}
    </div>
  );
}

/**
 * The corrected text.
 *
 * Pre-filled with what is there now, because a correction is almost always an
 * edit of a sentence rather than a replacement for it, and retyping a paragraph to
 * fix a word is how somebody decides not to bother.
 */
function CorrectForm({
  message,
  onDone,
  onCancel,
}: {
  message: ChatMessage;
  onDone: (note: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(message.content);
  const correct = useMutation({
    mutationFn: (next: string) => api.correctMessage(message.id, next),
    onSuccess: () =>
      onDone(
        "Corrected. This is version 2 of the memory; the previous version is kept and marked superseded.",
      ),
  });

  return (
    <form
      className="flex flex-col gap-2 border-l-2 border-accent pl-3"
      onSubmit={(event) => {
        event.preventDefault();
        const next = text.trim();
        if (!next || next === message.content.trim()) return;
        correct.mutate(next);
      }}
    >
      <label htmlFor={`correct-${message.id}`} className="meta-label">
        the corrected text
      </label>
      <textarea
        id={`correct-${message.id}`}
        className="min-h-20 w-full resize-y border border-rule bg-ground p-2 text-ink"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      <p className="meta text-ink-3">
        The original stays in this conversation, marked superseded. Both remain
        readable — what you thought before you corrected it is part of the record.
      </p>
      <div className="flex items-baseline gap-3">
        <button
          type="submit"
          className="meta text-accent hover:underline"
          disabled={correct.isPending}
        >
          {correct.isPending ? "correcting…" : "save the correction"}
        </button>
        <button type="button" className="meta text-ink-3" onClick={onCancel}>
          cancel
        </button>
      </div>
      {correct.isError ? (
        <p className="meta text-deny" role="alert">
          {String(correct.error)}
        </p>
      ) : null}
    </form>
  );
}

/**
 * Tags, typed the way they are typed everywhere else.
 *
 * A text field taking `#project #idea` rather than a chip picker, and the parsing
 * happens server-side. One definition of what a tag looks like, shared by the chat
 * command, the API and this box — a second one in the browser is how `#Idea` and
 * `idea` become two tags.
 */
function TagForm({
  memoryId,
  onDone,
  onCancel,
}: {
  memoryId: string;
  onDone: (note: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState("");
  const existing = useQuery({ queryKey: ["tags"], queryFn: api.tags });
  const tag = useMutation({
    mutationFn: (tags: string) => api.tagMemory(memoryId, tags),
    onSuccess: (result) => {
      const joined = result.applied.length - result.entities_created;
      onDone(
        [
          result.applied.length
            ? `Tagged ${result.applied.join(" ")}.`
            : "Nothing new.",
          result.already.length ? `Already had ${result.already.join(" ")}.` : "",
          // The interesting half. A tag that joined an existing concept connects
          // this memory to everything that concept already reaches.
          joined > 0
            ? `${joined} joined ${joined === 1 ? "a concept" : "concepts"} the corpus already knew about.`
            : "",
          result.entities_created
            ? `${result.entities_created} new concept${result.entities_created === 1 ? "" : "s"}.`
            : "",
        ]
          .filter(Boolean)
          .join(" "),
      );
    },
  });

  return (
    <form
      className="flex flex-col gap-2 border-l-2 border-rule pl-3"
      onSubmit={(event) => {
        event.preventDefault();
        if (text.trim()) tag.mutate(text.trim());
      }}
    >
      <label htmlFor={`tag-${memoryId}`} className="meta-label">
        tags
      </label>
      <input
        id={`tag-${memoryId}`}
        className="w-full border border-rule bg-ground p-2 font-mono text-sm text-ink"
        placeholder="#project #idea"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      <p className="meta text-ink-3">
        A tag becomes a concept in the same vocabulary the corpus already uses, so
        it connects to everything that mentions it rather than sitting in a list of
        its own.
      </p>
      {existing.data && existing.data.length > 0 ? (
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="meta text-ink-3">in use:</span>
          {existing.data.slice(0, 12).map((entry) => (
            <button
              key={entry.tag}
              type="button"
              className="meta font-mono text-ink-2 hover:text-accent"
              onClick={() =>
                setText((was) => (was ? `${was} ${entry.tag}` : entry.tag))
              }
            >
              {entry.tag}
            </button>
          ))}
        </div>
      ) : null}
      <div className="flex items-baseline gap-3">
        <button type="submit" className="meta text-accent hover:underline">
          {tag.isPending ? "tagging…" : "apply"}
        </button>
        <button type="button" className="meta text-ink-3" onClick={onCancel}>
          cancel
        </button>
      </div>
      {tag.isError ? (
        <p className="meta text-deny" role="alert">
          {String(tag.error)}
        </p>
      ) : null}
    </form>
  );
}

/**
 * The recoverable level, and it acts on one click.
 *
 * No confirmation, deliberately. It is reversible by design — the row, the chunks,
 * the vectors and the bytes are all kept — and a dialog here would be the thing
 * that teaches somebody to dismiss the dialog that matters.
 */
function RemoveFromView({
  memoryId,
  onDone,
}: {
  memoryId: string;
  onDone: (note: string) => void;
}) {
  const remove = useMutation({
    mutationFn: () => api.deleteMemory(memoryId, false),
    onSuccess: (result) => onDone(result.detail),
  });
  return (
    <button
      type="button"
      className="meta text-ink-2 hover:text-accent"
      onClick={() => remove.mutate()}
      disabled={remove.isPending}
    >
      {remove.isPending ? "removing…" : "remove from view"}
    </button>
  );
}

/**
 * Permanent deletion: what will be lost, what will not, and a typed confirmation.
 *
 * The counts are fetched when the dialog opens rather than carried from the
 * message, because a confirmation must name what the operation will actually hit.
 * Anything reused from a previous render is a number that may have moved — and the
 * one thing this dialog must never do is understate.
 *
 * The word has to be typed. A button that only needs a second click is a button
 * somebody double-clicks; this is the only operation in the product that cannot be
 * undone.
 */
export function DeleteDialog({
  memoryId,
  preview,
  onDone,
  onCancel,
}: {
  memoryId: string;
  preview: string;
  onDone: (note: string) => void;
  onCancel: () => void;
}) {
  const [typed, setTyped] = useState("");
  const scope = useQuery({
    queryKey: ["deletion-scope", memoryId],
    queryFn: () => api.deletionScope(memoryId),
    // Never served from cache. The whole point is that these numbers are current.
    staleTime: 0,
    gcTime: 0,
  });
  const purge = useMutation({
    mutationFn: () => api.deleteMemory(memoryId, true),
    onSuccess: (result) => onDone(result.detail),
  });

  return (
    <div
      className="flex flex-col gap-3 border-l-2 border-deny bg-ground p-3"
      role="alertdialog"
      aria-label="Delete this memory permanently"
    >
      <p className="text-ink">
        Permanently delete{" "}
        <span className="italic">“{preview.slice(0, 80)}”</span>?
      </p>

      {scope.isPending ? (
        <p className="meta text-ink-3">counting what this would remove…</p>
      ) : null}
      {scope.data ? <ScopeList scope={scope.data} /> : null}
      {scope.data ? (
        // The honest sentence, from the API. Not a paraphrase written here.
        <p className="meta max-w-prose text-ink-2" data-testid="log-note">
          {scope.data.log_note}
        </p>
      ) : null}

      <label htmlFor={`confirm-${memoryId}`} className="meta-label">
        type “delete” to confirm
      </label>
      <input
        id={`confirm-${memoryId}`}
        className="w-40 border border-deny bg-ground p-2 font-mono text-sm text-ink"
        value={typed}
        onChange={(event) => setTyped(event.target.value)}
        autoComplete="off"
      />
      <div className="flex items-baseline gap-3">
        <button
          type="button"
          className="meta text-deny hover:underline disabled:text-ink-3 disabled:no-underline"
          disabled={typed.trim().toLowerCase() !== "delete" || purge.isPending}
          onClick={() => purge.mutate()}
        >
          {purge.isPending ? "deleting…" : "delete permanently"}
        </button>
        <button type="button" className="meta text-ink-3" onClick={onCancel}>
          cancel
        </button>
      </div>
      {purge.isError ? (
        <p className="meta text-deny" role="alert">
          {String(purge.error)}
        </p>
      ) : null}
    </div>
  );
}

/**
 * The counts, in the order somebody minds them.
 *
 * The memory first, then what the corpus loses, then what other parts of the
 * product lose. Zero-valued lines are omitted: a list of noughts is a list nobody
 * reads to the end, and the lines that are there are the ones that apply.
 */
function ScopeList({ scope }: { scope: DeletionScope }) {
  const lines: string[] = [
    `${scope.memories} version${scope.memories === 1 ? "" : "s"} of this memory`,
    `${scope.chunks} chunk${scope.chunks === 1 ? "" : "s"}, ${scope.embedded_chunks} with vectors`,
  ];
  if (scope.mentions) {
    lines.push(
      scope.orphaned_entities
        ? `${scope.mentions} entity mentions, leaving ${scope.orphaned_entities} ${
            scope.orphaned_entities === 1 ? "entity" : "entities"
          } the corpus will no longer know about`
        : `${scope.mentions} entity mentions`,
    );
  }
  if (scope.tags) lines.push(`${scope.tags} tag${scope.tags === 1 ? "" : "s"}`);
  if (scope.turns) {
    lines.push(
      `${scope.turns} conversation turn${scope.turns === 1 ? "" : "s"} carrying its text`,
    );
  }
  if (scope.evidence) {
    lines.push(
      `${scope.evidence} decision evidence link${scope.evidence === 1 ? "" : "s"} — a decision will lose what it rested on`,
    );
  }
  lines.push(
    scope.shared_blobs
      ? `${scope.blobs} stored file${scope.blobs === 1 ? "" : "s"}, and ${scope.shared_blobs} kept because something else uses them`
      : `${scope.blobs} stored file${scope.blobs === 1 ? "" : "s"}`,
  );

  return (
    <ul className="flex flex-col gap-0.5" data-testid="deletion-scope">
      {lines.map((line) => (
        <li key={line} className="meta text-ink">
          {line}
        </li>
      ))}
    </ul>
  );
}

/** Bring a memory removed from view back. Shown only where one has been. */
export function RestoreControl({
  memoryId,
  onDone,
}: {
  memoryId: string;
  onDone: (note: string) => void;
}) {
  const restore = useMutation({
    mutationFn: () => api.restoreMemory(memoryId),
    onSuccess: (result) => onDone(result.detail),
  });
  return (
    <button
      type="button"
      className="meta text-accent hover:underline"
      onClick={() => restore.mutate()}
      disabled={restore.isPending}
    >
      {restore.isPending ? "restoring…" : "restore"}
    </button>
  );
}
