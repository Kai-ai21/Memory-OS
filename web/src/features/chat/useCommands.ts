/**
 * Slash commands, in the box that already exists.
 *
 * **M10.4's objective is that none of this requires leaving the chat**, and the
 * chat is one text box. So `/correct`, `/delete`, `/tag` and the rest are typed
 * where messages are typed, and each one calls the same endpoint the buttons call —
 * not a parallel implementation that happens to agree today.
 *
 * ## Why this is intercepted before the classifier, always
 *
 * `/delete that last thing` reads as a *statement* to `classify`, which is correct
 * behaviour for a rules classifier biased towards storing — and catastrophic here:
 * the command would be stored as a memory and the thing it named would still be
 * there. So a leading `/` is handled in the browser and never sent to `/chat`.
 *
 * That is the one place in this feature where the client decides something, and it
 * is not a second classifier: it is a syntax check on the first character, with no
 * opinion about what the message means. The server still classifies everything that
 * is not a command.
 *
 * ## What "the message" means
 *
 * The last turn in this conversation that stored something. Not the last turn — an
 * answer stores nothing, and neither does a question — so `/correct` after asking
 * something still corrects the note you typed before it, which is what somebody
 * means. A memory id may be given explicitly to reach any other.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type ChatMessage } from "../../api/client";

const HELP = [
  "/correct <text> — replace what the last stored message says; the original is kept",
  "/delete — remove it from view (recoverable)",
  "/delete --permanent — destroy it: memory, chunks, vectors, mentions, bytes",
  "/restore — bring back a message removed from view",
  "/tag #project #idea — file it under these concepts",
  "/untag #idea — remove tags",
  "/filter #idea — show only turns carrying a tag",
  "/help — this list",
].join(" · ");

export interface Commands {
  note: string | null;
  run: (typed: string) => Promise<void>;
  clear: () => void;
}

/**
 * `messages` is passed in rather than re-fetched, so a command acts on exactly
 * what is on screen. A command that resolved "the last stored message" from a
 * fresher list than the reader is looking at would occasionally correct a different
 * message than the one they meant.
 */
export function useCommands({
  messages,
  onFilter,
}: {
  messages: ChatMessage[];
  onFilter: (tags: string[]) => void;
}): Commands {
  const client = useQueryClient();
  const [note, setNote] = useState<string | null>(null);

  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["chat-messages"] });
    void client.invalidateQueries({ queryKey: ["chat-sessions"] });
    void client.invalidateQueries({ queryKey: ["tags"] });
  };

  const run = useMutation({
    mutationFn: async (typed: string) => {
      const [head, ...words] = typed.trim().split(/\s+/);
      const command = head.toLowerCase();
      let rest = typed.trim().slice(head.length).trim();

      if (command === "/help" || command === "/?") return HELP;
      if (command === "/filter") {
        const tags = parseTags(rest);
        onFilter(tags);
        return tags.length
          ? `Filtering this conversation by ${tags.map((tag) => `#${tag}`).join(" ")}.`
          : "Filter cleared.";
      }

      // An explicit memory id as the first argument, otherwise the last stored
      // message. Recognised by shape rather than by a flag, because a uuid is
      // unmistakable and `--id` would be ceremony on the rare case.
      let target = lastStored(messages);
      if (words[0] && isUuid(words[0])) {
        const named = messages.find(
          (message) => message.memory_id === words[0] || message.id === words[0],
        );
        target = named ?? null;
        rest = rest.slice(words[0].length).trim();
        if (!target) {
          throw new Error(`nothing in this conversation stores memory ${words[0]}`);
        }
      }
      if (!target?.memory_id) {
        throw new Error("nothing in this conversation has been stored yet");
      }
      const memoryId = target.memory_id;

      switch (command) {
        case "/correct": {
          if (!rest) throw new Error("/correct needs the corrected text");
          await api.correctMessage(target.id, rest);
          refresh();
          return "Corrected. This is version 2; the previous version is kept and marked superseded.";
        }
        case "/delete":
        case "/rm": {
          if (rest.split(/\s+/).includes("--permanent")) {
            // **Not performed from here.** A typed command is not a confirmation:
            // the guardrail is that permanent deletion names what will be lost and
            // requires that to be acknowledged, and a client that destroyed a
            // memory because a line ended in `--permanent` would be the guardrail
            // with a shortcut around it. The command opens the dialog.
            const scope = await api.deletionScope(memoryId);
            return (
              `Permanent deletion needs confirmation: ${scope.memories} version(s), ` +
              `${scope.chunks} chunk(s), ${scope.mentions} mention(s), ` +
              `${scope.turns} turn(s) and ${scope.blobs} file(s) would go. ` +
              `Use “manage → delete permanently” on the message to confirm.`
            );
          }
          const result = await api.deleteMemory(memoryId, false);
          refresh();
          return result.detail;
        }
        case "/restore": {
          const result = await api.restoreMemory(memoryId);
          refresh();
          return result.detail;
        }
        case "/tag": {
          const result = await api.tagMemory(memoryId, rest);
          refresh();
          const joined = result.applied.length - result.entities_created;
          return [
            result.applied.length ? `Tagged ${result.applied.join(" ")}.` : "",
            result.already.length ? `Already had ${result.already.join(" ")}.` : "",
            joined > 0
              ? `${joined} joined ${joined === 1 ? "a concept" : "concepts"} the corpus already knew about.`
              : "",
          ]
            .filter(Boolean)
            .join(" ");
        }
        case "/untag": {
          const result = await api.untagMemory(memoryId, rest);
          refresh();
          return result.applied.length
            ? `Removed ${result.applied.join(" ")}.`
            : "None of those tags were on it.";
        }
        default:
          throw new Error(`unknown command ${command}. /help lists them.`);
      }
    },
    onSuccess: (message) => setNote(message),
    onError: (error) => setNote(String(error)),
  });

  return {
    note,
    run: async (typed: string) => {
      setNote(null);
      await run.mutateAsync(typed).catch(() => undefined);
    },
    clear: () => setNote(null),
  };
}

/** The last turn that stored something. An answer and a question both store nothing. */
function lastStored(messages: ChatMessage[]): ChatMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user" && message.memory_id) return message;
  }
  return null;
}

/**
 * `#tag` tokens, without the sigil, casefolded.
 *
 * The same character class the server's parser uses. Kept in step deliberately, and
 * only used for the client-side filter — every *write* sends the raw text and lets
 * the server parse it, so there is one authority on what a tag is.
 */
function parseTags(text: string): string[] {
  const found = text.match(/#[\w-]+/g) ?? [];
  return [...new Set(found.map((tag) => tag.slice(1).toLowerCase()))];
}

function isUuid(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
}
