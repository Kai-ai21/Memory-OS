/**
 * The connection line, in words.
 *
 * Its own module rather than a helper inside `ChatPage`, because this one string
 * is the thing M10.0 is for — "Stored. Connects to 3 earlier memories via
 * `postgres`, `indexing`." is the difference between a notes app and a memory
 * system — and a sentence that carries that much weight should be testable
 * without rendering a page.
 *
 * Four states, and every one of them says something. The temptation is to render
 * nothing until there is a connection, which would collapse three of these into
 * a blank line and leave the reader unable to tell "still working" from "nothing
 * connected" from "this is not indexed yet".
 */

import type { MessageStatus } from "../api/client";

/** Entities a line names. Two or three is a sentence; ten is a tag cloud. */
export const NAMED_ENTITIES = 3;

export function describeConnections(status: MessageStatus): string {
  // A failure outranks everything. A document that will never be indexed must not
  // be drawn as one still working on it, and the parser's own sentence is what is
  // shown — it names the file, counts what it found, and says what the file
  // probably is, which is the difference between running OCR and opening a ticket.
  if (status.stage === "failed") {
    return status.failure ?? "this could not be read, and no reason was recorded";
  }
  // The honest middle. A PDF is not searchable the moment its upload finishes:
  // bytes, then text, then chunks, then vectors, and on a real document that is
  // tens of seconds. Each stage says which one it is in rather than "done".
  if (status.stage === "stored") return "stored · waiting for a worker";
  if (status.stage === "parsing") return "reading it…";
  if (status.stage === "chunking") return "chunking and embedding…";
  // Searchable, and extraction has not run. Distinguished from the state below
  // because "we have not looked" and "we looked and found nothing shared" are
  // different facts about the corpus.
  if (!status.extracted) return "searchable · looking for what it connects to…";
  if (status.connections.length === 0) {
    return "searchable · nothing here appears in an earlier memory yet";
  }
  const named = status.connections
    .slice(0, NAMED_ENTITIES)
    .map((connection) => connection.name)
    .join(", ");
  const reached = status.connected_memories;
  return `connects to ${reached} earlier ${reached === 1 ? "memory" : "memories"} via ${named}`;
}
