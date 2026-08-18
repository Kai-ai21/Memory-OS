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
 *
 * **`describeConnections` and `connectionParts` are one implementation.** The
 * Luminous reference sets the entity names in cyan inside the line, which needs
 * the names as data rather than as a finished sentence — and the obvious way to
 * get them, formatting the line twice, is how the rendered line and the tested
 * line start disagreeing. So the parts are the source and the sentence is
 * assembled from them.
 */

import type { MessageStatus } from "../api/client";

/** Entities a line names. Two or three is a sentence; ten is a tag cloud. */
export const NAMED_ENTITIES = 3;

/**
 * The line, split into the part that is prose and the part that is data.
 *
 * `entities` is empty for every state except the connected one, which is what
 * lets the renderer style the names without knowing which state it is in.
 */
export interface ConnectionParts {
  /** Everything before the entity names. */
  lead: string;
  /** The named entities, in the order the extractor ranked them. */
  entities: string[];
}

export function connectionParts(status: MessageStatus): ConnectionParts {
  // A failure outranks everything. A document that will never be indexed must not
  // be drawn as one still working on it, and the parser's own sentence is what is
  // shown — it names the file, counts what it found, and says what the file
  // probably is, which is the difference between running OCR and opening a ticket.
  if (status.stage === "failed") {
    return {
      lead: status.failure ?? "this could not be read, and no reason was recorded",
      entities: [],
    };
  }
  // The honest middle. A PDF is not searchable the moment its upload finishes:
  // bytes, then text, then chunks, then vectors, and on a real document that is
  // tens of seconds. Each stage says which one it is in rather than "done".
  if (status.stage === "stored") return { lead: "stored · waiting for a worker", entities: [] };
  if (status.stage === "parsing") return { lead: "reading it…", entities: [] };
  if (status.stage === "chunking") return { lead: "chunking and embedding…", entities: [] };
  // Searchable, and extraction has not run. Distinguished from the state below
  // because "we have not looked" and "we looked and found nothing shared" are
  // different facts about the corpus.
  if (!status.extracted) {
    return { lead: "searchable · looking for what it connects to…", entities: [] };
  }
  if (status.connections.length === 0) {
    return {
      lead: "searchable · nothing here appears in an earlier memory yet",
      entities: [],
    };
  }
  const reached = status.connected_memories;
  return {
    lead: `connects to ${reached} earlier ${reached === 1 ? "memory" : "memories"} via `,
    entities: status.connections
      .slice(0, NAMED_ENTITIES)
      .map((connection) => connection.name),
  };
}

export function describeConnections(status: MessageStatus): string {
  const { lead, entities } = connectionParts(status);
  return lead + entities.join(", ");
}
