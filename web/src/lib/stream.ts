/**
 * Reading server-sent events, for the two streams that need different things.
 *
 * **`/chat/ask` cannot use `EventSource`.** That API only issues GET, and a
 * question with three turns of conversation behind it does not belong in a query
 * string — it is long, it is the user's own words, and it would land in every
 * access log between here and the server. So it is a POST read with a stream
 * reader, and the cost is this file: framing, and a reconnect policy nobody
 * writes for you.
 *
 * `/chat/events` *is* `EventSource`-shaped and uses the browser's own, which
 * reconnects and replays `Last-Event-ID` without a line of code here. Two
 * transports for two shapes, rather than one abstraction that fits neither.
 */

import { API_BASE } from "../api/client";

/** One decoded frame. `id` is absent on the answer stream, which cannot resume. */
export interface Frame {
  event: string;
  data: unknown;
  id?: string;
}

/**
 * Split a byte stream into SSE frames.
 *
 * Frames are separated by a blank line and a chunk boundary can fall anywhere —
 * mid-frame, mid-field, mid-UTF-8-sequence. The buffer is what makes that safe;
 * `TextDecoder` with `stream: true` is what makes the last one safe, because a
 * multi-byte character split across two chunks decodes to a replacement character
 * without it.
 */
export async function* frames(body: ReadableStream<Uint8Array>): AsyncGenerator<Frame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const frame = parse(raw);
        if (frame) yield frame;
        split = buffer.indexOf("\n\n");
      }
    }
  } finally {
    // Releasing matters on the abort path: a reader still locked to a cancelled
    // body keeps the connection from being collected.
    reader.releaseLock();
  }
}

function parse(raw: string): Frame | null {
  // A comment. Heartbeats are comments precisely so that a client can ignore
  // them without knowing what they are for.
  if (raw.startsWith(":")) return null;

  let event = "message";
  let id: string | undefined;
  const data: string[] = [];

  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("id:")) id = line.slice(3).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (data.length === 0) return null;

  try {
    return { event, data: JSON.parse(data.join("\n")), id };
  } catch {
    // A frame this client cannot read is skipped rather than thrown: one bad
    // frame must not end a stream that is otherwise working.
    return null;
  }
}

export interface AskArgs {
  question: string;
  sessionId?: string | null;
  k?: number;
  signal?: AbortSignal;
}

/** POST a question and yield its events as they arrive. */
export async function* ask({
  question,
  sessionId,
  k,
  signal,
}: AskArgs): AsyncGenerator<Frame> {
  const response = await fetch(`${API_BASE}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId ?? null, k: k ?? 10 }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`the answer stream did not start (${response.status})`);
  }
  yield* frames(response.body);
}
