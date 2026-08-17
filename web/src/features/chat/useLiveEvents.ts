/**
 * Connection lines that arrive without a refresh.
 *
 * **`EventSource`, not a hand-written reader.** `/chat/events` is a GET with no
 * body, so the browser's own implementation reconnects with backoff and replays
 * `Last-Event-ID` on its own — the whole of step 4's reconnect requirement, for
 * free, and better tested than anything written here would be.
 *
 * What this adds is what the browser cannot know: which query to invalidate when
 * a memory moves, and what to do about a `gap`. A gap means the server no longer
 * holds the events since the client's last id — a restart, or a disconnection
 * longer than the buffer — and the honest response is to refetch everything
 * rather than assume nothing happened. An empty replay and a lost one look
 * identical otherwise, and the thing silently dropped would be the connection
 * line this milestone exists to deliver.
 */

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { API_BASE } from "../../api/client";

export function useLiveEvents() {
  const client = useQueryClient();

  useEffect(() => {
    // Guarded rather than assumed. Every browser this ships to has `EventSource`
    // — it is older than `fetch` — but a test environment does not, and neither
    // does a server-rendered pass. Absence degrades to the polling that already
    // exists on each message rather than to a crashed page: live updates are a
    // latency improvement over a working fallback, not the only path.
    if (typeof EventSource === "undefined") return;

    const source = new EventSource(`${API_BASE}/chat/events`);

    source.addEventListener("memory_ready", (event) => {
      let memoryId: string | undefined;
      try {
        memoryId = (JSON.parse((event as MessageEvent).data) as { memory_id?: string })
          .memory_id;
      } catch {
        return;
      }
      if (!memoryId) return;
      // Invalidated rather than written from the payload, and deliberately: the
      // notification says *which* memory moved and nothing about where it moved
      // to, so the client re-reads the status and gets the state at delivery
      // rather than at publication. Those differ whenever two jobs finish close
      // together, and the second one is the one worth showing.
      //
      // **Never optimistic.** A connection line is drawn only from a status the
      // server confirmed. A line that appeared and then changed would be worse
      // than one that arrived late.
      void client.invalidateQueries({ queryKey: ["message-status", memoryId] });
    });

    source.addEventListener("gap", () => {
      // Everything, because the point of a gap is that this client cannot know
      // what it missed.
      void client.invalidateQueries({ queryKey: ["message-status"] });
      void client.invalidateQueries({ queryKey: ["chat-messages"] });
    });

    // No `onerror` handler that closes the stream. `EventSource` reconnects by
    // itself with its own backoff, and closing it here would replace a working
    // recovery with a dead connection — the one thing worth doing on an error is
    // nothing.

    return () => source.close();
  }, [client]);
}
