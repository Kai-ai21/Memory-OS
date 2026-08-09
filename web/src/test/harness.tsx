/**
 * Rendering a component with the providers it needs, and a stubbed `fetch`.
 *
 * `fetch` is stubbed rather than the api module mocked, so the tests exercise the
 * real client — including its URL building and its error translation. A mocked
 * `api` object would let a wrong query string pass.
 */

import type { ReactNode } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi } from "vitest";

export function renderWithProviders(ui: ReactNode, { route = "/" }: { route?: string } = {}) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

export interface Route {
  /** Matched against the request URL with `includes`. */
  match: string;
  body?: unknown;
  status?: number;
}

/** Recorded calls, so a test can assert what was sent as well as what rendered. */
export interface Recorded {
  url: string;
  method: string;
  body: unknown;
}

export function stubFetch(routes: Route[]): Recorded[] {
  const calls: Recorded[] = [];

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({
        url,
        method: init?.method ?? "GET",
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });

      const route = routes.find((candidate) => url.includes(candidate.match));
      if (!route) {
        // Loud rather than an empty 200: an unrouted request in a test is a test
        // that is not asserting what it thinks it is.
        throw new Error(`no stub for ${url}`);
      }
      const status = route.status ?? 200;
      return new Response(JSON.stringify(route.body ?? {}), {
        status,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );

  return calls;
}

/** The chunk text used by the fixture, exported so a test can assert on it. */
export const MATCHED_TEXT = "a worker claims a job and holds a lease on it";

/**
 * A hit whose stored chunk text carries a borrowed lead-in, the way every chunk
 * after ordinal 0 does in the real corpus: the span covers only the tail.
 */
export function borrowingHit() {
  const borrowed = "borrowed context from the previous chunk. ";
  return {
    memory_id: "11111111-1111-7111-8111-111111111111",
    external_key: "src/memoryos/application/worker.py",
    source_name: "self",
    title: null,
    kind: "note",
    occurred_at: null,
    score: 0.77,
    matched_chunks: [
      {
        chunk_id: "22222222-2222-7222-8222-222222222222",
        ordinal: 3,
        text: borrowed + MATCHED_TEXT,
        score: 0.77,
        char_start: 1000,
        char_end: 1000 + MATCHED_TEXT.length,
        metadata: {},
      },
    ],
  };
}

/** A search response with one hit, shaped like the real thing. */
export function searchResponse(overrides: Record<string, unknown> = {}) {
  return {
    query: "leases",
    timing: { embed_ms: 12, search_ms: 3, total_ms: 15 },
    hits: [
      {
        memory_id: "11111111-1111-7111-8111-111111111111",
        external_key: "src/memoryos/application/worker.py",
        source_name: "self",
        title: null,
        kind: "code",
        occurred_at: "2026-08-01T10:00:00Z",
        score: 0.8147,
        matched_chunks: [
          {
            chunk_id: "22222222-2222-7222-8222-222222222222",
            ordinal: 3,
            text: MATCHED_TEXT,
            score: 0.8147,
            // Offsets into the parent memory. Computed from the text's length,
            // because hand-counting a character offset is exactly how a
            // highlight ends up one character short.
            char_start: 100,
            char_end: 100 + MATCHED_TEXT.length,
            metadata: { definition: "Worker.run" },
          },
        ],
      },
    ],
    ...overrides,
  };
}
