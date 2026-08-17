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

/**
 * A request body a test can assert on.
 *
 * JSON where it is JSON, the `FormData` itself where it is multipart, and the raw
 * value otherwise. Nothing here decides what a body *means* — a test that wants
 * the fields of a multipart body reads them off the `FormData` it gets back.
 */
function readBody(body: BodyInit | null | undefined): unknown {
  if (!body) return undefined;
  if (body instanceof FormData) return body;
  try {
    return JSON.parse(String(body));
  } catch {
    return body;
  }
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
        // Parsed when it is JSON and kept as-is when it is not. M10.2's upload
        // sends `FormData`, whose `String()` is "[object FormData]" — and because
        // the object literal is evaluated before the push, a `JSON.parse` that
        // threw here lost the *record of the call* as well as its body, so a test
        // asserting that an upload was posted saw no request at all.
        body: readBody(init?.body),
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

/**
 * What `/stats` returns. Deliberately not round numbers.
 *
 * A fixture of 100 memories and 1000 chunks is one a component can accidentally
 * hardcode and still pass. These are ugly on purpose, so a number that appears
 * on screen without coming through the API is visible as a number nobody would
 * have typed.
 */
export const STATS = {
  memories: 282,
  current_memories: 271,
  chunks: 3833,
  embedded_chunks: 3820,
  cache_entries: 3719,
  coverage: 0.9966,
  models: { "bge-small-en-v1.5@1": 3820 },
  entities: 34,
  relationships: 7,
  embedding_model: "bge-small-en-v1.5@1",
  chunker_version: "structural-v2",
  model_window: 512,
};

/** A healthy `/health/ready`. */
export const READY = {
  status: "ok",
  database: true,
  pgvector_version: "0.8.0",
  graph: true,
};

/**
 * The requests the shell itself makes on every route.
 *
 * The sidebar reads stats and readiness on all fourteen views, so a test that
 * renders `<App />` for any reason needs these routed or the harness throws on
 * an unstubbed request. Spread it into the route list rather than repeated.
 */
/** What `/chat/attach/limits` returns: 50MB, and the parsers' own suffixes. */
export const ATTACH_LIMITS = {
  max_file_bytes: 50 * 1024 * 1024,
  suffixes: [".md", ".pdf", ".py", ".txt"],
};

export const SHELL_ROUTES: Route[] = [
  { match: "/stats", body: STATS },
  { match: "/health/ready", body: READY },
  // `/` is the chat as of M10.0, so every route test loads the session list and
  // the upload limits on the way in. Listed before any bare `/chat` match a test
  // adds, because the stub matcher takes the first route whose string the URL
  // contains — and `/chat` contains neither of these.
  { match: "/chat/attach/limits", body: ATTACH_LIMITS },
  { match: "/chat/sessions", body: [] },
];

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
