/**
 * A typed fetch client over the generated schema.
 *
 * Every request and response type here is derived from `schema.d.ts`, which is
 * generated from the FastAPI routes by `make types`. Nothing in this file
 * describes the API's shape in its own words — that is the point. Hand-written
 * API types drift from the backend silently, and CI fails if regenerating
 * produces a diff.
 */

import type { paths } from "./schema";

/** Where the API lives. Same-origin in tests; overridable for a real run. */
export const API_BASE: string =
  (import.meta.env?.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

/** An API response that was not a 2xx, carrying enough to show the user. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly path: string;

  constructor(status: number, detail: string, path: string) {
    super(`${status} on ${path}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.path = path;
  }
}

/** A request that never reached the API — the usual cause is nothing listening. */
export class NetworkError extends Error {
  readonly path: string;

  constructor(path: string, cause: unknown) {
    super(`could not reach the API at ${API_BASE}${path}`);
    this.name = "NetworkError";
    this.path = path;
    this.cause = cause;
  }
}

type Json = Record<string, unknown>;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    // Distinguished from an API error on purpose: "the API said no" and "there is
    // no API" need different things from the person reading the screen.
    throw new NetworkError(path, cause);
  }

  if (!response.ok) {
    throw new ApiError(response.status, await readDetail(response), path);
  }
  return (await response.json()) as T;
}

/**
 * FastAPI's error shape is `{detail: string}` for raised HTTPExceptions and
 * `{detail: [{loc, msg, ...}]}` for validation failures. Both are flattened to
 * one line, because a UI that renders a raw pydantic error array is telling the
 * user to read a stack trace.
 */
async function readDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as Json;
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          const entry = item as { loc?: unknown[]; msg?: string };
          const where = Array.isArray(entry.loc) ? entry.loc.join(".") : "";
          return where ? `${where}: ${entry.msg ?? ""}` : (entry.msg ?? "");
        })
        .join("; ");
    }
    return response.statusText;
  } catch {
    return response.statusText;
  }
}

// --------------------------------------------------------------------------
// Response aliases, so components never reach into `paths` themselves
// --------------------------------------------------------------------------

type Ok<T> = T extends { responses: { 200: { content: { "application/json": infer R } } } }
  ? R
  : never;

export type SearchResult = Ok<paths["/search"]["get"]>;
export type MemoryHit = SearchResult["hits"][number];
export type MatchedChunk = MemoryHit["matched_chunks"][number];
export type MemoryDetail = Ok<paths["/memories/{memory_id}"]["get"]>;
export type MemoryChunk = MemoryDetail["chunks"][number];
export type Source = Ok<paths["/sources"]["get"]>[number];
export type Stats = Ok<paths["/stats"]["get"]>;
export type Doctor = Ok<paths["/doctor"]["get"]>;
export type JudgementSummary = Ok<paths["/judgements"]["get"]>[number];
export type JudgementIn =
  paths["/judgements"]["post"]["requestBody"]["content"]["application/json"];
export type Verdict = JudgementIn["verdict"];
export type GoldenSet = Ok<paths["/judgements/export"]["get"]>;

export interface SearchArgs {
  q: string;
  k?: number;
  /** Repeated in the query string: `?source=a&source=b`. */
  sources?: string[];
  kind?: string;
  exact?: boolean;
}

export const api = {
  search: ({ q, k, sources, kind, exact }: SearchArgs) => {
    const params = new URLSearchParams({ q, k: String(k ?? 10) });
    // Repeated rather than comma-joined: FastAPI reads a list parameter that
    // way, and a source name could legitimately contain a comma.
    for (const name of sources ?? []) params.append("source", name);
    if (kind) params.set("kind", kind);
    if (exact) params.set("exact", "true");
    return request<SearchResult>(`/search?${params.toString()}`);
  },

  memory: (id: string) => request<MemoryDetail>(`/memories/${id}`),

  sources: () => request<Source[]>("/sources"),

  stats: () => request<Stats>("/stats"),

  doctor: () => request<Doctor>("/doctor"),

  judgements: () => request<JudgementSummary[]>("/judgements"),

  judge: (body: JudgementIn) =>
    request<{ id: string }>("/judgements", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  goldenSet: () => request<GoldenSet>("/judgements/export"),
};
