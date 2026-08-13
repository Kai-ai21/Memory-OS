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
  // 204 has no body, and `response.json()` on an empty one throws a SyntaxError
  // that reads like a malformed API rather than a successful request. The
  // reject-a-suggestion route is the first endpoint here that returns one.
  if (response.status === 204) return null as T;
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
// Present unless the caller passed `explain=false`, hence the NonNullable: the
// components that render these are only mounted when they exist.
export type Citation = NonNullable<MemoryHit["citations"]>[number];
export type Explanation = NonNullable<MemoryHit["explanation"]>;
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
export type MemoryVersion = MemoryDetail["versions"][number];
export type Timeline = Ok<paths["/timeline"]["get"]>;
export type Bucket = Timeline["buckets"][number];
export type ProvenanceBand = Timeline["provenance"][number];
export type Period = NonNullable<
  paths["/timeline"]["get"]["parameters"]["query"]
>["period"];
export type MemoriesAt = Ok<paths["/memories/at"]["get"]>;
export type MemoryAt = MemoriesAt["memories"][number];
export type Gap = Ok<paths["/gaps"]["get"]>[number];
export type Evolution = Ok<paths["/memories/{memory_id}/evolution"]["get"]>;
export type EvolutionVersion = Evolution["versions"][number];
export type VersionDiff = Evolution["diffs"][number];
export type DiffSpan = VersionDiff["spans"][number];
export type ChangeSummary = NonNullable<VersionDiff["summary"]>;
export type DecisionSummary = Ok<paths["/decisions"]["get"]>[number];
export type DecisionDetail = Ok<paths["/decisions/{decision_id}"]["get"]>;
export type DecisionOption = DecisionDetail["options"][number];
export type DecisionAssumption = DecisionDetail["assumptions"][number];
export type DecisionEvidence = DecisionDetail["evidence"][number];
export type DecisionIn =
  paths["/decisions"]["post"]["requestBody"]["content"]["application/json"];
export type DecisionEditIn =
  paths["/decisions/{decision_id}"]["patch"]["requestBody"]["content"]["application/json"];
export type DecisionStatus = NonNullable<DecisionSummary["status"]>;
export type EvidenceRelation = DecisionEvidence["relation"];
export type Suggestion = Ok<paths["/decisions/suggestions"]["get"]>[number];
export type Outcome = DecisionDetail["outcomes"][number];
export type OutcomeVerdict = Outcome["verdict"];
export type OutcomeIn =
  paths["/decisions/{decision_id}/outcomes"]["post"]["requestBody"]["content"]["application/json"];
export type OutcomeSuggestion = Ok<paths["/outcomes/suggestions"]["get"]>[number];
export type SuccessRate = Ok<paths["/outcomes/rate"]["get"]>;
export type Assumption = DecisionDetail["assumptions"][number];
export type AssumptionVerdict = NonNullable<Assumption["held"]>;
export type AssumptionDetail = Ok<paths["/assumptions"]["get"]>[number];
export type AssumptionStats = Ok<paths["/assumptions/stats"]["get"]>;
export type AssumptionGroup = AssumptionStats["groups"][number];
export type Pattern = Ok<paths["/patterns"]["get"]>[number];
export type PatternEvidence = Pattern["supporting"][number];
export type PatternKind = NonNullable<Pattern["kind"]>;
export type Calibration = Ok<paths["/patterns/calibration"]["get"]>;
export type CalibrationBand = Calibration["decisions"][number];

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

  timeline: ({ period, from, to, source }: TimelineArgs) => {
    const params = new URLSearchParams({ period });
    // Omitted rather than sent empty: the API defaults the window to what the
    // corpus covers, and a blank `from` would be a 422 instead of a default.
    if (from) params.set("from", from);
    if (to) params.set("to", to);
    if (source) params.set("source", source);
    return request<Timeline>(`/timeline?${params.toString()}`);
  },

  memoriesAt: ({ date, windowDays, source }: MemoriesAtArgs) => {
    const params = new URLSearchParams({ date, window_days: String(windowDays) });
    if (source) params.set("source", source);
    return request<MemoriesAt>(`/memories/at?${params.toString()}`);
  },

  evolution: ({ id, from, to, summarize }: EvolutionArgs) => {
    const params = new URLSearchParams();
    // Both or neither: the API rejects one alone rather than guessing the other,
    // and sending an empty `from` would be that guess made client-side.
    if (from && to) {
      params.set("from", from);
      params.set("to", to);
    }
    if (summarize) params.set("summarize", "true");
    const query = params.toString();
    return request<Evolution>(`/memories/${id}/evolution${query ? `?${query}` : ""}`);
  },

  gaps: ({ minDays, source }: GapsArgs) => {
    const params = new URLSearchParams({ min_days: String(minDays) });
    if (source) params.set("source", source);
    return request<Gap[]>(`/gaps?${params.toString()}`);
  },

  decisions: (status?: DecisionStatus) => {
    const params = new URLSearchParams();
    // Omitted rather than sent empty: no status means every status, and a blank
    // one would be a 422.
    if (status) params.set("status", status);
    const query = params.toString();
    return request<DecisionSummary[]>(`/decisions${query ? `?${query}` : ""}`);
  },

  decision: (id: string) => request<DecisionDetail>(`/decisions/${id}`),

  decide: (body: DecisionIn) =>
    request<{ id: string }>("/decisions", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  editDecision: (id: string, body: DecisionEditIn) =>
    request<DecisionDetail>(`/decisions/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  suggestions: (status: Suggestion["status"] = "pending") =>
    request<Suggestion[]>(`/decisions/suggestions?status=${status}`),

  // Two calls rather than one with a verdict, because they are not symmetric:
  // accepting writes a decision and rejecting writes nothing but a status. A
  // single endpoint taking a verdict would hide that one of them is a create.
  acceptSuggestion: (id: string, body?: DecisionIn) =>
    request<{ id: string }>(`/decisions/suggestions/${id}/accept`, {
      method: "POST",
      body: body ? JSON.stringify(body) : "null",
    }),

  rejectSuggestion: (id: string) =>
    request<null>(`/decisions/suggestions/${id}/reject`, { method: "POST" }),

  recordOutcome: (decisionId: string, body: OutcomeIn) =>
    request<{ id: string }>(`/decisions/${decisionId}/outcomes`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  outcomeSuggestions: (status: OutcomeSuggestion["status"] = "pending") =>
    request<OutcomeSuggestion[]>(`/outcomes/suggestions?status=${status}`),

  acceptOutcomeSuggestion: (id: string) =>
    request<{ id: string }>(`/outcomes/suggestions/${id}/accept`, { method: "POST" }),

  rejectOutcomeSuggestion: (id: string) =>
    request<null>(`/outcomes/suggestions/${id}/reject`, { method: "POST" }),

  successRate: () => request<SuccessRate>("/outcomes/rate"),

  assumptions: (decision?: string) => {
    const params = new URLSearchParams();
    if (decision) params.set("decision", decision);
    const query = params.toString();
    return request<AssumptionDetail[]>(`/assumptions${query ? `?${query}` : ""}`);
  },

  assumptionStats: () => request<AssumptionStats>("/assumptions/stats"),

  patterns: (includeDismissed = false) =>
    request<Pattern[]>(
      `/patterns${includeDismissed ? "?include_dismissed=true" : ""}`,
    ),

  calibration: () => request<Calibration>("/patterns/calibration"),

  dismissPattern: (id: string, reason: string) =>
    request<null>(`/patterns/${id}/dismiss`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    }),
};

export interface TimelineArgs {
  period: Period;
  /** ISO instants. Omitted means "whatever the corpus covers". */
  from?: string;
  to?: string;
  source?: string;
}

export interface MemoriesAtArgs {
  /** ISO instant: the *start* of the window, not its centre. */
  date: string;
  windowDays: number;
  source?: string;
}

export interface GapsArgs {
  minDays: number;
  source?: string;
}

export interface EvolutionArgs {
  id: string;
  /** A specific pair to diff. Both or neither. */
  from?: string;
  to?: string;
  /** Generate missing summaries. Costs a model call per diff, so opt-in. */
  summarize?: boolean;
}
