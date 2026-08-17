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

/**
 * Readiness, which is the one endpoint whose failure body is the answer.
 *
 * `/health/ready` returns 503 when Postgres is unreachable and 200 with
 * `status: degraded` when only the graph is. Both carry the same shape, and
 * both are things the indicator should say in words. Only a request that never
 * arrived is an error here — that one means the API itself is not running,
 * which is a different sentence.
 */
async function readiness(): Promise<Readiness> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/health/ready`);
  } catch (cause) {
    throw new NetworkError("/health/ready", cause);
  }
  try {
    return (await response.json()) as Readiness;
  } catch {
    throw new ApiError(response.status, "unreadable readiness body", "/health/ready");
  }
}

// --------------------------------------------------------------------------
// Response aliases, so components never reach into `paths` themselves
// --------------------------------------------------------------------------

type Ok<T> = T extends { responses: { 200: { content: { "application/json": infer R } } } }
  ? R
  : never;

/** The 201 counterpart, for routes that answer Created rather than OK. */
type Created<T> = T extends {
  responses: { 201: { content: { "application/json": infer R } } };
}
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
export type MemorySummary = Ok<paths["/memories"]["get"]>[number];
export type Readiness = Ok<paths["/health/ready"]["get"]>;
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
export type Reflection = Ok<paths["/reflections"]["get"]>[number];
export type ReflectionCitation = Reflection["citations"][number];
export type Surfaced = Ok<paths["/surfacing"]["get"]>[number];
export type SurfaceReason = Surfaced["reason"];

export type UserModel = Ok<paths["/model"]["get"]>;
export type Facet = UserModel["facets"][string][number];
export type Assessment = UserModel["assessments"][number];

export type AgentAnswer = Ok<paths["/agent/ask"]["post"]>;
export type AgentClaim = AgentAnswer["verification"]["claims"][number];
export type AgentStep = AgentAnswer["steps"][number];

export type ChatMessage = Ok<paths["/chat/{session_id}"]["get"]>[number];
export type MessageIntent = NonNullable<ChatMessage["intent"]>;
export type ChatCitation = ChatMessage["citations"][number];
export type ChatExchange = Created<paths["/chat"]["post"]>;
export type ChatSession = Ok<paths["/chat/sessions"]["get"]>[number];
export type MessageStatus = Ok<paths["/chat/messages/{memory_id}/status"]["get"]>;
export type Stage = MessageStatus["stage"];
export type Attachment = ChatMessage["attachments"][number];
export type AttachLimits = Ok<paths["/chat/attach/limits"]["get"]>;
export type Connection = MessageStatus["connections"][number];
export type CreateSourceIn =
  paths["/sources"]["post"]["requestBody"]["content"]["application/json"];

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

  /**
   * Current memories, newest ingestion first.
   *
   * Two callers with different reasons. The overview shows the first handful as
   * recent activity. The command palette pulls a larger page once when it
   * opens and filters client-side, because "open a memory by path" has to
   * answer on every keystroke and a request per keystroke against a local API
   * is both slower and noisier than holding 500 keys in memory.
   */
  memories: (limit = 50) => request<MemorySummary[]>(`/memories?limit=${limit}`),

  /**
   * What this instance can currently do. Cheap: two connectivity checks.
   *
   * The only endpoint here read through `readiness` rather than `request`,
   * because it answers 503 when Postgres is unreachable and that response has
   * a *body worth rendering*. Letting the generic client raise on it would
   * turn "the database is down, the graph is up" into "request failed", which
   * is the one thing a health indicator must never do.
   */
  ready: () => readiness(),

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

  /**
   * Reflections, and this is the only place the client asks for them. Nothing
   * else in this file returns one as part of a larger payload, which is what
   * keeps them off the home screen and out of search results by construction
   * rather than by a component remembering not to render them.
   */
  reflections: (includeDismissed = false) =>
    request<Reflection[]>(
      `/reflections${includeDismissed ? "?include_dismissed=true" : ""}`,
    ),

  acknowledgeReflection: (id: string) =>
    request<null>(`/reflections/${id}/acknowledge`, { method: "POST" }),

  dismissReflection: (id: string, reason: string) =>
    request<null>(`/reflections/${id}/dismiss`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    }),

  /**
   * What the system volunteered, and — with `includeRefused` — every time it
   * decided not to.
   *
   * The default is the panel's view: shown, unjudged, recent. The other one is
   * the answer to "why didn't it show me anything", which is a question only a
   * push system has to be able to answer and only a log of refusals can.
   */
  surfacing: (includeRefused = false, focus?: string) => {
    const params = new URLSearchParams();
    if (includeRefused) params.set("include_refused", "true");
    if (focus) params.set("focus", focus);
    const query = params.toString();
    return request<Surfaced[]>(`/surfacing${query ? `?${query}` : ""}`);
  },

  // Two calls rather than one taking a verdict, mirroring the API. They are not
  // symmetric: dismissing raises this focus's bar and suppresses the same
  // context for a month, marking useful lowers it and does nothing else.
  dismissSurfacing: (id: string) =>
    request<null>(`/surfacing/${id}/dismiss`, { method: "POST" }),

  markSurfacingUseful: (id: string) =>
    request<null>(`/surfacing/${id}/useful`, { method: "POST" }),

  // A minute is a normal duration for this one: six hops is six model calls.
  // No timeout is set here because the browser's default is longer than any
  // trajectory this system produces, and a shorter one would abandon work that
  // has already been paid for.
  model: (dimension?: string) => {
    const query = dimension ? `?dimension=${encodeURIComponent(dimension)}` : "";
    return request<UserModel>(`/model${query}`);
  },

  facetHistory: (id: string) => request<Facet[]>(`/model/${id}/history`),

  agentAsk: (question: string, maxHops?: number) =>
    request<AgentAnswer>("/agent/ask", {
      method: "POST",
      body: JSON.stringify({ question, max_hops: maxHops ?? null }),
    }),

  /**
   * The front door.
   *
   * One call for both behaviours, because the server decides which one this is
   * — a client that classified the message itself would be a second classifier,
   * and the failure of two classifiers is a message stored by one and answered
   * by the other.
   */
  chatSessions: (includeArchived = false) =>
    request<ChatSession[]>(
      `/chat/sessions${includeArchived ? "?include_archived=true" : ""}`,
    ),

  /**
   * One conversation's turns, oldest first.
   *
   * `q` filters *within* the session by substring, server-side. It is
   * deliberately not corpus search: this answers "where in this conversation did
   * I say that" over rows already on screen, and `/search` is the other question.
   */
  chatMessages: (sessionId: string, q?: string) => {
    const params = new URLSearchParams();
    if (q && q.trim()) params.set("q", q.trim());
    const query = params.toString();
    return request<ChatMessage[]>(`/chat/${sessionId}${query ? `?${query}` : ""}`);
  },

  /**
   * Send a message.
   *
   * `deferAnswer` stores and classifies without answering, so a streaming client
   * can open `/chat/ask` for the questions. The *server* still decides which is
   * which — a second classifier in the browser would eventually disagree with it,
   * and then a statement stored by one would be answered by the other.
   */
  send: (
    text: string,
    sessionId?: string | null,
    newSession = false,
    deferAnswer = false,
  ) =>
    request<ChatExchange>("/chat", {
      method: "POST",
      body: JSON.stringify({
        text,
        session_id: sessionId ?? null,
        new_session: newSession,
        defer_answer: deferAnswer,
      }),
    }),

  attachLimits: () => request<AttachLimits>("/chat/attach/limits"),

  /**
   * Files into the chat, as multipart.
   *
   * `FormData` and *no* `Content-Type` header: the browser has to set it, because
   * only the browser knows the multipart boundary it generated. Passing the
   * generic client's `application/json` here would produce a body the server
   * cannot parse, which is why this is the one call that goes around `request`.
   */
  attach: async (
    files: File[],
    { note, sessionId, newSession }: AttachArgs = {},
  ): Promise<ChatExchange> => {
    const body = new FormData();
    for (const file of files) body.append("files", file);
    if (note?.trim()) body.append("note", note.trim());
    if (sessionId) body.append("session_id", sessionId);
    if (newSession) body.append("new_session", "true");

    let response: Response;
    try {
      response = await fetch(`${API_BASE}/chat/attach`, { method: "POST", body });
    } catch (cause) {
      throw new NetworkError("/chat/attach", cause);
    }
    if (!response.ok) {
      throw new ApiError(response.status, await readDetail(response), "/chat/attach");
    }
    return (await response.json()) as ChatExchange;
  },

  archiveSession: (id: string, archived = true) =>
    request<null>(
      `/chat/sessions/${id}/archive${archived ? "" : "?archived=false"}`,
      { method: "POST" },
    ),

  /**
   * How far a stored message has got, and what it connects to.
   *
   * Polled by the chat page until `extracted`, which is the honest end state:
   * extraction records the *attempt*, so a message that mentions nothing stops
   * being pending rather than spinning forever.
   */
  messageStatus: (memoryId: string) =>
    request<MessageStatus>(`/chat/messages/${memoryId}/status`),

  createSource: (body: CreateSourceIn) =>
    request<Source>("/sources", { method: "POST", body: JSON.stringify(body) }),

  syncSource: (id: string, full = false) =>
    request<{ job_id: string | null; source_id: string; full: boolean }>(
      `/sources/${id}/sync${full ? "?full=true" : ""}`,
      { method: "POST" },
    ),
};

export interface AttachArgs {
  note?: string;
  sessionId?: string | null;
  newSession?: boolean;
}

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
