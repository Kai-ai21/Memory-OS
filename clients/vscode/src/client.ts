/**
 * Talking to the local API, and never throwing while doing it.
 *
 * **This module does not import `vscode`.** That is deliberate and is what lets
 * the failure behaviour be tested with `node --test` and a stubbed `fetch`,
 * rather than inside an editor host where "does it throw" is answered by
 * watching a notification appear.
 *
 * The rule the whole file exists for: a dev tool that throws errors while you
 * are trying to work gets uninstalled within a day. The API being stopped is the
 * *normal* state of a laptop, not an exception — so an unreachable server, a
 * 500, a rate limit and a malformed body all resolve to the same thing here, a
 * result object with `error` set, and the panel renders a quiet line. There is
 * no `throw` in this file and no code path that produces a modal.
 */

export interface ContextItem {
  position: number;
  title: string;
  category: string;
  text: string;
  tokens: number;
  /** Which of the four sources proposed it, and its rank in each. */
  sources: Record<string, number>;
  memory_id: string | null;
  decision_id: string | null;
  external_key: string | null;
}

export interface ContextResult {
  focus: string;
  /** False while the server is still building it. Not an error. */
  ready: boolean;
  items: ContextItem[];
  tokensUsed: number;
  tokenBudget: number;
  /** Set only when something went wrong. Rendered as a quiet line, never a modal. */
  error?: string;
  /** Round trip in milliseconds, for the latency the milestone is judged on. */
  elapsedMs: number;
}

export interface ClientOptions {
  apiUrl: string;
  tokenBudget?: number;
  /** Injected so tests can stub it and so a host without global fetch can pass one. */
  fetchImpl?: typeof fetch;
  /** Injected so tests do not sleep. */
  sleep?: (ms: number) => Promise<void>;
  log?: (message: string) => void;
}

/** How long a request may take before it is treated as a failure. */
const TIMEOUT_MS = 4000;

/**
 * How many times to re-ask while the server is building, and how long between.
 *
 * Assembly is about a second warm and up to twenty cold, so this covers the warm
 * case comfortably and gives up on the cold one rather than polling for half a
 * minute. Giving up is not a failure: the build continues on the worker, and the
 * next time the panel opens on this file the answer is already there. That is
 * the difference between "cached-or-loading" and blocking.
 */
const POLL_ATTEMPTS = 4;
const POLL_INTERVAL_MS = 700;

export class MemoryOsClient {
  private readonly apiUrl: string;
  private readonly tokenBudget: number;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly log: (message: string) => void;

  /**
   * Consecutive failures, used to stop hammering a server that is down.
   *
   * Reset on any success. While it is above zero the client still tries — it
   * simply waits longer first — because a watcher or panel that gave up
   * permanently would need the editor restarting after every `docker compose
   * restart`, which is the same as not surviving one.
   */
  private failures = 0;

  constructor(options: ClientOptions) {
    this.apiUrl = options.apiUrl.replace(/\/+$/, "");
    this.tokenBudget = options.tokenBudget ?? 4000;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
    this.sleep = options.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.log = options.log ?? (() => undefined);
  }

  /**
   * Tell the API a file is being worked on. Fire and forget.
   *
   * The result is deliberately ignored by callers: this is a record of what
   * happened, and a failure to record it must not change what the editor does.
   */
  async emitFocus(relativePath: string): Promise<boolean> {
    const body = {
      kind: "file_focused",
      source: "vscode",
      payload: { path: relativePath },
      // M6.0's partial unique index collapses repeats of this key while the
      // first is still unprocessed, so switching between two tabs repeatedly
      // does not queue a unit of work per switch.
      dedupe_key: relativePath,
    };
    const response = await this.send("/events", { method: "POST", body });
    if (response.error !== undefined) {
      this.log(`focus not recorded: ${response.error}`);
      return false;
    }
    return true;
  }

  /**
   * Context for a focus, polling briefly while the server builds it.
   *
   * Returns `ready: false` rather than an error when the build is still running,
   * because "not yet" and "went wrong" need different faces in the panel and
   * collapsing them would make a working system look broken for its first
   * second.
   */
  async fetchContext(focus: string): Promise<ContextResult> {
    const started = Date.now();
    const query = `?focus=${encodeURIComponent(focus)}&token_budget=${this.tokenBudget}`;

    for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
      const response = await this.send(`/context${query}`, { method: "GET" });
      if (response.error !== undefined) {
        return {
          focus,
          ready: false,
          items: [],
          tokensUsed: 0,
          tokenBudget: this.tokenBudget,
          error: response.error,
          elapsedMs: Date.now() - started,
        };
      }

      const body = response.body as {
        ready?: boolean;
        items?: ContextItem[];
        tokens_used?: number;
        token_budget?: number;
      };
      if (body?.ready) {
        return {
          focus,
          ready: true,
          items: Array.isArray(body.items) ? body.items : [],
          tokensUsed: body.tokens_used ?? 0,
          tokenBudget: body.token_budget ?? this.tokenBudget,
          elapsedMs: Date.now() - started,
        };
      }
      // Still building. The last attempt does not sleep, or the caller waits
      // an interval it will never use the result of.
      if (attempt < POLL_ATTEMPTS - 1) {
        await this.sleep(POLL_INTERVAL_MS);
      }
    }

    return {
      focus,
      ready: false,
      items: [],
      tokensUsed: 0,
      tokenBudget: this.tokenBudget,
      elapsedMs: Date.now() - started,
    };
  }

  /**
   * One request, with every failure turned into a value.
   *
   * `AbortSignal.timeout` rather than a race with a timer, so a slow response
   * releases the socket instead of leaving it open behind a promise nobody is
   * waiting on any more.
   */
  private async send(
    path: string,
    options: { method: string; body?: unknown },
  ): Promise<{ body?: unknown; error?: string }> {
    if (this.failures > 0) {
      // Linear, capped, and short. Long enough that a stopped stack does not
      // produce a request per keystroke; short enough that starting the stack
      // back up is noticed within a few seconds rather than needing a reload.
      await this.sleep(Math.min(this.failures, 5) * 400);
    }

    try {
      const response = await this.fetchImpl(`${this.apiUrl}${path}`, {
        method: options.method,
        headers: options.body ? { "Content-Type": "application/json" } : undefined,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });

      if (!response.ok && response.status !== 202) {
        this.failures += 1;
        return { error: `api returned ${response.status}` };
      }

      this.failures = 0;
      // A body that is not JSON is a proxy or a captive portal answering
      // instead of the API. Caught here rather than at the call site, because
      // every call site would have to catch it identically.
      try {
        return { body: await response.json() };
      } catch {
        return { error: "api returned a body that is not json" };
      }
    } catch (caught) {
      this.failures += 1;
      const message = caught instanceof Error ? caught.message : String(caught);
      return { error: message };
    }
  }
}
