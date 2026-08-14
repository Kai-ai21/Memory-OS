/**
 * The extension's failure behaviour, with `fetch` stubbed.
 *
 * One property, tested six ways: **nothing here throws.** A dev tool that
 * throws while you are trying to work is uninstalled the same day, and the API
 * being stopped is the normal state of a laptop rather than an exception — so
 * every way the server can fail has to resolve to a value the panel can render
 * quietly.
 *
 * Run with `npm test`, which needs only node. No editor host is involved,
 * because none of this imports `vscode` — that separation is what makes the
 * failure paths testable at all.
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MemoryOsClient } from "../client";
import { renderPanel, webLinkFor, escapeHtml } from "../panel";

const noSleep = async () => undefined;

function clientWith(fetchImpl: typeof fetch): MemoryOsClient {
  return new MemoryOsClient({
    apiUrl: "http://localhost:8000",
    fetchImpl,
    sleep: noSleep,
  });
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("the client does not throw when the api misbehaves", () => {
  it("survives the api being unreachable", async () => {
    const client = clientWith(async () => {
      throw new TypeError("fetch failed");
    });

    const result = await client.fetchContext("src/a.py");

    assert.equal(result.ready, false);
    assert.equal(result.items.length, 0);
    assert.match(result.error ?? "", /fetch failed/);
  });

  it("survives a 500", async () => {
    const client = clientWith(async () => jsonResponse({ detail: "boom" }, 500));

    const result = await client.fetchContext("src/a.py");

    assert.equal(result.ready, false);
    assert.equal(result.error, "api returned 500");
  });

  it("survives a body that is not json", async () => {
    // A captive portal or a proxy answering instead of the API. The status is
    // 200 and the body is HTML, which is the one failure a status check misses.
    const client = clientWith(
      async () => new Response("<html>hello</html>", { status: 200 }),
    );

    const result = await client.fetchContext("src/a.py");

    assert.equal(result.ready, false);
    assert.equal(result.error, "api returned a body that is not json");
  });

  it("survives a rate limit without treating it as data", async () => {
    const client = clientWith(async () => jsonResponse({ detail: "slow down" }, 429));

    const result = await client.fetchContext("src/a.py");

    assert.equal(result.ready, false);
    assert.match(result.error ?? "", /429/);
  });

  it("reports a failed focus rather than raising", async () => {
    const client = clientWith(async () => {
      throw new Error("connection refused");
    });

    assert.equal(await client.emitFocus("src/a.py"), false);
  });
});

describe("building is not an error", () => {
  it("treats 202 as loading and stops polling rather than hanging", async () => {
    let calls = 0;
    const client = clientWith(async () => {
      calls += 1;
      return jsonResponse({ focus: "src/a.py", ready: false, items: [] }, 202);
    });

    const result = await client.fetchContext("src/a.py");

    assert.equal(result.ready, false);
    // No error: the server is working on it, and the build continues on the
    // worker whether or not this client waits for it.
    assert.equal(result.error, undefined);
    assert.ok(calls > 1, "should poll more than once");
    assert.ok(calls <= 4, "should give up rather than poll forever");
  });

  it("returns the context as soon as a poll finds it ready", async () => {
    let calls = 0;
    const client = clientWith(async () => {
      calls += 1;
      if (calls === 1) {
        return jsonResponse({ ready: false, items: [] }, 202);
      }
      return jsonResponse({
        focus: "src/a.py",
        ready: true,
        tokens_used: 120,
        token_budget: 4000,
        items: [
          {
            position: 1,
            title: "self::src/a.py",
            category: "code",
            text: "def handler():",
            tokens: 120,
            sources: { retrieval: 3, temporal: 1 },
            memory_id: "11111111-1111-7111-8111-111111111111",
            decision_id: null,
            external_key: "src/a.py",
          },
        ],
      });
    });

    const result = await client.fetchContext("src/a.py");

    assert.equal(result.ready, true);
    assert.equal(result.items.length, 1);
    assert.equal(calls, 2);
  });
});

describe("the panel says which state it is in", () => {
  const item = {
    position: 1,
    title: "self::src/a.py",
    category: "code",
    text: "def handler():",
    tokens: 120,
    sources: { retrieval: 3, temporal: 1 },
    memory_id: "11111111-1111-7111-8111-111111111111",
    decision_id: null,
    external_key: "src/a.py",
  };
  const options = { webUrl: "http://localhost:5173" };

  it("renders a failure as a quiet line, never as an alarm", () => {
    const html = renderPanel(
      "src/a.py",
      {
        focus: "src/a.py",
        ready: false,
        items: [],
        tokensUsed: 0,
        tokenBudget: 4000,
        error: "fetch failed",
        elapsedMs: 3,
      },
      options,
    );

    assert.match(html, /Not available/);
    assert.match(html, /try again when you switch files/);
    // The distinction the whole extension turns on: nothing that interrupts.
    assert.doesNotMatch(html, /<script/);
  });

  it("distinguishes building from empty", () => {
    const building = renderPanel(
      "src/a.py",
      {
        focus: "src/a.py",
        ready: false,
        items: [],
        tokensUsed: 0,
        tokenBudget: 4000,
        elapsedMs: 3,
      },
      options,
    );
    const empty = renderPanel(
      "src/a.py",
      {
        focus: "src/a.py",
        ready: true,
        items: [],
        tokensUsed: 0,
        tokenBudget: 4000,
        elapsedMs: 3,
      },
      options,
    );

    assert.match(building, /Assembling/);
    assert.match(empty, /Nothing in the corpus/);
  });

  it("shows the routes that put an item where it is", () => {
    const html = renderPanel(
      "src/a.py",
      {
        focus: "src/a.py",
        ready: true,
        items: [item],
        tokensUsed: 120,
        tokenBudget: 4000,
        elapsedMs: 40,
      },
      options,
    );

    assert.match(html, /retrieval #3/);
    assert.match(html, /temporal #1/);
    assert.match(html, /memory\/11111111-1111-7111-8111-111111111111/);
  });

  it("escapes corpus text rather than interpolating it", () => {
    // The text is arbitrary content from your own repository. A webview that
    // interpolated it raw would run whatever a file happened to contain.
    const hostile = { ...item, text: "<script>alert(1)</script>", title: "<img>" };
    const html = renderPanel(
      "src/a.py",
      {
        focus: "src/a.py",
        ready: true,
        items: [hostile],
        tokensUsed: 1,
        tokenBudget: 4000,
        elapsedMs: 1,
      },
      options,
    );

    assert.doesNotMatch(html, /<script>alert/);
    assert.match(html, /&lt;script&gt;/);
    assert.equal(escapeHtml("<&>"), "&lt;&amp;&gt;");
  });

  it("links a decision to its own page and an item with neither to nothing", () => {
    const decision = { ...item, memory_id: null, decision_id: "abc" };
    assert.equal(
      webLinkFor(decision, "http://localhost:5173"),
      "http://localhost:5173/decisions/abc",
    );
    assert.equal(
      webLinkFor({ ...item, memory_id: null, decision_id: null }, "http://x"),
      null,
    );
  });
});

describe("what the panel volunteers", () => {
  const surfaced = {
    id: "9f1c2d3e-0000-7000-8000-000000000001",
    focus: "src/a.py",
    reason: "cleared",
    explanation: "two independent routes agreed on something you do not already have open",
    score: 0.0331,
    threshold: 0.0295,
    top_title: "self::src/b.py",
    item_count: 9,
    trigger_kind: "file_focused",
  };
  const options = { webUrl: "http://localhost:5173" };
  const item = {
    position: 1,
    title: "self::src/a.py",
    category: "code",
    text: "def handler():",
    tokens: 120,
    sources: { retrieval: 3, temporal: 1 },
    memory_id: "11111111-1111-7111-8111-111111111111",
    decision_id: null,
    external_key: "src/a.py",
  };
  const result = {
    focus: "src/a.py",
    ready: true,
    items: [item],
    tokensUsed: 120,
    tokenBudget: 4000,
    elapsedMs: 40,
  };

  it("renders nothing at all when nothing was surfaced", () => {
    // The common case by a wide margin, and the feature working. A heading
    // saying "nothing surfaced" would be the panel talking about its own
    // silence, which is a way of not being silent.
    const html = renderPanel("src/a.py", result, { ...options, surfaced: [] });

    // The strip, not the stylesheet — `.surfaced` is a rule in every render.
    assert.doesNotMatch(html, /<div class="surfaced">/);
    assert.doesNotMatch(html, /command:/);
  });

  it("shows the gate's reason beside what it volunteered", () => {
    const html = renderPanel("src/a.py", result, { ...options, surfaced: [surfaced] });

    assert.match(html, /two independent routes agreed/);
    assert.match(html, /self::src\/b\.py/);
    // Both verdicts, as command links rather than script.
    assert.match(html, /command:memoryos\.markUseful/);
    assert.match(html, /command:memoryos\.dismiss/);
  });

  it("keeps the strip when the context itself is still building", () => {
    // The two arrive by different routes and the gate decided on a context that
    // was cached at the time. A surfacing that vanished while the panel
    // reassembled would flicker on every tab change.
    const html = renderPanel("src/a.py", null, { ...options, surfaced: [surfaced] });

    assert.match(html, /Assembling/);
    assert.match(html, /command:memoryos\.dismiss/);
  });

  it("treats every failure to fetch a surfacing as nothing surfaced", async () => {
    // Silence is this feature's default and its correct behaviour, so no
    // failure here deserves to be visible. An error line about not being able
    // to fetch what was not going to be shown is noise about the absence of
    // noise.
    for (const fetchImpl of [
      async () => {
        throw new TypeError("fetch failed");
      },
      async () => jsonResponse({ detail: "nope" }, 500),
      async () => new Response("<html>captive portal</html>", { status: 200 }),
      async () => jsonResponse({ not: "an array" }),
    ]) {
      const client = clientWith(fetchImpl as typeof fetch);
      assert.deepEqual(await client.fetchSurfaced("src/a.py"), []);
    }
  });

  it("reads a 204 as the verdict being recorded", async () => {
    // The verdict endpoints answer 204, which has no body — and `.json()` on an
    // empty one throws. Reported as a failure, a successful dismissal would
    // look like a lost one, and the reader would click it again.
    const client = clientWith(async () => new Response(null, { status: 204 }));

    assert.equal(await client.rate(surfaced.id, false), true);
  });

  it("reports a refused verdict as not recorded rather than throwing", async () => {
    const client = clientWith(async () => jsonResponse({ detail: "already rated" }, 409));

    assert.equal(await client.rate(surfaced.id, true), false);
  });
});
