/**
 * The panel's HTML, built as a pure function of the result.
 *
 * Separated from `extension.ts` for the same reason `client.ts` is: it does not
 * import `vscode`, so what the panel says in each state can be checked without
 * an editor host. The three states that matter are the two nobody tests —
 * loading and failed — and getting either wrong is what makes a tool feel
 * broken when it is merely waiting.
 *
 * Everything is escaped. The text in a context item is arbitrary content from
 * the corpus — source files, READMEs, decisions somebody typed — and a webview
 * that interpolated it raw would execute whatever a file in your own repository
 * happened to contain.
 */

import type { ContextItem, ContextResult, SurfacedItem } from "./client";

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Where this item lives in the web UI, or null when it has no page.
 *
 * A decision and a memory have different routes, and exactly one of the two ids
 * is set — the category is not consulted, because the id is the fact and the
 * category is a label put on it by the selection layer.
 */
export function webLinkFor(item: ContextItem, webUrl: string): string | null {
  const base = webUrl.replace(/\/+$/, "");
  if (item.memory_id) {
    return `${base}/memory/${item.memory_id}`;
  }
  if (item.decision_id) {
    return `${base}/decisions/${item.decision_id}`;
  }
  return null;
}

export interface RenderOptions {
  webUrl: string;
  /** Shown in the footer. The milestone's requirement is a number, not a feel. */
  lastLatencyMs?: number;
  /**
   * What the gate volunteered for this focus and nobody has judged yet.
   *
   * Almost always empty, and that is the feature working. The panel is a pull
   * surface — it shows this file's context because you opened the file — and
   * this is the one part of it that says something without being asked.
   */
  surfaced?: SurfacedItem[];
}

/**
 * A `command:` link, which is how this panel gets a button without a script.
 *
 * The webview keeps `enableScripts: false` — the excerpts in it are arbitrary
 * text from your own corpus, and a panel that ran script to collect two clicks
 * would be a much larger surface than two clicks are worth. Command URIs are
 * the mechanism VS Code provides for exactly this, and `enableCommandUris` is
 * scoped to the two commands named here.
 */
export function commandLink(command: string, ...args: unknown[]): string {
  return `command:${command}?${encodeURIComponent(JSON.stringify(args))}`;
}

export function renderPanel(
  focus: string | null,
  result: ContextResult | null,
  options: RenderOptions,
): string {
  const body = renderBody(focus, result, options);
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; img-src data:;">
<style>
  body { font-family: var(--vscode-font-family); font-size: 12px;
         color: var(--vscode-foreground); padding: 0 8px 16px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
       color: var(--vscode-descriptionForeground); font-weight: 400;
       border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 4px; }
  .item { border-bottom: 1px solid var(--vscode-panel-border); padding: 8px 0; }
  .title { font-weight: 600; }
  .meta { color: var(--vscode-descriptionForeground); font-size: 11px;
          margin: 2px 0 4px; }
  .excerpt { white-space: pre-wrap; font-family: var(--vscode-editor-font-family);
             font-size: 11px; max-height: 6.5em; overflow: hidden;
             color: var(--vscode-descriptionForeground); }
  .quiet { color: var(--vscode-descriptionForeground); padding: 12px 0; }
  .tag { border: 1px solid var(--vscode-panel-border); padding: 0 4px;
         margin-right: 4px; border-radius: 2px; }
  /* A left rule rather than a coloured box. The panel is already open; this
     needs to be findable, not attention-grabbing. */
  .surfaced { border-left: 2px solid var(--vscode-textLink-foreground);
              padding: 6px 0 6px 8px; margin: 6px 0; }
  .verdict { margin-top: 4px; }
  .verdict a { margin-right: 10px; }
  a { color: var(--vscode-textLink-foreground); }
  footer { color: var(--vscode-descriptionForeground); font-size: 11px;
           padding-top: 8px; }
</style>
</head>
<body>${body}</body>
</html>`;
}

function renderBody(
  focus: string | null,
  result: ContextResult | null,
  options: RenderOptions,
): string {
  if (!focus) {
    return `<p class="quiet">No file open.</p>`;
  }

  const heading = `<h2>${escapeHtml(focus)}</h2>`;
  const volunteered = renderSurfaced(options.surfaced ?? []);

  // Failure first, and it is a line rather than a banner. The API being stopped
  // is the normal state of a laptop; a red box every time you open the editor
  // before the stack is a red box you learn to ignore, and then you ignore the
  // real one too.
  if (result?.error) {
    return `${heading}<p class="quiet">Not available — ${escapeHtml(
      result.error,
    )}. It will try again when you switch files.</p>`;
  }

  if (!result || (!result.ready && result.items.length === 0)) {
    // "Assembling" and not a spinner: the work is happening on a worker and
    // will still be there when this panel is next opened, so the honest word is
    // what is happening rather than an animation implying a wait.
    return `${heading}${volunteered}<p class="quiet">Assembling context…</p>`;
  }

  if (result.items.length === 0) {
    return `${heading}${volunteered}<p class="quiet">Nothing in the corpus relates to this file yet.</p>`;
  }

  const items = result.items.map((item) => renderItem(item, options)).join("");
  const latency =
    options.lastLatencyMs === undefined
      ? ""
      : ` · ${options.lastLatencyMs}ms`;
  return `${heading}${volunteered}${items}<footer>${result.items.length} items · ${
    result.tokensUsed
  }/${result.tokenBudget} tokens${latency}</footer>`;
}

/**
 * The one part of this panel that says something nobody asked for.
 *
 * **It is a strip at the top of a panel you already opened, and that is the
 * loudest this extension is allowed to be.** No notification, no modal, no
 * badge, no sound. M6.2 said "M6.3 is where anything is allowed to interrupt";
 * this is what was allowed, and the reason it is this small is that the panel
 * is already open on the file — a toast would be a second claim on attention
 * for something already on screen.
 *
 * The reason the gate gave is rendered beside the item, not behind a tooltip.
 * It is what lets a reader decide the interruption was fair, and a reader who
 * cannot do that either trusts everything or nothing.
 */
function renderSurfaced(items: SurfacedItem[]): string {
  if (items.length === 0) {
    return "";
  }
  return items
    .map(
      (item) => `<div class="surfaced">
  <div class="meta"><span class="tag">surfaced</span>${escapeHtml(
    item.explanation,
  )}</div>
  <div class="title">${escapeHtml(item.top_title ?? item.focus)}</div>
  <div class="verdict">
    <a href="${escapeHtml(commandLink("memoryos.markUseful", item.id))}">useful</a>
    <a href="${escapeHtml(commandLink("memoryos.dismiss", item.id))}">dismiss</a>
  </div>
</div>`,
    )
    .join("");
}

function renderItem(item: ContextItem, options: RenderOptions): string {
  const link = webLinkFor(item, options.webUrl);
  const title = escapeHtml(item.title);
  const heading = link
    ? `<a href="${escapeHtml(link)}">${title}</a>`
    : title;

  // The routes are on screen rather than in a tooltip. They are the only thing
  // that lets a reader judge an item instead of taking it: two sources means two
  // independent routes agreed, and one means it scraped in on a single mention.
  const routes = Object.entries(item.sources)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([source, rank]) => `<span class="tag">${escapeHtml(source)} #${rank}</span>`)
    .join("");

  return `<div class="item">
  <div class="title">${heading}</div>
  <div class="meta"><span class="tag">${escapeHtml(item.category)}</span>${routes}</div>
  <div class="excerpt">${escapeHtml(item.text.slice(0, 400))}</div>
</div>`;
}
