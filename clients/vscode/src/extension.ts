/**
 * The editor half of Phase 6: three features now, and no fourth one.
 *
 * On the active editor changing, POST `file_focused` with the path relative to
 * the workspace, and ask for that file's context to render in a sidebar. M6.3
 * adds the third: anything the gate volunteered for this file appears at the top
 * of that panel with two links that judge it.
 *
 * **Still no notifications, no modals, no status bar item and no badge.** M6.2
 * said "M6.3 is where anything is allowed to interrupt", and what was allowed is
 * a strip in a panel the reader already has open. A toast would be a second
 * claim on attention for something already on screen, and the first one nobody
 * wanted is the one that gets the extension uninstalled.
 *
 * **The relative path matters more than it looks.** It is the corpus's
 * `external_key`, so it is what M6.1's by-name source matches on — the only one
 * of the four that can find the file you are actually looking at. An absolute
 * path would produce a focus that matches nothing by name, and the panel would
 * quietly show a worse answer with no sign that anything was wrong.
 *
 * Everything that can fail lives in `client.ts`, which never throws. Nothing in
 * this file catches, because there is nothing to catch.
 */

import * as vscode from "vscode";

import {
  MemoryOsClient,
  type ContextResult,
  type SurfacedItem,
} from "./client";
import { renderPanel } from "./panel";

/**
 * How long to wait after a tab change before doing anything.
 *
 * Cycling through files with the keyboard produces an editor change per file,
 * and assembling for each would queue work for every file passed *through*
 * rather than the one landed on. Shorter than the watcher's thirty seconds
 * because this is a person's attention rather than a save burst — a third of a
 * second is below what anybody notices and above what a keyboard repeat
 * produces.
 */
const SETTLE_MS = 350;

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel("Memory OS");
  const provider = new ContextViewProvider(output);

  context.subscriptions.push(
    output,
    vscode.window.registerWebviewViewProvider("memoryos.context", provider),
    // The two verdicts, as commands rather than as script inside the webview.
    // The panel keeps `enableScripts: false` — its excerpts are arbitrary text
    // from the reader's own corpus — and `command:` links are the mechanism VS
    // Code provides for collecting a click without one.
    vscode.commands.registerCommand("memoryos.dismiss", (id: string) =>
      provider.rate(id, false),
    ),
    vscode.commands.registerCommand("memoryos.markUseful", (id: string) =>
      provider.rate(id, true),
    ),
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      provider.focusOn(editor);
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("memoryos")) {
        provider.reload();
      }
    }),
  );

  // The editor that is already open when the extension activates. Without this
  // the panel is empty until you switch tabs, which reads as broken on the one
  // occasion a person is most likely to be looking at it.
  provider.focusOn(vscode.window.activeTextEditor);
  output.appendLine("memoryos: watching the active editor. Local only, no auth.");
}

export function deactivate(): void {
  // Nothing to do. Everything owned is in `context.subscriptions`.
}

class ContextViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private client: MemoryOsClient;
  private webUrl: string;
  private emitEvents: boolean;
  private focus: string | null = null;
  private result: ContextResult | null = null;
  private surfaced: SurfacedItem[] = [];
  private settle?: ReturnType<typeof setTimeout>;
  /**
   * Which focus the in-flight request is for.
   *
   * A slow response for a file you have already navigated away from must not
   * overwrite the panel — the reader would see context for a file they are not
   * looking at, with nothing on screen saying so.
   */
  private inFlight: string | null = null;

  constructor(private readonly output: vscode.OutputChannel) {
    const config = vscode.workspace.getConfiguration("memoryos");
    this.webUrl = config.get<string>("webUrl") ?? "http://localhost:5173";
    this.emitEvents = config.get<boolean>("emitEvents") ?? true;
    this.client = this.buildClient(config);
  }

  reload(): void {
    const config = vscode.workspace.getConfiguration("memoryos");
    this.webUrl = config.get<string>("webUrl") ?? "http://localhost:5173";
    this.emitEvents = config.get<boolean>("emitEvents") ?? true;
    this.client = this.buildClient(config);
    this.render();
  }

  private buildClient(config: vscode.WorkspaceConfiguration): MemoryOsClient {
    return new MemoryOsClient({
      apiUrl: config.get<string>("apiUrl") ?? "http://localhost:8000",
      tokenBudget: config.get<number>("tokenBudget") ?? 4000,
      // Logged to an output channel, which is the loudest this extension is
      // ever allowed to be. `showErrorMessage` would be a modal in the corner
      // every time the stack is not running.
      log: (message) => this.output.appendLine(message),
    });
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    // No scripts, and command URIs limited to the two this extension owns.
    // `enableCommandUris: true` would allow any command in the editor to be
    // invoked by a link in a page built from corpus text, which is a much
    // larger surface than two verdicts are worth.
    view.webview.options = {
      enableScripts: false,
      enableCommandUris: ["memoryos.dismiss", "memoryos.markUseful"],
    };
    this.render();
  }

  /**
   * Record a verdict and drop the row.
   *
   * Removed from the panel whatever the server says, and the two cases differ
   * only in the output channel. A verdict the API refused is still a reader
   * saying "stop showing me this", and leaving the strip on screen to be
   * technically accurate about a failed POST would be arguing with them.
   */
  async rate(id: string, useful: boolean): Promise<void> {
    const recorded = await this.client.rate(id, useful);
    this.output.appendLine(
      `${useful ? "useful" : "dismissed"} ${id}${recorded ? "" : " (not recorded)"}`,
    );
    this.surfaced = this.surfaced.filter((item) => item.id !== id);
    this.render();
  }

  focusOn(editor: vscode.TextEditor | undefined): void {
    const relative = relativePathOf(editor);
    if (relative === this.focus) {
      return;
    }
    this.focus = relative;
    this.result = null;
    this.surfaced = [];
    this.render();

    if (this.settle) {
      clearTimeout(this.settle);
    }
    if (relative === null) {
      return;
    }
    this.settle = setTimeout(() => void this.load(relative), SETTLE_MS);
  }

  private async load(relative: string): Promise<void> {
    this.inFlight = relative;
    if (this.emitEvents) {
      // Not awaited before the context request: recording that a file was
      // focused is bookkeeping, and making the panel wait on it would put an
      // HTTP round trip in front of the thing the reader is waiting for.
      void this.client.emitFocus(relative);
    }

    const result = await this.client.fetchContext(relative);
    // After the context rather than beside it. The gate decides on a context
    // that already exists, so asking first would reliably ask before there was
    // anything to have decided about — and this is one request against an
    // indexed table rather than something worth racing.
    const surfaced = await this.client.fetchSurfaced(relative);
    if (this.inFlight !== relative || this.focus !== relative) {
      // Navigated away while this was in flight. Dropped rather than rendered.
      return;
    }
    this.result = result;
    this.surfaced = surfaced;
    this.output.appendLine(
      `context ${relative}: ${result.ready ? `${result.items.length} items` : "building"} in ${result.elapsedMs}ms`,
    );
    this.render();
  }

  private render(): void {
    if (!this.view) {
      return;
    }
    this.view.webview.html = renderPanel(this.focus, this.result, {
      webUrl: this.webUrl,
      lastLatencyMs: this.result?.elapsedMs,
      surfaced: this.surfaced,
    });
  }
}

/**
 * The workspace-relative path of an editor, or null when there is not one.
 *
 * `asRelativePath` with `false` so a multi-root workspace does not prefix the
 * folder name — the corpus key is relative to the *source root*, and a source
 * is registered per root rather than per workspace.
 */
export function relativePathOf(
  editor: vscode.TextEditor | undefined,
): string | null {
  if (!editor || editor.document.uri.scheme !== "file") {
    // Output channels, diff views and settings editors are all `TextEditor`s
    // with schemes that are not `file`. Assembling context for
    // `output:extension-output-…` would be a request per log line.
    return null;
  }
  return vscode.workspace.asRelativePath(editor.document.uri, false);
}
