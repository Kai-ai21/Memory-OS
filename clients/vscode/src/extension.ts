/**
 * The editor half of M6.2: two features, and no third one.
 *
 * On the active editor changing, POST `file_focused` with the path relative to
 * the workspace. On the same change, ask for that file's context and render it
 * in a sidebar. That is all of it — no commands, no status bar item, no
 * notifications. M6.3 is where anything is allowed to interrupt.
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

import { MemoryOsClient, type ContextResult } from "./client";
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
    view.webview.options = { enableScripts: false };
    this.render();
  }

  focusOn(editor: vscode.TextEditor | undefined): void {
    const relative = relativePathOf(editor);
    if (relative === this.focus) {
      return;
    }
    this.focus = relative;
    this.result = null;
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
    if (this.inFlight !== relative || this.focus !== relative) {
      // Navigated away while this was in flight. Dropped rather than rendered.
      return;
    }
    this.result = result;
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
