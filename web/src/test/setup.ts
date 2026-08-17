import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(cleanup);

/**
 * jsdom ships `<dialog>` but not its modal methods.
 *
 * The command palette uses `showModal` deliberately — it is what gives the
 * focus trap, the inert background and the dialog semantics without a
 * dependency — so the gap is in the test environment rather than in the
 * component, and the right fix is to fill it here rather than to write the
 * component around it. This is the minimum that makes `open` truthful and
 * `close` fire its event; it does not emulate the focus trap, which nothing
 * here asserts on.
 */
if (typeof HTMLDialogElement !== "undefined" && !HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
    this.open = true;
  };
  HTMLDialogElement.prototype.show = function show(this: HTMLDialogElement) {
    this.open = true;
  };
  HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
    if (!this.open) return;
    this.open = false;
    this.dispatchEvent(new Event("close"));
  };
}


/**
 * jsdom ships no `EventSource`, and M10.3's live updates use one.
 *
 * A stub rather than a shim: it records the URL it was opened with and lets a
 * test dispatch server events into it by hand, which is the only way to assert
 * what the page does when a background job finishes. It deliberately does *not*
 * emulate reconnection — that is the browser's own behaviour and the reason
 * `EventSource` was chosen over a hand-written reader, so a test asserting on a
 * reimplementation of it would be testing this file.
 */
class StubEventSource extends EventTarget {
  static instances: StubEventSource[] = [];
  readonly url: string;
  closed = false;

  constructor(url: string) {
    super();
    this.url = url;
    StubEventSource.instances.push(this);
  }

  close() {
    this.closed = true;
  }
}

if (typeof EventSource === "undefined") {
  (globalThis as unknown as { EventSource: unknown }).EventSource = StubEventSource;
}

export { StubEventSource };
