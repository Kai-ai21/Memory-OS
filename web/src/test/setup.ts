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
