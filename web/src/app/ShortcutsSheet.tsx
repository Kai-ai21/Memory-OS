/**
 * `?` — the keyboard model, written down.
 *
 * **A shortcut nobody can discover is a shortcut nobody uses.** This
 * application binds five keys and, until this milestone, named exactly one of
 * them anywhere in the interface: the `⌘K` cap in the sidebar's More menu.
 * The other four were knowable only by reading `Shell.tsx`. That is the normal
 * failure — the bindings get built, they work, and then they are invisible, so
 * the feature is paid for and not delivered.
 *
 * `?` is the convention for this, and the convention is worth following
 * precisely because it is one: somebody who has used any other keyboard-driven
 * application will try it, and finding nothing is what teaches them this one
 * has no shortcuts.
 *
 * **`<dialog showModal>`, the same as the palette.** The focus trap, the inert
 * background, the `aria-modal` semantics and the native `Esc` handler are the
 * platform's. See the header of `CommandPalette.tsx` — the argument is the same
 * one and there is no reason for two modal implementations in one application.
 *
 * The rows come from `SHORTCUTS` below rather than from the handlers that
 * implement them, which is a real seam: this table can drift from `Shell.tsx`.
 * The alternative — deriving the sheet from a registry the shell also reads —
 * is the right build at fifteen shortcuts and is over-built at five, where the
 * two are twenty lines apart in the same directory. `Shell.test.tsx` pins the
 * behaviour of every key listed here, so a binding that disappears fails a test
 * rather than only going stale on this sheet.
 */

import { useEffect, useRef } from "react";

interface Shortcut {
  /** The caps, in press order. Split so each renders in its own `.kbd`. */
  keys: string[];
  what: string;
  /** Where it applies, when that is not "anywhere". */
  scope?: string;
}

/**
 * Every binding this application has.
 *
 * The symbols are the macOS ones because that is what the sidebar already
 * draws and what the `title` on the search button says. On Windows and Linux
 * the shell accepts `Ctrl` for all of these — see the `metaKey || ctrlKey`
 * check in `Shell.tsx` — and rendering both spellings in every row would
 * double the width of the sheet to serve the platform this is not usually run
 * on. The note at the foot says so instead.
 */
const SHORTCUTS: Shortcut[] = [
  { keys: ["⌘", "K"], what: "Open the command palette — jump to any view or memory" },
  { keys: ["/"], what: "Focus the search box", scope: "goes to search where there is none" },
  { keys: ["⌘", "↵"], what: "Send", scope: "in any message or answer box" },
  { keys: ["↵"], what: "Send", scope: "in the chat composer" },
  { keys: ["⇧", "↵"], what: "Start a new line", scope: "in the chat composer" },
  { keys: ["?"], what: "Open this sheet" },
  { keys: ["Esc"], what: "Close whatever is over the page" },
];

export function ShortcutsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const node = dialog.current;
    if (!node) return;
    if (!node.open) node.showModal();
  }, [open]);

  /* **Mounted only while open, unlike the palette.** The palette stays in the
     tree because it holds a fetched list of memories and a query somebody may
     be part-way through typing; there is nothing here but static rows, so
     keeping a closed dialog in the document buys nothing and costs the
     guarantee that exactly one dialog element exists at a time. That guarantee
     is worth having: a closed `<dialog>` is hidden by the UA stylesheet in a
     browser, but it is still an element with `role="dialog"` in the document,
     and anything walking the tree — a test, an assistive technology with its
     own idea of visibility — has to know to skip it. */
  if (!open) return null;

  return (
    <dialog
      ref={dialog}
      // `Esc` fires `cancel` before `close`; both are routed to the same place
      // so the parent's `open` state can never disagree with the element's.
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClose={onClose}
      // The backdrop dismisses, as it does on the palette.
      onClick={(event) => {
        if (event.target === dialog.current) onClose();
      }}
      aria-labelledby="shortcuts-title"
      data-testid="shortcuts-sheet"
      className="panel m-0 w-full max-w-md p-0 text-ink backdrop:bg-scrim sm:mt-[14vh] sm:ml-[max(0px,calc(50%-14rem))]"
    >
      <div className="flex items-baseline justify-between border-b border-rule px-4 py-2.5">
        <h2 id="shortcuts-title" className="meta-label text-ink-2">
          keyboard shortcuts
        </h2>
        <button
          type="button"
          className="meta text-ink-3 hover:text-accent"
          onClick={onClose}
          aria-label="Close shortcuts"
        >
          esc
        </button>
      </div>

      <dl className="flex flex-col px-4 py-2">
        {SHORTCUTS.map((shortcut) => (
          <div
            key={`${shortcut.keys.join("")}-${shortcut.what}`}
            className="flex items-baseline gap-4 border-b border-rule/60 py-2 last:border-b-0"
          >
            {/* The caps first and at a fixed width, so the column of keys lines
                up as a column. A description-first layout puts the thing being
                looked up — the key — at a different x on every row. */}
            <dt className="flex w-20 shrink-0 items-baseline gap-0.5">
              {shortcut.keys.map((key) => (
                <kbd key={key} className="kbd">
                  {key}
                </kbd>
              ))}
            </dt>
            <dd className="flex-1 font-prose text-sm text-ink">
              {shortcut.what}
              {shortcut.scope ? (
                <span className="meta ml-1.5 text-ink-3">{shortcut.scope}</span>
              ) : null}
            </dd>
          </div>
        ))}
      </dl>

      <p className="meta border-t border-rule px-4 py-2 text-ink-3">
        On Windows and Linux, <span className="kbd">Ctrl</span> stands in for{" "}
        <span className="kbd">⌘</span>.
      </p>
    </dialog>
  );
}
