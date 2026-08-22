/**
 * Settings — and the whole of it is about this browser.
 *
 * **There is still no settings *page*, and this is not one.** The sidebar note
 * from M9.8 said this application has no server-side preferences to configure
 * and should not invent a page to hold none. That is still true. What M9.11
 * created is different: five pieces of state that live in `localStorage` and
 * belong to this machine — pins, search history, recents, the split width, the
 * density — and every one of them needs somewhere it can be seen and cleared.
 * A feature that quietly accumulates a record of what you searched for, with no
 * surface that admits it exists and no button that removes it, is not a feature
 * anybody should ship.
 *
 * So: a sheet, built exactly like `ShortcutsSheet`, listing what is stored and
 * offering to forget it. Nothing here is sent anywhere, which is the first
 * thing the sheet says.
 *
 * It holds the search history and the recents list at M9.11.4. Pins and the
 * density control arrive with the features that create them, each in its own
 * commit, so that any one of them can be reverted without taking this sheet
 * apart.
 */

import { useEffect, useRef, useState } from "react";

import { clearHistory, readHistory } from "../lib/history";
import { clearRecents, readRecents } from "../lib/recents";

export function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const node = dialog.current;
    if (!node) return;
    if (!node.open) node.showModal();
  }, [open]);

  // Mounted only while open, like the shortcuts sheet — see the note there.
  if (!open) return null;

  return (
    <dialog
      ref={dialog}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === dialog.current) onClose();
      }}
      aria-labelledby="settings-title"
      data-testid="settings-sheet"
      className="panel m-0 w-full max-w-md p-0 text-ink backdrop:bg-scrim sm:mt-[14vh] sm:ml-[max(0px,calc(50%-14rem))]"
    >
      <div className="flex items-baseline justify-between border-b border-rule px-4 py-2.5">
        <h2 id="settings-title" className="meta-label text-ink-2">
          settings
        </h2>
        <button
          type="button"
          className="meta text-ink-3 hover:text-accent"
          onClick={onClose}
          aria-label="Close settings"
        >
          esc
        </button>
      </div>

      <div className="flex flex-col gap-5 px-4 py-4">
        <StoredLocally />
      </div>

      <p className="meta border-t border-rule px-4 py-2 text-ink-3">
        Everything on this screen is stored in this browser and has never been sent to the
        API.
      </p>
    </dialog>
  );
}

/** Everything in `localStorage`, with counts and a way to remove it. */
function StoredLocally() {
  const [history, setHistory] = useState(() => readHistory().length);
  const [recents, setRecents] = useState(() => readRecents().length);

  return (
    <div className="flex flex-col gap-1.5">
      <p className="meta-label text-ink-2">stored in this browser</p>
      <ul className="flex flex-col">
        <StoredRow
          label="search history"
          detail={`${history} ${history === 1 ? "query" : "queries"}`}
          disabled={history === 0}
          onClear={() => {
            clearHistory();
            setHistory(0);
          }}
        />
        <StoredRow
          label="recently opened"
          detail={`${recents} ${recents === 1 ? "item" : "items"}`}
          disabled={recents === 0}
          onClear={() => {
            clearRecents();
            setRecents(0);
          }}
        />
      </ul>
    </div>
  );
}

function StoredRow({
  label,
  detail,
  disabled,
  onClear,
}: {
  label: string;
  detail: string;
  disabled: boolean;
  onClear: () => void;
}) {
  return (
    <li className="flex items-baseline justify-between gap-3 border-b border-rule/60 py-1.5 last:border-b-0">
      <span className="meta text-ink">{label}</span>
      <span className="meta flex-1 text-ink-3">{detail}</span>
      <button
        type="button"
        className="meta text-ink-3 hover:text-deny disabled:text-rule-strong disabled:hover:text-rule-strong"
        disabled={disabled}
        onClick={onClear}
      >
        clear
      </button>
    </li>
  );
}
