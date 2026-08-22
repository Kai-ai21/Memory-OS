/**
 * The provider and the one line it renders.
 *
 * Bottom right, above everything, and never more than one — see `lib/toast`
 * for the rules and the arguments for them.
 *
 * `role="status"` with `aria-live="polite"`, not `alert`: this is a
 * confirmation that something worked, and `alert` interrupts whatever a screen
 * reader is currently saying. Errors are the ones that interrupt, and errors do
 * not come through here.
 *
 * The entry animation is a 150ms fade and rise, which the global
 * `prefers-reduced-motion` rule in `index.css` reduces to nothing — the toast
 * still appears, it simply does not travel.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Undo2, X } from "lucide-react";

import { TOAST_MS, TOAST_UNDO_MS, ToastContext, type Toast } from "../lib/toast";

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toast, setToast] = useState<Toast | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const nextId = useRef(0);

  const dismiss = useCallback(() => {
    clearTimeout(timer.current);
    setToast(null);
  }, []);

  const show = useCallback((message: string, options?: { undo?: () => void }) => {
    // The previous timer dies with the previous toast. Without this, a toast
    // shown at t=0 and replaced at t=2.4s takes the replacement down with it a
    // tenth of a second later.
    clearTimeout(timer.current);
    nextId.current += 1;
    const id = nextId.current;
    setToast({ id, message, undo: options?.undo });
    timer.current = setTimeout(
      () => setToast((current) => (current?.id === id ? null : current)),
      options?.undo ? TOAST_UNDO_MS : TOAST_MS,
    );
  }, []);

  useEffect(() => () => clearTimeout(timer.current), []);

  const api = useMemo(() => ({ show, dismiss }), [show, dismiss]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <Toaster toast={toast} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

function Toaster({ toast, onDismiss }: { toast: Toast | null; onDismiss: () => void }) {
  return (
    /* The region is always mounted so a screen reader has something to
       announce *into*. A live region created at the same moment its content
       arrives is a region most assistive technology does not read. */
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed right-4 bottom-4 z-50 flex justify-end"
      data-testid="toast-region"
    >
      {toast ? (
        <div
          key={toast.id}
          data-testid="toast"
          className="panel popover pointer-events-auto flex max-w-sm items-center gap-3 py-2 pr-2 pl-3 motion-safe:animate-[toast-in_150ms_var(--ease-out)]"
        >
          <span className="meta text-ink">{toast.message}</span>
          {toast.undo ? (
            <button
              type="button"
              data-testid="toast-undo"
              className="meta inline-flex items-center gap-1 text-accent hover:underline"
              onClick={() => {
                toast.undo?.();
                onDismiss();
              }}
            >
              <Undo2 size={12} strokeWidth={2} />
              undo
            </button>
          ) : null}
          <button
            type="button"
            className="shrink-0 rounded-sm p-0.5 text-ink-3 hover:text-ink"
            aria-label="Dismiss"
            onClick={onDismiss}
          >
            <X size={12} strokeWidth={2} />
          </button>
        </div>
      ) : null}
    </div>
  );
}
