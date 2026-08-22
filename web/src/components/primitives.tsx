/**
 * The small shared pieces. Presentational only — no data fetching, no routing.
 *
 * There is no component library here on purpose: this is eight components, and a
 * library would cost more in configuration and lock-in than it saves. These are
 * thin wrappers over the CSS component classes in `index.css`, which exist so the
 * visual language stays in one file rather than smeared across every element.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useToast } from "../lib/toast";

/** A labelled metadata pair. The workhorse of the whole interface. */
export function Meta({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="inline-flex items-baseline gap-1.5">
      <span className="meta-label">{label}</span>
      <span className="meta text-ink">{children}</span>
    </span>
  );
}

/** A kind, a chunker version, a model — a short factual tag, not a status pill. */
export function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="meta border border-rule px-1 py-px text-ink-3 uppercase tracking-wider">
      {children}
    </span>
  );
}

/**
 * A section heading. Rules rather than cards: the structural device throughout is
 * a ruled sheet, and a heading is where one rule sits.
 */
export function SectionHeading({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between border-b border-rule-strong pb-1">
      <h2 className="meta-label text-ink-2">{children}</h2>
      {right ? <div className="meta">{right}</div> : null}
    </div>
  );
}

/**
 * The empty state. Says what to do next rather than only that there is nothing —
 * "no results" with no suggestion is a dead end, and the two things that actually
 * help here are widening the filters and raising k.
 */
export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-rule-strong bg-surface p-6" data-testid="empty">
      <p className="meta-label text-ink-2">{title}</p>
      {children ? <div className="meta mt-2 max-w-prose leading-relaxed">{children}</div> : null}
    </div>
  );
}

/** An error, with the failure named. A spinner that never stops is worse. */
export function Failure({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  const isNetwork = error instanceof Error && error.name === "NetworkError";
  return (
    <div className="rounded-md border border-rule border-l-2 border-l-deny bg-surface p-4" role="alert" data-testid="error">
      <p className="meta-label text-deny">{isNetwork ? "cannot reach the api" : "request failed"}</p>
      <p className="meta mt-1 text-ink">{message}</p>
      {isNetwork ? (
        <p className="meta mt-2 text-ink-2">
          Start it with <code className="kbd">make dev</code>, or set{" "}
          <code className="kbd">VITE_API_URL</code> if it is somewhere else.
        </p>
      ) : null}
    </div>
  );
}

/**
 * A loading state that reserves the right amount of room.
 *
 * Skeleton rules rather than a spinner: the layout does not jump when content
 * arrives, and on a local API most requests resolve before this is even seen.
 *
 * **M9.10 gave the bars the shimmer and moved them back to `surface-tint`.**
 * The tint was tried before and abandoned because a static tint block laid on
 * white is a four-point difference in luminance and effectively invisible — a
 * loading state nobody can see is a blank screen with extra steps. What makes
 * it work now is that it is no longer static: the travelling highlight is what
 * the eye catches, and it is legible at a contrast a flat fill is not. That
 * buys back the reason to prefer the tint, which is that `rule` is the colour
 * of the borders these bars sit between, so a skeleton drawn in it read as a
 * stack of thick dividers rather than as absent text.
 */
export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3" data-testid="loading" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading</span>
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="border-b border-rule pb-3">
          <div className="skeleton h-3 w-1/3" />
          <div className="skeleton mt-2 h-3 w-full" />
          <div className="skeleton mt-1 h-3 w-4/5" />
        </div>
      ))}
    </div>
  );
}

/**
 * One skeleton block, for a layout the generic `Loading` stack does not match.
 *
 * Takes its size from the caller because the whole value of a skeleton is that
 * it is the shape of the thing that is coming — a component that picked its own
 * dimensions would be a spinner with square corners.
 */
export function Skeleton({ className = "" }: { className?: string }) {
  return <span className={`skeleton block ${className}`} aria-hidden />;
}

/**
 * The spinner, which belongs on a button and almost nowhere else.
 *
 * See `.spinner` in `index.css` for why. `aria-hidden` because the button it
 * sits in carries `aria-busy` and the accessible name is unchanged — a
 * screen reader announcing "loading" beside a button already announced as busy
 * is the same fact twice.
 */
export function Spinner({ className = "" }: { className?: string }) {
  return <span className={`spinner ${className}`} aria-hidden />;
}

/**
 * A button that says when it is working.
 *
 * **The label does not change and this is the point.** The obvious build swaps
 * "save the correction" for "correcting…", and it costs two things: the button
 * changes width mid-click, so a second press lands somewhere else on the page,
 * and the reader loses the name of the thing they just asked for at exactly the
 * moment they are wondering whether it happened. The spinner takes the icon's
 * place instead — same box, same words, one glyph swapped — so the only thing
 * that changed is the one thing being reported.
 *
 * `disabled` while busy, which is the half that actually prevents damage: a
 * slow send with a live button is a double send, and this is the most common
 * way a queue ends up with two of everything.
 *
 * `type="button"` by default. A button inside a form with no explicit type is a
 * submit button, which is how a "remove" control ends up posting the form.
 */
export function Button({
  loading = false,
  disabled = false,
  icon,
  children,
  className = "",
  type = "button",
  ...rest
}: {
  /** Async action in flight: disables, and swaps the icon for a spinner. */
  loading?: boolean;
  /** Shown at rest, and replaced by the spinner while `loading`. */
  icon?: ReactNode;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      data-loading={loading || undefined}
      className={className}
    >
      {loading ? <Spinner /> : icon}
      {children}
    </button>
  );
}

/** How long the tick stays before the clipboard icon comes back. */
const COPIED_MS = 1500;

/**
 * Copy one mono value — a path, an id, a hash.
 *
 * **Confirmed by the icon becoming a tick, not by a toast.** A toast for a copy
 * is a notification about something that already succeeded, appearing somewhere
 * other than where you were looking, and having to be dismissed. The
 * confirmation belongs on the control that was pressed.
 *
 * Hidden until the row it sits in is hovered or something inside that row has
 * keyboard focus — `group-hover` and `group-focus-within`, so the parent marks
 * itself `group` and nothing here needs to know what a row is. It stays a real
 * focusable button at all times: `opacity-0` rather than `hidden`, because a
 * control that is not in the tab order is a control a keyboard cannot reach,
 * and `focus-visible:opacity-100` is what makes it appear when tabbed to.
 *
 * `navigator.clipboard` can reject — it is permission-gated and unavailable on
 * insecure origins — and a failure leaves the icon alone rather than claiming a
 * copy that did not happen.
 */
export function CopyButton({
  value,
  label,
  className = "",
}: {
  value: string;
  /** What was copied, for the tooltip and the accessible name. */
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const toast = useToast();

  // A component unmounted inside the 1.5s window — a row that scrolled out of a
  // virtualised list, a panel closed straight after a copy — would otherwise
  // set state on nothing and warn.
  useEffect(() => () => clearTimeout(timer.current), []);

  const what = label ?? "value";

  return (
    <button
      type="button"
      data-testid="copy"
      className={`inline-flex shrink-0 items-center justify-center rounded-sm p-0.5 text-ink-3 opacity-0 transition-opacity duration-(--dur-state) ease-(--ease-out) hover:text-accent group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 ${className}`}
      aria-label={copied ? `Copied ${what}` : `Copy ${what}`}
      title={copied ? "Copied" : `Copy ${what}`}
      onClick={async (event) => {
        // Inside a row that is itself a link or an expander almost everywhere
        // this appears. Copying must not also navigate.
        event.preventDefault();
        event.stopPropagation();
        try {
          await navigator.clipboard.writeText(value);
        } catch {
          // No clipboard, or permission refused. Say nothing rather than
          // showing a tick for a copy that did not happen.
          return;
        }
        setCopied(true);
        clearTimeout(timer.current);
        timer.current = setTimeout(() => setCopied(false), COPIED_MS);
        /* Both, and they are not redundant. The tick is the confirmation for
           somebody watching the button; the toast is for somebody whose eyes
           had already moved on, and it names *what* was copied, which the tick
           cannot. */
        toast.show(`Copied ${what}`);
      }}
    >
      {copied ? <CheckGlyph /> : <ClipboardGlyph />}
    </button>
  );
}

/* The two glyphs, drawn here rather than imported. They are 12px, they appear
   beside mono at 13px, and they are the two smallest drawings in the
   application — see `components/Icon.tsx` for why this set is hand-drawn. */

function ClipboardGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className="text-affirm">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}
