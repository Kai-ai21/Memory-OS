/**
 * The small shared pieces. Presentational only — no data fetching, no routing.
 *
 * There is no component library here on purpose: this is eight components, and a
 * library would cost more in configuration and lock-in than it saves. These are
 * thin wrappers over the CSS component classes in `index.css`, which exist so the
 * visual language stays in one file rather than smeared across every element.
 */

import type { ReactNode } from "react";

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
    <div className="border border-dashed border-rule-strong p-6" data-testid="empty">
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
    <div className="border-l-2 border-deny bg-surface p-4" role="alert" data-testid="error">
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
 */
export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3" data-testid="loading" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading</span>
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="border-b border-rule pb-3">
          <div className="h-3 w-1/3 bg-surface-tint" />
          <div className="mt-2 h-3 w-full bg-surface-tint" />
          <div className="mt-1 h-3 w-4/5 bg-surface-tint" />
        </div>
      ))}
    </div>
  );
}
