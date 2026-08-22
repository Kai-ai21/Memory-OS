/**
 * What you searched for last, under an empty box.
 *
 * **Only when the box is empty.** That is the moment the list can help and the
 * only moment it is not in the way — under a query in progress it would be a
 * column of near misses competing for attention with the results being read.
 *
 * Each row runs the query again; the `×` forgets one. The `×` is what makes
 * the list usable over time: one mistyped query stays at the top of a
 * ten-item list for a week otherwise, and a history you cannot correct is one
 * you stop reading.
 *
 * Rendered from state seeded at mount rather than read on every render, so
 * removing an entry does not need a re-read and the list does not flicker.
 */

import { useState } from "react";
import { X } from "lucide-react";

import { forgetQuery, readHistory } from "../../lib/history";

export function SearchHistory({ onPick }: { onPick: (query: string) => void }) {
  const [queries, setQueries] = useState(readHistory);

  // Nothing to show and nothing to say about it. An empty state here would be
  // a box explaining that you have not searched yet, directly beneath a box
  // inviting you to search.
  if (queries.length === 0) return null;

  return (
    <div className="flex flex-col gap-1" data-testid="search-history">
      <p className="meta-label text-ink-3">recent searches</p>
      <ul className="flex flex-col">
        {queries.map((query) => (
          <li
            key={query}
            className="group flex items-baseline gap-2 border-b border-rule/60 last:border-b-0"
          >
            <button
              type="button"
              className="min-w-0 flex-1 truncate py-1 text-left font-mono text-xs text-ink-2 hover:text-accent"
              onClick={() => onPick(query)}
            >
              {query}
            </button>
            <button
              type="button"
              aria-label={`Forget "${query}"`}
              title="Forget this search"
              className="shrink-0 rounded-sm p-0.5 text-ink-3 opacity-0 transition-opacity duration-(--dur-state) ease-(--ease-out) hover:text-deny group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100"
              onClick={() => setQueries(forgetQuery(query))}
            >
              <X size={12} strokeWidth={2} />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
