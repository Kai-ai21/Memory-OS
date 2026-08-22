/**
 * Subsequence matching, and a score that puts the obvious answer first.
 *
 * **Why fuzzy at all.** The palette's old matcher was `includes`, which means
 * `dnew` finds nothing and `decisions/new` has to be typed in full — and the
 * whole promise of a command palette is that four characters get you there.
 * Subsequence matching is what every palette worth using does: the typed
 * characters must appear in order, but not adjacently, so `dnw` reaches
 * `decisions/new` and `rslt` reaches `ResultRow`.
 *
 * **Why not a fuzzy-search dependency.** `fuse.js` is 12KB gzipped and is built
 * for ranking prose across weighted fields. This ranks short identifiers —
 * route labels and file paths — where the useful signals are few, cheap, and
 * listed below. The whole implementation is forty lines.
 *
 * The scoring exists because subsequence matching alone produces garbage
 * ordering: `s` matches nearly everything, so without a score the first result
 * for `s` is whichever route happens to be declared first. Four bonuses, in
 * descending order of how much they mean:
 *
 *   1. **An exact match**, which should never be second to anything.
 *   2. **A prefix match** — you typed the start of the word, you meant it.
 *   3. **Consecutive characters**, which separate a real word from an
 *      accidental subsequence spread across a long path.
 *   4. **A match at a word boundary** — after `/`, `-`, `_`, `.` or a space —
 *      which is what makes `dn` prefer `decisions/new` over `decisions`.
 *
 * Shorter candidates win ties, because a query that matches both `chat` and
 * `chat/sessions/archived` almost always meant the short one.
 */

export interface FuzzyMatch {
  /** Higher is better. Only meaningful relative to other scores for one query. */
  score: number;
  /** Indexes in the candidate that the query matched, for highlighting. */
  positions: number[];
}

/** Characters after which a match counts as starting a word. */
const BOUNDARY = /[/\-_. ]/;

/**
 * Score one candidate against one query, or `null` if it does not match at all.
 *
 * Case-insensitive. An empty query matches everything with score 0, which is
 * what lets the palette show its resting list through the same code path
 * rather than branching around it.
 */
export function fuzzyMatch(candidate: string, query: string): FuzzyMatch | null {
  if (query.length === 0) return { score: 0, positions: [] };

  const haystack = candidate.toLowerCase();
  const needle = query.toLowerCase();

  // The two cases worth answering before walking anything.
  if (haystack === needle) return { score: 1000, positions: range(candidate.length) };

  const positions: number[] = [];
  let at = 0;
  let score = 0;
  let consecutive = 0;

  for (const char of needle) {
    const found = haystack.indexOf(char, at);
    if (found === -1) return null;

    // A run of adjacent characters is the strongest in-string signal there is.
    consecutive = found === at && positions.length > 0 ? consecutive + 1 : 0;
    score += consecutive * 8;

    if (found === 0) score += 40;
    else if (BOUNDARY.test(haystack[found - 1])) score += 20;

    positions.push(found);
    at = found + 1;
  }

  // Prefix beats a match that merely starts early, which beats one that starts
  // late in a long string.
  if (haystack.startsWith(needle)) score += 100;
  score -= positions[0];

  // Shorter is better on a tie. Scaled small so it breaks ties rather than
  // deciding matches.
  score -= candidate.length / 20;

  return { score, positions };
}

function range(length: number): number[] {
  return Array.from({ length }, (_, index) => index);
}

/**
 * Filter and rank a list in one pass.
 *
 * `key` says what to match against, which is not always what is displayed — the
 * palette matches a route against its label *and* its aliases, and takes the
 * best of the two.
 */
export function fuzzyRank<T>(items: T[], query: string, keys: (item: T) => string[]): T[] {
  const scored: { item: T; score: number }[] = [];

  for (const item of items) {
    let best: number | null = null;
    for (const key of keys(item)) {
      const match = fuzzyMatch(key, query);
      if (match && (best === null || match.score > best)) best = match.score;
    }
    if (best !== null) scored.push({ item, score: best });
  }

  // A stable sort, which `Array.prototype.sort` has been required to be since
  // ES2019 — so equal scores keep their declaration order, and the route table
  // stays the tie-breaker it was designed to be.
  return scored.sort((a, b) => b.score - a.score).map((entry) => entry.item);
}
