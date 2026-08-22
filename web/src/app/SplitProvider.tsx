/**
 * The provider component, and nothing else.
 *
 * Its own file so that `lib/split.ts` can export the hooks and the clamp
 * without also exporting a component — a module that mixes the two breaks fast
 * refresh for everything that imports it.
 */

import { useCallback, useMemo, useState } from "react";

import { KEYS, read, write } from "../lib/local";
import { clampWidth, DEFAULT_WIDTH, readRecentMemoryId, SplitContext } from "../lib/split";

export function SplitProvider({ children }: { children: React.ReactNode }) {
  const [memoryId, setMemoryId] = useState<string | null>(null);
  const [width, setWidthState] = useState(() => clampWidth(read(KEYS.splitWidth, DEFAULT_WIDTH)));

  const open = useCallback((id: string) => setMemoryId(id), []);
  const close = useCallback(() => setMemoryId(null), []);

  const setWidth = useCallback((percent: number) => {
    const next = clampWidth(percent);
    setWidthState(next);
    write(KEYS.splitWidth, next);
  }, []);

  /**
   * `⌘\` with nothing open falls back to the most recent memory.
   *
   * The alternative is a shortcut that does nothing most of the time, which is
   * how a keyboard binding gets a reputation for being broken. Read lazily
   * rather than held in state so it reflects anything opened since mount.
   */
  const toggle = useCallback(() => {
    setMemoryId((current) => (current ? null : readRecentMemoryId()));
  }, []);

  const value = useMemo(
    () => ({ memoryId, width, open, close, toggle, setWidth }),
    [memoryId, width, open, close, toggle, setWidth],
  );

  return <SplitContext.Provider value={value}>{children}</SplitContext.Provider>;
}

