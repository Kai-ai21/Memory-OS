/**
 * Comfortable or compact, as one attribute on `<html>`.
 *
 * **A token swap, not a second set of components.** The temptation with a
 * density setting is a `compact` prop threaded into every row, and that is how
 * you end up with two of everything and a bug that only reproduces in one of
 * them. What actually changes is four numbers; they live as custom properties
 * on `:root`, and `[data-density="compact"]` redefines them. Every row in the
 * application already spends those properties, so nothing has to be told the
 * setting exists.
 *
 * **Type sizes do not change, and that is the whole distinction.** Compact
 * means less space *between* things. Smaller text is a different setting with
 * different consequences — it is an accessibility decision, not a layout
 * preference — and conflating the two is why "compact mode" is so often just
 * "harder to read". `density.test.tsx` asserts the font size is untouched.
 *
 * Applied to `document.documentElement` rather than to a wrapper so the value
 * is inherited by anything that escapes the React tree — the command palette
 * and the sheets are `<dialog>` elements, which the browser promotes to the top
 * layer, outside whatever div a provider would have wrapped.
 */

import { KEYS, read, write } from "./local";

export type Density = "comfortable" | "compact";

export const DENSITY_ATTRIBUTE = "data-density";

export function readDensity(): Density {
  return read<Density>(KEYS.density, "comfortable") === "compact" ? "compact" : "comfortable";
}

export function writeDensity(density: Density): void {
  write(KEYS.density, density);
  applyDensity(density);
}

/**
 * Put the current setting on the root element.
 *
 * Comfortable removes the attribute rather than setting it, so the default
 * costs nothing to express and `:root` alone remains the single definition of
 * the comfortable values — there is no state in which two rules both apply.
 */
export function applyDensity(density: Density): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (density === "compact") root.setAttribute(DENSITY_ATTRIBUTE, "compact");
  else root.removeAttribute(DENSITY_ATTRIBUTE);
}
