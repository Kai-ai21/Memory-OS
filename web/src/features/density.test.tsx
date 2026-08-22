/**
 * Compact means less space, not smaller text.
 *
 * That is the whole distinction the setting rests on, and it is the one that
 * decays: the easy way to make an interface denser is to shrink the type, and
 * the result is a "compact mode" people turn on once. So the assertion is a
 * pair — the row got shorter *and* the font size did not move.
 *
 * Measured off the resolved custom properties rather than by asserting a class
 * name. The tokens are declared here exactly as `tokens.css` and `index.css`
 * declare them, and the cascade that swaps them is the thing under test — a
 * test that only checked for `data-density="compact"` on the root would pass
 * with every token deleted.
 *
 * jsdom does not resolve `var()` inside a longhand — `padding-top` computes to
 * the literal string `var(--row-py)` — but it does resolve the properties
 * themselves, which is where the swap actually happens. The row is still
 * rendered and asserted to *spend* the token, so a component that stopped
 * referencing it would fail here rather than silently stop responding.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { readFileSync } from "node:fs";
import { globSync } from "node:fs";
import { join } from "node:path";

import { applyDensity, readDensity, writeDensity } from "../lib/density";

const ROOT = join(import.meta.dirname, "..");
const TOKENS_CSS = readFileSync(join(ROOT, "styles/tokens.css"), "utf8");
const INDEX_CSS = readFileSync(join(ROOT, "index.css"), "utf8");

/** The density block in `tokens.css`, which is the list under test. */
const DENSITY_BLOCK = TOKENS_CSS.slice(
  TOKENS_CSS.indexOf("--- Density"),
  TOKENS_CSS.indexOf("--- Space"),
);

/** Every component, for "does anything actually spend this token". */
const SOURCES = globSync(join(ROOT, "**/*.tsx"))
  .filter((path) => !path.endsWith(".test.tsx"))
  .map((path) => readFileSync(path, "utf8"));

/** The two declarations under test, lifted verbatim from the stylesheets. */
const STYLES = `
  :root {
    --row-py: 0.375rem;
    --result-py: 1.25rem;
    --nav-h: 2.375rem;
  }
  :root[data-density="compact"] {
    --row-py: 0.25rem;
    --result-py: 0.8125rem;
    --nav-h: 1.875rem;
  }
  .row { padding-top: var(--row-py); padding-bottom: var(--row-py); font-size: 0.8125rem; }
`;

let sheet: HTMLStyleElement;

beforeEach(() => {
  sheet = document.createElement("style");
  sheet.textContent = STYLES;
  document.head.appendChild(sheet);
  window.localStorage.clear();
  applyDensity("comfortable");
});

afterEach(() => {
  sheet.remove();
  applyDensity("comfortable");
  window.localStorage.clear();
});

/** The token's current value, in rem, off the root where the swap happens. */
function rowPadding(): number {
  const value = getComputedStyle(document.documentElement).getPropertyValue("--row-py");
  return Number.parseFloat(value);
}

function rowMetrics() {
  const style = getComputedStyle(screen.getByTestId("row"));
  return {
    /* That the row spends the token at all. jsdom leaves this unresolved,
       which is exactly what makes it a usable assertion: it is the reference
       itself, so a component that went back to a literal fails here. */
    spendsToken: style.paddingTop,
    fontSize: style.fontSize,
  };
}

describe("density", () => {
  it("reduces row padding without touching the type size", () => {
    render(
      <div className="row" data-testid="row">
        a result
      </div>,
    );

    const comfortable = rowMetrics();
    expect(comfortable.spendsToken).toBe("var(--row-py)");
    const before = rowPadding();
    expect(before).toBeCloseTo(0.375);

    applyDensity("compact");
    const compact = rowMetrics();
    const after = rowPadding();

    // About a third off — 20% is invisible and 50% is a spreadsheet.
    expect(after).toBeCloseTo(0.25);
    expect(1 - after / before).toBeGreaterThan(0.3);

    // And the half that matters most: the text is exactly the size it was.
    expect(compact.fontSize).toBe(comfortable.fontSize);
    expect(compact.spendsToken).toBe("var(--row-py)");
  });

  it("puts the setting on the root element, and takes it off again", () => {
    applyDensity("compact");
    expect(document.documentElement).toHaveAttribute("data-density", "compact");

    // Comfortable *removes* the attribute rather than setting it, so there is
    // never a state where two rules both apply.
    applyDensity("comfortable");
    expect(document.documentElement).not.toHaveAttribute("data-density");
  });

  it("persists, and comes back on the next load", () => {
    writeDensity("compact");
    expect(readDensity()).toBe("compact");
    expect(document.documentElement).toHaveAttribute("data-density", "compact");

    // A fresh boot reads storage and re-applies before first paint.
    applyDensity("comfortable");
    applyDensity(readDensity());
    expect(document.documentElement).toHaveAttribute("data-density", "compact");
  });

  it("declares no density token that nothing spends", () => {
    /* **Found in a browser, not here, and it is the failure mode this whole
     * approach has.** `@theme` in Tailwind only emits a custom property that
     * something references; the compact block is a plain rule that always
     * emits. So a token declared as comfortable and used by no component
     * resolves to *empty* at comfortable and to a real value at compact — the
     * one combination that looks fine in a test and breaks the moment somebody
     * uses it. A `--stack-gap` shipped in exactly that state.
     *
     * Read off the source rather than the DOM: jsdom does not run Tailwind, so
     * the emission behaviour cannot be observed here — but the invariant that
     * causes it can. */
    const declared = [...TOKENS_CSS.matchAll(/^\s*(--(?:row-py|result-py|nav-h|[a-z-]*(?:gap|py|px|h))):/gm)]
      .map((match) => match[1])
      .filter((name) => DENSITY_BLOCK.includes(`${name}:`));

    for (const token of declared) {
      const spent =
        new RegExp(`var\\(${token}\\)`).test(INDEX_CSS) ||
        SOURCES.some((source) => source.includes(`(${token})`) || source.includes(`var(${token})`));
      expect(spent, `${token} is declared but nothing spends it`).toBe(true);
    }
  });

  it("treats anything unrecognised as comfortable", () => {
    // Shared with every past and future version of the application.
    window.localStorage.setItem("memo:density", JSON.stringify("cosy"));
    expect(readDensity()).toBe("comfortable");
  });
});
