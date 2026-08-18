/**
 * The three rules the light theme is made of, checked mechanically.
 *
 * Each of these is a sentence somebody could otherwise quietly break in a
 * hurry, and none of them is visible in a component test: a stray hex, a
 * second frosted panel, or an accent creeping onto the active nav item all
 * render perfectly and all cost the theme the thing that makes it work.
 *
 * These read the stylesheets and the component sources as text rather than
 * rendering anything. jsdom does not apply a stylesheet, so a computed-colour
 * assertion here would be checking nothing — the files are the contract.
 */

/// <reference types="node" />
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

// Read from disk rather than imported through Vite. `?raw` would be tidier for
// the stylesheet, but the Tailwind plugin claims every `.css` request and hands
// back an empty string for a raw one — measured, not assumed. The node types
// are pulled in by the reference above so they stay scoped to this file rather
// than putting `process` in scope for every component in the application.
const SRC = resolve(process.cwd(), "src");
const INDEX_CSS = readFileSync(join(SRC, "index.css"), "utf8");
const SHELL_SOURCE = readFileSync(join(SRC, "app", "Shell.tsx"), "utf8");

/**
 * Every hand-written component source, as [path, text].
 *
 * Walked rather than listed, so a component added tomorrow is checked without
 * anybody remembering to add it. The generated API client is excluded: it is
 * not ours to keep hex-free.
 */
function components(dir = SRC, found: [string, string][] = []): [string, string][] {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      if (entry !== "api") components(path, found);
    } else if (/\.tsx?$/.test(entry) && !entry.includes(".test.")) {
      found.push([path.replace(`${SRC}/`, ""), readFileSync(path, "utf8")]);
    }
  }
  return found;
}

/** The body of one CSS rule, by selector. */
function rule(selector: string): string {
  const at = INDEX_CSS.indexOf(`${selector} {`);
  if (at === -1) throw new Error(`${selector} is not declared in index.css`);
  return INDEX_CSS.slice(at, INDEX_CSS.indexOf("}", at));
}

describe("accent means interaction, never position", () => {
  it("the active nav item carries no accent", () => {
    // The sharpest case of the rule. On dark this was cyan with a fill and a
    // glow, which was right against a void; here it is an ink rule and a weight
    // change. If the accent comes back, the one blue thing on the screen is the
    // one thing you cannot click.
    const active = rule(".nav-item-on");

    expect(active).toMatch(/--color-ink/);
    expect(active).not.toMatch(/accent/);
  });

  it("the section-heading treatment carries no accent either", () => {
    // `meta-label-on` was cyan on dark and sits on a dozen headings. A heading
    // is not an action.
    expect(rule(".meta-label-on")).not.toMatch(/accent/);
  });
});

describe("hairlines, not shadows", () => {
  it("declares no drop shadow that is not one of the four sanctioned ones", () => {
    // Shadows accumulate into noise on light faster than anything else. The
    // allowed ones: the glass button's two-part shadow, its hover, the panel
    // inset that marks an opened row, the focus rings — which are rings rather
    // than drop shadows and are how `accent` says "you are here now" — and, as
    // of M9.4, `.panel-raised`.
    //
    // **`.panel-raised` is the only true drop shadow in the system and it is
    // used on exactly one element**, the landing page's sign-in card. It earns
    // the exception by being the one panel that floats over a moving canvas
    // rather than sitting on paper, where a hairline reads as a shape the
    // background is making. The row below is what stops it becoming two.
    const shadows = [...INDEX_CSS.matchAll(/^\s*box-shadow:([^;]+);/gm)].map((m) =>
      m[1].replace(/\s+/g, " ").trim(),
    );

    for (const shadow of shadows) {
      const isRing = shadow.includes("0 0 0 3px");
      const isInset = shadow.startsWith("inset");
      // Both remaining users draw in ink at low alpha rather than in black: a
      // neutral shadow on a warm white ground goes grey, and the palette has
      // one darkness for everything.
      const isInkShadow = shadow.includes("15 23 42");
      expect(
        isRing || isInset || isInkShadow,
        `unexpected drop shadow: ${shadow}`,
      ).toBe(true);
    }
  });

  it("raises exactly one thing, on exactly one screen", () => {
    // The other half of the rule above, and the half a colour check cannot
    // see: `.panel-raised` could spread to a dozen components without changing
    // a single declared value. One class, one user, and that user is the
    // landing card.
    const users = components()
      .filter(([, source]) => /className=[^>]*\bpanel-raised\b/.test(source))
      .map(([path]) => path);

    expect(users).toHaveLength(1);
    expect(users[0]).toMatch(/WelcomePage\.tsx$/);
  });

  it("uses the hairline token for panel edges", () => {
    expect(rule(".panel")).toMatch(/--color-rule/);
  });
});

describe("glass is used on exactly one element", () => {
  it("declares one glass class and no other backdrop-filter", () => {
    const blurred = [...INDEX_CSS.matchAll(/^\s*backdrop-filter:/gm)];
    // One declaration, plus its `-webkit-` twin, and both are `.glass-button`.
    expect(blurred).toHaveLength(1);
    expect(rule(".glass-button")).toMatch(/backdrop-filter/);
  });

  it("is applied by exactly one component, and it is the primary action", () => {
    const users = components()
      .filter(([, source]) => /className=[^>]*\bglass-button\b/.test(source))
      .map(([path]) => path);

    expect(users).toHaveLength(1);
    expect(users[0]).toMatch(/Sidebar\.tsx$/);
  });

  it("has something behind it to frost", () => {
    // Frosted glass over a flat colour is a flat colour. `BackgroundLayer` is
    // the only reason the one glass element reads as glass, so the shell has to
    // mount it — a refactor that dropped it would leave a white button with a
    // white border and no test would otherwise notice.
    expect(SHELL_SOURCE).toMatch(/<BackgroundLayer\s*\/>/);
  });
});

describe("no component hardcodes a colour", () => {
  it("has no hex literal anywhere under src, outside the generated client", () => {
    // The whole argument for a token file: a hex in a component is a colour
    // that will not change when the palette is revised, and this palette has
    // now been revised twice.
    const offenders = components()
      .filter(([, source]) => /#[0-9a-fA-F]{3,8}\b/.test(source))
      .map(([path]) => path);

    expect(offenders).toEqual([]);
  });
});
