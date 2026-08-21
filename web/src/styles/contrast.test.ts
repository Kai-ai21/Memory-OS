/**
 * That the text on this theme can actually be read.
 *
 * **This is the test the light theme was designed against, not one written
 * after it.** The palette that was specified for this milestone set `ink-3` —
 * the colour every timestamp, path, count and gap sentence in the interface is
 * drawn in — to #94A3B8, which measures 2.56:1 on white. That is not a marginal
 * miss of the 4.5:1 bar; it is a little over half of it. The value moved to
 * #5D6B80, the lightest step on the same ramp that clears 4.5:1 against both
 * surfaces, and this test is what keeps it there.
 *
 * The ratios are computed from the tokens as they are actually declared in
 * `tokens.css` — the file is read and parsed rather than having its values
 * copied here. A test with the palette pasted into it proves that the numbers
 * in the test are consistent, which is not the question. The question is
 * whether the shipped colours are legible.
 *
 * Both grounds are checked. The brief asks for AA on `surface`, but metadata in
 * this interface sits on `ground` about as often — every hairline-separated
 * search row, every chat message — and a tier that passed on white while
 * failing on the page it usually appears on would satisfy the letter of the
 * requirement and none of the point.
 */

/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { HEAD_ALPHA } from "../lib/brush";

// Read from disk rather than imported. `?raw` would be tidier, but the Tailwind
// Vite plugin claims every `.css` request and hands back an empty string for a
// raw one — measured, not assumed. The node types are pulled in by the
// reference above so they stay scoped to this file: the app's tsconfig exposes
// only `vite/client`, and widening it globally would put `process` in scope for
// every component in the application.
const TOKENS = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");
const INDEX_CSS = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");

/** Pull one `--color-*: #rrggbb;` declaration out of the stylesheet. */
function token(name: string): string {
  const match = TOKENS.match(new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6})\\s*;`));
  if (!match) throw new Error(`--color-${name} is not declared as a hex in tokens.css`);
  return match[1];
}

/** WCAG 2.1 relative luminance. */
function luminance(hex: string): number {
  const channels = [1, 3, 5].map((at) => {
    const value = parseInt(hex.slice(at, at + 2), 16) / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrast(foreground: string, background: string): number {
  const a = luminance(foreground);
  const b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

const AA = 4.5;

describe("text contrast", () => {
  const surface = token("surface");
  const ground = token("ground");

  it.each([
    ["ink", "body text"],
    ["ink-2", "secondary text"],
    ["ink-3", "muted text and metadata"],
  ])("%s (%s) meets WCAG AA on surface", (name) => {
    expect(contrast(token(name), surface)).toBeGreaterThanOrEqual(AA);
  });

  it.each([["ink"], ["ink-2"], ["ink-3"]])(
    "%s also meets WCAG AA on the page ground",
    (name) => {
      // Not in the brief, and checked anyway: metadata sits on `ground` as often
      // as on `surface` — every hairline-separated search row is on the ground.
      expect(contrast(token(name), ground)).toBeGreaterThanOrEqual(AA);
    },
  );

  it("keeps the three tiers visibly apart", () => {
    // Passing AA is not enough on its own. Three greys that all clear the bar
    // but sit within a hair of each other are one grey, and the hierarchy the
    // whole light theme leans on — weight and darkness rather than colour —
    // stops working.
    const ink = contrast(token("ink"), surface);
    const ink2 = contrast(token("ink-2"), surface);
    const ink3 = contrast(token("ink-3"), surface);

    expect(ink).toBeGreaterThan(ink2);
    expect(ink2).toBeGreaterThan(ink3);
    expect(ink2 - ink3).toBeGreaterThan(1);
  });

  it.each([
    ["accent", "links, focus rings and the send button"],
    ["warn", "refusals, gaps and counter-evidence"],
    ["affirm", "a verdict that held"],
    ["deny", "a verdict that failed"],
  ])("%s (%s) meets WCAG AA on surface", (name) => {
    // These carry meaning rather than decoration — a refusal that cannot be
    // read is a refusal that did not happen.
    expect(contrast(token(name), surface)).toBeGreaterThanOrEqual(AA);
  });

  it("does not ship the palette value that fails", () => {
    // Named explicitly so the reason is in the suite rather than only in a
    // commit message: #94A3B8 measures 2.56:1 on white and was the specified
    // `ink-3`. If it ever comes back, this says why it left.
    expect(contrast("#94A3B8", surface)).toBeLessThan(AA);
    expect(token("ink-3")).not.toBe("#94A3B8");
  });
});

/**
 * The sidebar is not a surface any more, and this block is what M9.8 owes for
 * that.
 *
 * Every ratio above is text on an opaque token colour. The nav labels are now
 * text on 72% white over *whatever is behind the panel*, which is the page
 * ground, two soft radial washes, and — because the sidebar is deliberately not
 * a particle shelter — the cursor trail at up to `HEAD_ALPHA` of ink. "It looks
 * fine" is not an answer for a background that moves, so the composite is
 * computed and the worst case is the one asserted.
 *
 * The blur is ignored, which makes this pessimistic in the right direction: a
 * 12px gaussian spreads the stroke's peak out and can only *raise* the measured
 * background luminance under any given label.
 */
describe("text contrast on the glass panel", () => {
  /** The panel's own white alpha, read from the declaration rather than typed. */
  function panelAlpha(): number {
    const match = INDEX_CSS.match(
      /\.glass-panel\s*\{[^}]*background:\s*rgb\(255 255 255 \/ (\d+)%\)/,
    );
    if (!match) throw new Error(".glass-panel declares no rgb(255 255 255 / N%) background");
    return Number(match[1]) / 100;
  }

  /** `over` composited under `alpha` of white, as a hex. */
  function underGlass(over: string, alpha: number): string {
    const channels = [1, 3, 5].map((at) => {
      const value = parseInt(over.slice(at, at + 2), 16);
      return Math.round(alpha * 255 + (1 - alpha) * value);
    });
    return `#${channels.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
  }

  /** `ink` laid over `ground` at `alpha` — the cursor trail, at its head. */
  function inked(alpha: number): string {
    const ink = token("ink");
    const ground = token("ground");
    const channels = [1, 3, 5].map((at) => {
      const a = parseInt(ink.slice(at, at + 2), 16);
      const b = parseInt(ground.slice(at, at + 2), 16);
      return Math.round(alpha * a + (1 - alpha) * b);
    });
    return `#${channels.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
  }

  const alpha = panelAlpha();
  /** The panel at rest: nothing behind it but the page. */
  const calm = underGlass(token("ground"), alpha);
  /** The panel at its darkest: the head of the cursor stroke directly behind. */
  const worst = underGlass(inked(HEAD_ALPHA), alpha);

  it.each([
    ["ink", "the active row"],
    ["ink-2", "every other row"],
  ])("%s (%s) meets AA on the panel at rest", (name) => {
    expect(contrast(token(name), calm)).toBeGreaterThanOrEqual(AA);
  });

  it("holds AA for the nav label with the cursor trail directly behind it", () => {
    // **This is the tightest number in the theme and it is worth knowing the
    // margin.** `ink-2` on the calm panel is comfortable; under the head of the
    // stroke it is at the bar rather than above it. If the panel is ever made
    // more transparent, or the stroke darker, this is what says so — and the
    // fix is one of those two values, not the label colour.
    expect(contrast(token("ink-2"), worst)).toBeGreaterThanOrEqual(AA);
  });

  it("the active row clears AA on its own fill", () => {
    // The active row is the one place in the panel with an opaque background:
    // `surface-tint` at full strength, which is what M9.9 replaced a
    // half-transparent mix with. Nothing behind the panel can reach the text on
    // it, so this one is a flat pair and is comfortable — 15.8:1.
    expect(contrast(token("ink"), token("surface-tint"))).toBeGreaterThanOrEqual(AA);
  });

  it("keeps the resting glyph above the 3:1 bar for non-text", () => {
    // **The icons are `ink-3` and they are not text.** WCAG asks 3:1 of a
    // graphical object that carries meaning rather than the 4.5:1 it asks of a
    // label, and that is the right bar here: the glyph identifies a row, and
    // the word beside it says the same thing at 4.5:1 or better. Under the head
    // of the cursor stroke this is 3.2:1, which is the tightest non-text
    // measurement in the interface — if the panel is made more transparent, or
    // the stroke darker, this is the row that says so first.
    expect(contrast(token("ink-3"), worst)).toBeGreaterThanOrEqual(3);
  });

  it("keeps the panel lighter than the ground it floats on", () => {
    // The panel has to read as *above* the page. If the composite ever goes
    // darker than `ground`, the glass has become a grey rectangle and no
    // amount of shadow will rescue it.
    expect(luminance(calm)).toBeGreaterThan(luminance(token("ground")));
  });
});
