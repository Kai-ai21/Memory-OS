/**
 * Reading a colour token at runtime, for the two things that cannot use one.
 *
 * Everything drawn with CSS spends `var(--color-…)` directly and this file is
 * irrelevant to it. A canvas cannot: `fillStyle` takes a string, and the string
 * has to be a real colour by the time it is assigned. The choice is between
 * copying the palette into JavaScript — a second definition of a colour that
 * stays behind when the first one is revised, which this palette has now been
 * twice — and asking the document what the token currently says. This is the
 * second.
 *
 * The fallback is the value the token holds today, kept in channel form so
 * `rgba(${ink}, 0.08)` composes without a second parser.
 */

/** `r, g, b` — ready to interpolate into an `rgba()` string. */
export type Channels = string;

/**
 * Read one `--color-*` token off the document root as `"r, g, b"`.
 *
 * Accepts hex or `rgb()`: the token file declares hex today, a browser may hand
 * back either, and a future revision may declare either. Both are cheap to
 * parse and the cost of not parsing one is a silently wrong colour.
 */
export function readTokenRgb(name: string, fallback: Channels): Channels {
  if (typeof window === "undefined" || typeof getComputedStyle !== "function") {
    return fallback;
  }
  const declared = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();

  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(declared);
  if (hex) {
    const digits =
      hex[1].length === 3
        ? hex[1].split("").map((digit) => digit + digit)
        : [hex[1].slice(0, 2), hex[1].slice(2, 4), hex[1].slice(4, 6)];
    return digits.map((pair) => parseInt(pair, 16)).join(", ");
  }

  const channels = declared.match(/\d+(\.\d+)?/g);
  if (channels && channels.length >= 3) return channels.slice(0, 3).join(", ");

  return fallback;
}

/**
 * The ink the canvases draw in, and the paper they draw it on.
 *
 * **Ink, not black.** Pure black on a warm white ground reads as a hole punched
 * in the paper rather than as a mark on it; the ink token is the darkness the
 * type is, so anything drawn in it belongs to the same document.
 */
export const INK_FALLBACK: Channels = "15, 23, 42";
export const GROUND_FALLBACK: Channels = "247, 248, 250";

export const readInk = () => readTokenRgb("--color-ink", INK_FALLBACK);
export const readGround = () => readTokenRgb("--color-ground", GROUND_FALLBACK);
