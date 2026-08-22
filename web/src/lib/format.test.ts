/**
 * `relativeTime`, at the boundaries that decide what it prints.
 *
 * The three the brief names — an hour, a day, eight days — are each on a
 * different side of a decision this function makes, which is why they are the
 * three worth pinning: an hour picks a unit, a day proves the "yesterday"
 * wording is coming from the platform formatter rather than a template, and
 * eight days is past the cut-off where relative stops being useful at all.
 *
 * `now` is passed explicitly rather than faked with timers. The function takes
 * it as a parameter for exactly this reason: a test that moves the system clock
 * is a test that has to put it back, and one that forgets is a flake in
 * whatever runs next.
 */

import { describe, expect, it } from "vitest";

import { relativeTime } from "./format";

/** A fixed reference point. Nothing here depends on the real clock. */
const NOW = new Date("2026-08-22T12:00:00Z");

function ago(ms: number): string {
  return new Date(NOW.getTime() - ms).toISOString();
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

describe("relative time", () => {
  it("says an hour ago", () => {
    expect(relativeTime(ago(HOUR), NOW)).toBe("1 hour ago");
  });

  it("says yesterday rather than 1 day ago", () => {
    // `numeric: "auto"` is the whole reason to use `Intl.RelativeTimeFormat`
    // instead of a template, and this is the assertion that proves it is set.
    expect(relativeTime(ago(DAY), NOW)).toBe("yesterday");
  });

  it("falls back to a date past a week", () => {
    const formatted = relativeTime(ago(8 * DAY), NOW);

    // The date itself, in the reader's locale — not "8 days ago". The exact
    // spelling is the platform's, so this asserts the shape rather than the
    // string: it must name the year and must not be relative.
    expect(formatted).not.toMatch(/ago/);
    expect(formatted).toContain("2026");
  });

  it("holds the boundary in both directions", () => {
    // Six days is still relative; eight is not. The cut-off is a real edge and
    // an off-by-one here is invisible in every other test.
    expect(relativeTime(ago(6 * DAY), NOW)).toBe("6 days ago");
    expect(relativeTime(ago(8 * DAY), NOW)).not.toMatch(/ago/);
  });

  it("truncates towards zero rather than rounding up", () => {
    // 90 minutes is "1 hour ago". Rounding would report it as two, and for a
    // freshness read that is the direction that misleads.
    expect(relativeTime(ago(90 * MINUTE), NOW)).toBe("1 hour ago");
  });

  it("says just now under a minute, and nothing at all with no date", () => {
    expect(relativeTime(ago(20_000), NOW)).toBe("just now");
    expect(relativeTime(null, NOW)).toBe("—");
  });
});
