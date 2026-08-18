/**
 * Bars and a click handler. No chart library.
 *
 * A dependency for this would be several hundred kilobytes to draw rectangles
 * whose heights are a division, and it would bring its own opinions about
 * typography, tooltips and colour that would then have to be fought. What is
 * actually needed here is one flex row, one percentage per bar, and a button
 * element — which is also how it ends up keyboard-navigable for free.
 *
 * **Three things this draws that a text histogram cannot.**
 *
 * Stacked by kind, because "forty memories" and "forty memories that are all
 * code" are different facts and only one of them is on the screen in a terminal.
 *
 * Empty periods hatched rather than blank. `find_gaps` exists because absence is
 * the signal, and an absence drawn as absence looks like the end of the data or
 * like a chart that failed to load.
 *
 * Gaps as objects, in their own lane below the axis, positioned by real time
 * rather than snapped to bucket edges. A gap that runs from the 7th to the 46th
 * does not begin where a bar begins, and rounding it to one would move the thing
 * being measured.
 */

import { useRef } from "react";

import type { Bucket, Gap } from "../../api/client";
import { count } from "../../lib/format";
import { KIND_ORDER, kindColor } from "./model";

export function ActivityChart({
  buckets,
  gaps,
  selected,
  onSelect,
  labelFor,
}: {
  buckets: Bucket[];
  gaps: Gap[];
  /** The selected bucket's `start`, or null. */
  selected: string | null;
  onSelect: (bucket: Bucket) => void;
  labelFor: (bucket: Bucket) => string;
}) {
  const bars = useRef<HTMLDivElement>(null);
  const peak = Math.max(1, ...buckets.map((bucket) => bucket.count));

  // Every bar label at once is unreadable past about thirty periods, so they
  // thin out. The first and last are always kept: a chart whose axis does not
  // say where it starts and ends is a chart of nothing in particular.
  const stride = Math.max(1, Math.ceil(buckets.length / 24));

  const windowStart = buckets.length ? Date.parse(buckets[0].start) : 0;
  const windowEnd = buckets.length ? Date.parse(buckets[buckets.length - 1].end) : 1;
  const span = Math.max(1, windowEnd - windowStart);

  /**
   * Roving arrow-key focus across the bars.
   *
   * The keyboard is how this interface is driven, and a chart you can only
   * click is a chart half the tool cannot reach. Left/right walk the periods,
   * Home/End jump to the ends; Enter and Space are the button's own.
   */
  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
    if (!keys.includes(event.key)) return;
    const all = Array.from(bars.current?.querySelectorAll("button") ?? []);
    const here = all.indexOf(document.activeElement as HTMLButtonElement);
    if (here < 0) return;
    event.preventDefault();
    const next =
      event.key === "ArrowLeft"
        ? Math.max(0, here - 1)
        : event.key === "ArrowRight"
          ? Math.min(all.length - 1, here + 1)
          : event.key === "Home"
            ? 0
            : all.length - 1;
    all[next]?.focus();
  }

  if (buckets.length === 0) {
    return (
      <p className="meta text-faint">
        no periods in this window — nothing in the corpus carries a date inside it
      </p>
    );
  }

  return (
    <div className="flex flex-col">
      {/* The plot. `items-end` so bars grow from the axis upward. */}
      <div
        ref={bars}
        className="relative flex h-44 items-end gap-px border-b border-rule-strong"
        onKeyDown={onKeyDown}
        role="group"
        aria-label="activity per period"
        data-testid="activity-chart"
      >
        {/* The gap band, over the plot rather than in a lane under it.
            **This is the most important element on the timeline.** M4.0's whole
            argument is that absence is the signal, and a thin hatched strip
            below the axis makes it a footnote to the bars — the opposite of
            that argument. Drawn at full chart height, in magenta, dashed, it is
            the first thing the eye lands on, which is correct: a month with
            nothing in it is more interesting than a month with forty things.
            Dashed because a dashed boundary is the drawing convention for a
            region that is inferred rather than measured, and magenta because
            that is this palette's colour for what the system does not have. */}
        {gaps.map((gap) => {
          const from = clamp((Date.parse(gap.start) - windowStart) / span);
          const to = clamp((Date.parse(gap.end) - windowStart) / span);
          const width = Math.max(to - from, 0.01);
          return (
            <span
              key={`${gap.source_name}-${gap.start}`}
              className="gap-band pointer-events-none absolute inset-y-0 z-10 flex items-center justify-center"
              style={{ left: `${from * 100}%`, width: `${width * 100}%` }}
              title={`gap of ${gap.days.toFixed(1)} days — ${gap.before.external_key} then nothing until ${gap.after.external_key}`}
              data-testid="gap-marker"
              data-days={gap.days}
            >
              {/* Labelled inside the band when there is room for the words, and
                  only then: a label wider than the region it names points at
                  the wrong part of the chart. */}
              {width > 0.12 ? (
                <span className="meta-label whitespace-nowrap text-magenta">
                  {Math.round(gap.days)} day gap
                </span>
              ) : null}
            </span>
          );
        })}
        {buckets.map((bucket) => {
          const isSelected = bucket.start === selected;
          const empty = bucket.count === 0;
          return (
            <button
              key={bucket.start}
              type="button"
              onClick={() => onSelect(bucket)}
              aria-pressed={isSelected}
              aria-label={`${labelFor(bucket)}: ${bucket.count} memories`}
              title={`${labelFor(bucket)} — ${count(bucket.count)} ${describe(bucket)}`}
              data-testid="bar"
              data-bucket={bucket.start}
              data-count={bucket.count}
              className={`group relative flex h-full min-w-1.5 flex-1 cursor-pointer flex-col justify-end ${
                isSelected ? "bg-cyan/12" : "hover:bg-glass"
              }`}
            >
              {/* The count sits directly above its own bar rather than at the
                  top of the plot. As an absolutely-positioned label it floated
                  at the chart's ceiling and had to be matched to a bar by eye,
                  which is the reading error the number exists to remove. In the
                  column, `justify-end` puts it exactly on the bar's shoulder. */}
              {!empty ? (
                <span className="meta pointer-events-none pb-0.5 text-center text-faint group-hover:text-ink">
                  {bucket.count}
                </span>
              ) : null}

              {empty ? (
                // A void, drawn. Short, so it never competes with a real bar,
                // and hatched so it cannot be read as a very small count.
                <span
                  className="hatch h-2 w-full opacity-45"
                  data-testid="empty-period"
                  aria-hidden="true"
                />
              ) : (
                <span
                  className="flex w-full flex-col-reverse"
                  style={{ height: `${(bucket.count / peak) * 100}%` }}
                >
                  {KIND_ORDER.filter((kind) => bucket.by_kind?.[kind]).map((kind) => (
                    <span
                      key={kind}
                      // Washed rather than filled. At full strength a wide bar
                      // is a field of saturated colour, which is a dashboard —
                      // and this interface spends its one accent on the matched
                      // -span highlight. 70% puts the bars back in the register
                      // of ink on paper without losing the categorical read.
                      className="opacity-90 shadow-[0_0_10px_currentColor]"
                      style={{
                        height: `${((bucket.by_kind?.[kind] ?? 0) / bucket.count) * 100}%`,
                        backgroundColor: kindColor(kind),
                      }}
                      data-kind={kind}
                    />
                  ))}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* The axis. */}
      <div className="flex gap-px">
        {buckets.map((bucket, index) => (
          <span
            key={bucket.start}
            className="meta min-w-1.5 flex-1 overflow-hidden pt-1 text-center text-faint"
          >
            {index % stride === 0 || index === buckets.length - 1 ? labelFor(bucket) : ""}
          </span>
        ))}
      </div>
    </div>
  );
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function describe(bucket: Bucket): string {
  const parts = Object.entries(bucket.by_kind ?? {}).map(([kind, n]) => `${n} ${kind}`);
  return parts.length ? `(${parts.join(", ")})` : "memories";
}
