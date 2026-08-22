/**
 * What the system said without being asked, and the two clicks that judge it.
 *
 * **The refusals are on this page at the same weight as the surfacings, and that
 * is the whole design.** A proactive feature is judged on what it interrupted
 * you with, and a screen showing only that is a screen that flatters itself: the
 * gate refuses far more often than it speaks, and a reader who cannot see the
 * refusals has no way to tell a careful system from a broken one. Same argument
 * the patterns page makes about counter-evidence, one phase along.
 *
 * The dismissal rate is at the top with the sentence that makes it readable.
 * Above 50% this feature is noise and the page says so in those words — a tool
 * that reports its own failure is the only kind whose successes mean anything.
 *
 * Feedback is two buttons and no free text. A dismissal reason would be the
 * right thing to collect for a *pattern*, which is a claim about somebody's
 * judgement they may want to argue with; this is a list of files, and asking for
 * prose about why a list of files was not useful is how you get no feedback at
 * all.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type Surfaced } from "../../api/client";
import { Empty, Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { RelativeTime } from "../../components/RelativeTime";

export function SurfacingPage() {
  const [includeRefused, setIncludeRefused] = useState(true);
  const client = useQueryClient();

  const rows = useQuery({
    queryKey: ["surfacing", includeRefused],
    queryFn: () => api.surfacing(includeRefused),
  });

  const rate = useMutation({
    mutationFn: ({ id, useful }: { id: string; useful: boolean }) =>
      useful ? api.markSurfacingUseful(id) : api.dismissSurfacing(id),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["surfacing"] });
    },
  });

  if (rows.isLoading) return <Loading rows={5} />;
  if (rows.isError) return <Failure error={rows.error} />;

  const decisions = rows.data ?? [];
  const surfaced = decisions.filter((row) => row.surfaced);
  const dismissed = surfaced.filter((row) => row.verdict === "dismissed");
  const dismissalRate =
    surfaced.length === 0 ? null : dismissed.length / surfaced.length;
  const noisy = dismissalRate !== null && dismissalRate > 0.5;
  // Exactly half gets its own sentence rather than being rounded into one of
  // its neighbours. It is the number the first real run produced, and both of
  // the other two would have been a lie about it.
  const evens = dismissalRate === 0.5;

  return (
    <div className="flex flex-col gap-5">
      <SectionHeading
        right={
          <button
            type="button"
            className={includeRefused ? "text-accent underline" : "text-ink-2 hover:text-ink"}
            onClick={() => setIncludeRefused((current) => !current)}
          >
            show refusals
          </button>
        }
      >
        surfacing
      </SectionHeading>

      {rate.isError ? <Failure error={rate.error} /> : null}

      {surfaced.length === 0 ? null : (
        <p className={`meta max-w-prose leading-relaxed ${noisy ? "text-deny" : "text-ink-2"}`}>
          <span className="text-ink">
            {dismissed.length} of {surfaced.length} surfaced
          </span>{" "}
          dismissed
          {noisy
            ? " — above half, which means this is noise. A tool that interrupts with mediocre suggestions gets muted, and then its good ones are muted too."
            : evens
              ? " — exactly half. Not above the line and not below it: one interruption in two was worth having, and a coin flip is not a feature."
              : ". Below half, which is the bar this feature set itself. That is evidence it is not actively harmful, not that it is useful."}
        </p>
      )}

      {decisions.length === 0 ? (
        <Empty title="nothing decided yet">
          The gate runs when an event arrives — a file focused, an editor opened, a meeting
          about to start. It defaults to silence: the top item has to be found by two
          independent routes and must not be the file you already have open. Every decision
          either way is recorded, so this page is empty only when nothing has been triggered.
        </Empty>
      ) : (
        <ul className="flex flex-col">
          {decisions.map((row) => (
            <DecisionRow
              key={row.id}
              row={row}
              busy={rate.isPending}
              onRate={(useful) => rate.mutate({ id: row.id, useful })}
            />
          ))}
        </ul>
      )}

      <p className="meta max-w-prose leading-relaxed text-ink-3">
        The threshold is per focus and adapts: a focus whose context is dismissed raises its
        own bar, one whose context is useful lowers it slowly. Per focus rather than global,
        so one noisy file cannot silence the rest.{" "}
        <code className="kbd">memoryos surfacing stats</code> has the thresholds.
      </p>
    </div>
  );
}

function DecisionRow({
  row,
  busy,
  onRate,
}: {
  row: Surfaced;
  busy: boolean;
  onRate: (useful: boolean) => void;
}) {
  // A refusal is rendered quieter than a surfacing but is not hidden. The
  // distinction a reader needs is "did this interrupt me", and the type size
  // carries it without the row having to say so.
  const tone = row.surfaced ? "text-ink" : "text-ink-2";

  return (
    <li className="border-b border-rule/60 py-2.5">
      <div className="meta flex flex-wrap items-baseline gap-3 text-ink-3">
        <Tag>{row.surfaced ? "surfaced" : "quiet"}</Tag>
        <span className={tone}>{row.focus}</span>
        {row.trigger_kind ? <span>{row.trigger_kind}</span> : null}
        <RelativeTime value={row.decided_at} />
        {row.verdict === "dismissed" ? <span className="text-deny">dismissed</span> : null}
        {row.verdict === "useful" ? <span className="text-affirm">useful</span> : null}
      </div>

      <p className={`prose-content mt-0.5 max-w-prose text-sm ${tone}`}>
        {row.top_title ?? "nothing to show"}
        {row.item_count > 1 ? (
          <span className="meta text-ink-3"> and {row.item_count - 1} more</span>
        ) : null}
      </p>

      {/* The reason, always — including for the ones that were shown. "Two
          routes agreed on something you do not already have open" is the only
          thing on screen that lets a reader judge the interruption rather than
          take it, which is the same argument the panel's route tags make. */}
      <p className="meta mt-0.5 text-ink-3">
        {row.reason}: {row.explanation}.{" "}
        <span className="text-ink-2">
          scored {row.score.toFixed(4)} against {row.threshold.toFixed(4)}
        </span>
      </p>

      {row.surfaced && row.verdict === null ? (
        <div className="mt-1.5 flex items-center gap-2">
          <button
            type="button"
            className="meta-label border border-rule px-3 py-1 text-affirm disabled:opacity-40"
            disabled={busy}
            onClick={() => onRate(true)}
          >
            useful
          </button>
          <button
            type="button"
            className="meta-label border border-rule px-3 py-1 text-deny disabled:opacity-40"
            disabled={busy}
            onClick={() => onRate(false)}
          >
            dismiss
          </button>
        </div>
      ) : null}
    </li>
  );
}
