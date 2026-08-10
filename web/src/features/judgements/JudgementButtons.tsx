/**
 * The three verdicts. The reason this milestone exists.
 *
 * Deliberately unglamorous: three small square buttons in the result header, no
 * modal, no confirmation, no toast. Capturing a judgement has to cost one click,
 * because a golden set is built by judging a hundred results in a sitting and
 * anything that adds friction per result is what stops it being built.
 *
 * The verdict currently recorded is shown as an active state, so re-judging is
 * visibly a change rather than a duplicate.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api, type JudgementIn, type Verdict } from "../../api/client";

export interface JudgementTarget {
  queryText: string;
  sourceName: string;
  externalKey: string;
  memoryId: string;
  /**
   * Which chunk the verdict is about. Left undefined the verdict is about the
   * memory — "the right file came back". Set, it is about that chunk alone,
   * which is the only way to record "right file, wrong chunk".
   */
  chunkOrdinal?: number | null;
  chunkId?: string | null;
  rank?: number | null;
  score?: number | null;
  filters: Record<string, unknown>;
}

const LABELS: Record<Verdict, string> = {
  relevant: "relevant",
  not_relevant: "not relevant",
  missing: "missing",
};

const COMPACT_LABELS: Record<Verdict, string> = {
  relevant: "✓",
  not_relevant: "✗",
  missing: "missing",
};

interface Props {
  target: JudgementTarget;
  current?: Verdict | null;
  onRecorded?: (verdict: Verdict) => void;
  /** `missing` only makes sense for something the search did not return. */
  verdicts?: Verdict[];
  /** Shorter labels, for the per-chunk row where the file header already said it. */
  compact?: boolean;
}

export function JudgementButtons({
  target,
  current,
  onRecorded,
  verdicts = ["relevant", "not_relevant"],
  compact = false,
}: Props) {
  const client = useQueryClient();

  const mutation = useMutation({
    mutationFn: (verdict: Verdict) => {
      const body: JudgementIn = {
        query_text: target.queryText,
        source_name: target.sourceName,
        external_key: target.externalKey,
        verdict,
        chunk_ordinal: target.chunkOrdinal ?? null,
        memory_id: target.memoryId,
        chunk_id: target.chunkId ?? null,
        // A `missing` verdict carries no rank — the point of it is that the item
        // was not in the ranking. The API rejects one that does, so this is the
        // client agreeing rather than the client deciding.
        rank_at_judgement: verdict === "missing" ? null : (target.rank ?? null),
        score_at_judgement: verdict === "missing" ? null : (target.score ?? null),
        filters: target.filters,
      };
      return api.judge(body);
    },
    onSuccess: (_data, verdict) => {
      // The /judgements page counts these, so its cache is now stale.
      void client.invalidateQueries({ queryKey: ["judgements"] });
      onRecorded?.(verdict);
    },
  });

  return (
    <div className="flex items-center gap-1" data-testid="judgement-buttons">
      {verdicts.map((verdict) => {
        const active = current === verdict;
        const tone =
          verdict === "relevant"
            ? "btn-affirm-on"
            : verdict === "not_relevant"
              ? "btn-deny-on"
              : "btn-on";
        return (
          <button
            key={verdict}
            type="button"
            className={`btn ${active ? tone : ""}`}
            disabled={mutation.isPending}
            aria-pressed={active}
            onClick={() => mutation.mutate(verdict)}
          >
            {compact ? COMPACT_LABELS[verdict] : LABELS[verdict]}
          </button>
        );
      })}
      {mutation.isError ? (
        <span className="meta text-deny" role="alert">
          not saved
        </span>
      ) : null}
    </div>
  );
}
