/**
 * The suggestion queue: a draft on the left, the passage it came from on the right.
 *
 * **The passage is the point of this screen.** A draft alone always reads well —
 * that is what a language model is for — so a queue showing only the draft asks
 * the reviewer to judge plausibility, which is the one judgement that cannot
 * distinguish a real decision from an invented one. Side by side, accept becomes
 * a question about evidence: does the passage actually say this was decided, and
 * does it actually name the alternative?
 *
 * Reject is as prominent as accept, and neither is the default. Nothing here is
 * pre-selected and there is no "accept all": the whole value of the queue is
 * that clicking accept is a considered act, and a bulk action would undo it.
 *
 * Edit-then-accept is a separate route rather than an inline form, because a
 * reviewer who has read the passage usually knows a confidence and an assumption
 * the model could not have — and those belong in the capture form, which asks
 * for them properly.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type Suggestion } from "../../api/client";
import { Empty, Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { RelativeTime } from "../../components/RelativeTime";

import type { PrefilledDraft } from "./DecisionForm";

interface Draft {
  question?: string | null;
  chosen?: string | null;
  reasoning?: string | null;
  confidence?: number | null;
  expected_outcome?: string | null;
  options?: { description?: string | null; rejected_because?: string | null }[] | null;
  assumptions?: { statement?: string | null }[] | null;
}

export function ReviewQueue() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [status, setStatus] = useState<Suggestion["status"]>("pending");
  const suggestions = useQuery({
    queryKey: ["suggestions", status],
    queryFn: () => api.suggestions(status),
  });

  const review = useMutation({
    mutationFn: ({ id, verdict }: { id: string; verdict: "accept" | "reject" }) =>
      verdict === "accept" ? api.acceptSuggestion(id) : api.rejectSuggestion(id),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["suggestions"] });
      await client.invalidateQueries({ queryKey: ["decisions"] });
    },
  });

  if (suggestions.isLoading) return <Loading rows={4} />;
  if (suggestions.isError) return <Failure error={suggestions.error} />;

  const rows = suggestions.data ?? [];
  // How much of each draft the model actually filled in. Reported because the
  // interesting number is how little: a suggestion pass that supplied
  // confidences and assumptions from a codebase would be fabricating them.
  const withConfidence = rows.filter((row) => draftOf(row).confidence != null).length;
  const withAssumptions = rows.filter(
    (row) => (draftOf(row).assumptions ?? []).length > 0,
  ).length;

  return (
    <div className="flex flex-col gap-4">
      <SectionHeading right={`${rows.length} ${status}`}>review queue</SectionHeading>

      <div className="meta flex items-baseline gap-4 text-ink-3">
        <span className="flex gap-2">
          {(["pending", "accepted", "rejected"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setStatus(value)}
              className={
                value === status ? "text-accent underline" : "text-ink-2 hover:text-ink"
              }
            >
              {value}
            </button>
          ))}
        </span>
        <span>
          {withConfidence} with a confidence · {withAssumptions} with assumptions
        </span>
      </div>

      {review.isError ? <Failure error={review.error} /> : null}

      {rows.length === 0 ? (
        <Empty title={`nothing ${status}`}>
          Propose drafts with{" "}
          <code className="kbd">memoryos decisions suggest --limit 10</code>. Nothing it
          finds is committed — every draft lands here, beside the passage it came from, and
          becomes a decision only when somebody accepts it.
        </Empty>
      ) : (
        <ul className="flex flex-col gap-6">
          {rows.map((row) => (
            <SuggestionRow
              key={row.id}
              row={row}
              busy={review.isPending}
              onReview={(verdict) => review.mutate({ id: row.id, verdict })}
              onEdit={() => navigate("/decisions/new", { state: prefillFrom(row) })}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function draftOf(row: Suggestion): Draft {
  return (row.draft ?? {}) as Draft;
}

/**
 * The draft, as the capture form's starting point.
 *
 * `confidence` and `expected_outcome` are deliberately not carried across even
 * when the model supplied them. Those two are claims about what somebody
 * believed, and a form that started them from a model's guess would make the
 * reviewer's job to disagree with a number rather than to state their own.
 */
function prefillFrom(row: Suggestion): PrefilledDraft {
  const draft = draftOf(row);
  return {
    acceptSuggestionId: row.id,
    question: draft.question ?? "",
    chosen: draft.chosen ?? "",
    reasoning: draft.reasoning ?? "",
    options: (draft.options ?? []).map((option) => ({
      description: option.description ?? "",
      rejected_because: option.rejected_because ?? "",
    })),
    assumptions: (draft.assumptions ?? []).map((item) => item.statement ?? ""),
  };
}

function SuggestionRow({
  row,
  busy,
  onReview,
  onEdit,
}: {
  row: Suggestion;
  busy: boolean;
  onReview: (verdict: "accept" | "reject") => void;
  onEdit: () => void;
}) {
  const draft = draftOf(row);
  const where =
    row.chunk_ordinal === null
      ? row.external_key
      : `${row.external_key}#${row.chunk_ordinal}`;

  return (
    <li className="border-b border-rule-strong pb-5">
      <div className="meta mb-2 flex flex-wrap items-baseline gap-3 text-ink-3">
        <Tag>{row.status}</Tag>
        <span className="text-ink-2">
          {row.source_name}:{where}
        </span>
        <span>{row.model_id}</span>
        <RelativeTime value={row.suggested_at} />
      </div>

      {/* Two columns on a wide screen, stacked on a narrow one. The draft never
          appears without the passage at any width. */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-2">
          <p className="meta-label text-ink-2">draft</p>
          <p className="prose-content text-sm text-ink">{draft.question}</p>
          <p className="prose-content text-sm text-accent">→ {draft.chosen}</p>
          {(draft.options ?? []).map((option, index) => (
            <div key={index} className="border-l-2 border-rule pl-3">
              <p className="prose-content text-sm text-ink">{option.description}</p>
              {option.rejected_because ? (
                <p className="meta text-ink-2">rejected: {option.rejected_because}</p>
              ) : null}
            </div>
          ))}
          {draft.reasoning ? (
            <p className="meta max-w-prose leading-relaxed text-ink-2">{draft.reasoning}</p>
          ) : null}
          <p className="meta text-ink-3">
            {/* Said explicitly rather than left as blank fields. These are the
                three the prompt is told to leave empty, and an interface that
                rendered nothing would look like a rendering bug. */}
            confidence {draft.confidence ?? "not stated"} · assumptions{" "}
            {(draft.assumptions ?? []).length} · expected{" "}
            {draft.expected_outcome ?? "not stated"}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <p className="meta-label text-ink-2">the passage it came from</p>
          <blockquote className="prose-content max-h-64 overflow-y-auto border-l-2 border-edge bg-surface p-3 text-sm leading-relaxed text-ink-2">
            {row.source_text}
          </blockquote>
        </div>
      </div>

      {row.status === "pending" ? (
        <div className="mt-3 flex gap-3">
          <button
            type="button"
            disabled={busy}
            onClick={() => onReview("accept")}
            className="meta-label border border-edge px-3 py-(--row-py) text-affirm"
          >
            accept
          </button>
          {/* Between the two verdicts, because it is the expected one: the
              reviewer has just read the passage and knows things the model
              could not. It opens the capture form prefilled, and submitting
              there accepts this suggestion in the same act. */}
          <button
            type="button"
            disabled={busy}
            onClick={onEdit}
            className="meta-label border border-rule px-3 py-(--row-py) text-accent"
          >
            edit
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onReview("reject")}
            className="meta-label border border-rule px-3 py-(--row-py) text-deny"
          >
            reject
          </button>
          <span className="meta self-center text-ink-3">
            Accepting writes a decision with this passage as `records` evidence. Confidence
            and assumptions stay empty until you add them.
          </span>
        </div>
      ) : null}
    </li>
  );
}
