/**
 * Reflections: a pattern in prose, with every `[n]` a link to the decision it
 * came from.
 *
 * **This page is reached, never delivered.** There is no reflection on the home
 * screen, none in a search result, none on a decision, and no nav tab pointing
 * here — you get here from the patterns view, deliberately. A system that
 * volunteers claims about your judgement unprompted is a system you stop
 * trusting, and the way that happens by accident is a component quietly
 * rendering one inside something you were already looking at.
 *
 * Three things are on screen next to the prose and none of them are decoration.
 *
 * **The citation rate**, because a paragraph that is only 60% attributable
 * should say so beside itself rather than in a log. **The supporting and
 * contradicting counts**, at equal weight, because a claim drawn from six
 * decisions with two arguing against it is a different claim from one drawn from
 * six with none. **Uncited sentences, marked in place**, because a sentence
 * silently dropped from the middle of a paragraph leaves prose that reads as
 * complete and is not.
 *
 * Dismiss is not "hide". It stops this sentence being shown and stops its
 * pattern being written about again — you have to be able to say "this is wrong
 * about me" and have the system believe you.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type Reflection, type ReflectionCitation } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Empty, Failure, Loading, SectionHeading, Tag } from "../../components/primitives";
import { percent } from "../../lib/format";

export function ReflectionsPage() {
  const [includeDismissed, setIncludeDismissed] = useState(false);
  const client = useQueryClient();
  const reflections = useQuery({
    queryKey: ["reflections", includeDismissed],
    queryFn: () => api.reflections(includeDismissed),
  });

  const invalidate = async () => {
    await client.invalidateQueries({ queryKey: ["reflections"] });
  };
  const acknowledge = useMutation({
    mutationFn: (id: string) => api.acknowledgeReflection(id),
    onSuccess: invalidate,
  });
  const dismiss = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      api.dismissReflection(id, reason),
    onSuccess: invalidate,
  });

  if (reflections.isLoading) return <Loading rows={4} />;
  if (reflections.isError) return <Failure error={reflections.error} />;

  const rows = reflections.data ?? [];

  return (
    <div className="flex flex-col gap-5">
      <SectionHeading
        right={
          <span className="flex items-baseline gap-3">
            <button
              type="button"
              className={
                includeDismissed ? "text-accent underline" : "text-ink-2 hover:text-ink"
              }
              onClick={() => setIncludeDismissed((current) => !current)}
            >
              show dismissed
            </button>
            <Link className="text-accent underline" to="/decisions/patterns">
              patterns
            </Link>
          </span>
        }
      >
        reflections
      </SectionHeading>

      <p className="meta max-w-prose leading-relaxed text-ink-2">
        Written only for patterns whose evidence clears a confidence bar set above the one
        the pattern itself had to clear. A pattern below it produces no reflection at all
        rather than a hedged one — run{" "}
        <code className="kbd">memoryos reflect --all</code> to see what each would need.
      </p>

      {acknowledge.isError ? <Failure error={acknowledge.error} /> : null}
      {dismiss.isError ? <Failure error={dismiss.error} /> : null}

      {rows.length === 0 ? (
        <Empty title="no reflections">
          Nothing has cleared the bar to be described in prose. That is a result rather
          than a failure: an unfalsifiable claim about your own judgement is the most
          damaging thing this system can produce, so it is refused before a model is
          called rather than hedged afterwards. The{" "}
          <Link className="text-accent underline" to="/decisions/patterns">
            patterns view
          </Link>{" "}
          shows the evidence as it stands.
        </Empty>
      ) : (
        <ul className="flex flex-col gap-6">
          {rows.map((reflection) => (
            <ReflectionRow
              key={reflection.id}
              reflection={reflection}
              busy={acknowledge.isPending || dismiss.isPending}
              onAcknowledge={() => acknowledge.mutate(reflection.id)}
              onDismiss={(reason) => dismiss.mutate({ id: reflection.id, reason })}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function ReflectionRow({
  reflection,
  busy,
  onAcknowledge,
  onDismiss,
}: {
  reflection: Reflection;
  busy: boolean;
  onAcknowledge: () => void;
  onDismiss: (reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  const uncited = new Set(reflection.uncited);

  return (
    <li className="border-b border-rule-strong pb-5">
      <div className="meta mb-2 flex flex-wrap items-baseline gap-3 text-ink-3">
        <Tag>reflection</Tag>
        <span className="text-ink">
          cited{" "}
          {reflection.citation_rate === null
            ? "—"
            : percent(reflection.citation_rate)}
        </span>
        <span className="text-affirm">{reflection.support_count} supporting</span>
        <span className="text-deny">{reflection.contradiction_count} contradicting</span>
        <span>{reflection.model_id}</span>
        <DateStamp value={reflection.generated_at} provenance="declared" />
        {reflection.acknowledged_at ? <span>acknowledged</span> : null}
        {reflection.dismissed_at ? (
          <span className="text-deny">dismissed: {reflection.dismissed_reason}</span>
        ) : null}
      </div>

      <Prose
        text={reflection.text}
        citations={reflection.citations}
        uncited={uncited}
      />

      <p className="meta mt-3 max-w-prose text-ink-3">
        from the pattern: {reflection.pattern_statement}{" "}
        <Link className="text-accent underline" to="/decisions/patterns">
          evidence
        </Link>
      </p>

      {reflection.dismissed_at === null ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="meta-label border border-rule px-3 py-(--row-py) text-ink-2 disabled:opacity-40"
            disabled={busy || reflection.acknowledged_at !== null}
            onClick={onAcknowledge}
          >
            acknowledge
          </button>
          <input
            className="field flex-1"
            placeholder="why this is wrong about you"
            aria-label={`dismiss reason for ${reflection.id}`}
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          />
          <button
            type="button"
            className="meta-label border border-rule px-3 py-(--row-py) text-deny disabled:opacity-40"
            // A reason is required by the API and by a CHECK constraint.
            disabled={busy || reason.trim().length === 0}
            onClick={() => onDismiss(reason.trim())}
          >
            dismiss
          </button>
        </div>
      ) : null}
    </li>
  );
}

/**
 * The text, sentence by sentence, with markers turned into links.
 *
 * Split on the same boundary the server's check uses — terminal punctuation
 * followed by whitespace — so a sentence the server flagged as uncited is the
 * same span this marks. Two different splitters would mark the wrong sentence,
 * which is worse than not marking one at all.
 */
function Prose({
  text,
  citations,
  uncited,
}: {
  text: string;
  citations: ReflectionCitation[];
  uncited: Set<string>;
}) {
  const byMarker = new Map(citations.map((item) => [item.marker, item]));
  const sentences = text.split(/(?<=[.!?])\s+/).filter((part) => part.trim().length > 0);

  return (
    <p className="prose-content max-w-prose text-sm leading-relaxed text-ink">
      {sentences.map((sentence, index) => {
        const flagged = uncited.has(sentence.trim());
        return (
          <span
            key={`${index}-${sentence.slice(0, 12)}`}
            className={flagged ? "border-b border-dashed border-deny" : undefined}
            title={flagged ? "no citation: nothing in the record supports this" : undefined}
          >
            {renderMarkers(sentence, byMarker)}{" "}
          </span>
        );
      })}
    </p>
  );
}

function renderMarkers(
  sentence: string,
  byMarker: Map<number, ReflectionCitation>,
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let cursor = 0;
  let match = pattern.exec(sentence);

  while (match !== null) {
    parts.push(sentence.slice(cursor, match.index));
    const numbers = match[1].split(",").map((part) => Number(part.trim()));
    parts.push(
      <span key={`${match.index}`}>
        [
        {numbers.map((number, position) => {
          const citation = byMarker.get(number);
          return (
            <span key={number}>
              {position > 0 ? ", " : ""}
              {citation ? (
                <Link
                  className="text-accent underline"
                  to={`/decisions/${citation.decision_id}`}
                  title={`${
                    citation.relation === "supports" ? "argues for" : "argues against"
                  }: ${citation.decision_question}`}
                >
                  {number}
                </Link>
              ) : (
                // Only reachable if a citation row went missing under the text.
                // Shown rather than hidden: a marker that links nowhere is a
                // fact about the record, not a rendering detail.
                <span className="text-deny">{number}</span>
              )}
            </span>
          );
        })}
        ]
      </span>,
    );
    cursor = match.index + match[0].length;
    match = pattern.exec(sentence);
  }
  parts.push(sentence.slice(cursor));
  return parts;
}
