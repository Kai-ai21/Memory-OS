/**
 * One decision, read the way it was made.
 *
 * The order on screen is the order the record is worth in: the question, then
 * what was chosen, then **what was not** and why each lost, then the
 * assumptions. Options before reasoning because the alternatives are what turn
 * a description into a decision; assumptions in their own ruled block because
 * they are what M5.2 evaluates and what M5.3 finds patterns in, and burying
 * them under the prose would make the one field that generalises look like a
 * footnote.
 *
 * An empty assumptions block says so out loud rather than rendering nothing. A
 * decision that rests on nothing recorded is a finding about the record, and an
 * absent heading would read as an absent feature.
 */

import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import {
  api,
  type DecisionAssumption,
  type DecisionEvidence,
  type Outcome,
} from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Failure, Loading, Meta, SectionHeading, Tag } from "../../components/primitives";

export function DecisionPage() {
  const { id = "" } = useParams();
  const decision = useQuery({
    queryKey: ["decision", id],
    queryFn: () => api.decision(id),
    enabled: Boolean(id),
  });

  if (decision.isLoading) return <Loading rows={6} />;
  if (decision.isError) return <Failure error={decision.error} />;
  if (!decision.data) return null;

  const row = decision.data;
  const chosen = row.options.find((option) => option.was_chosen);
  const rejected = row.options.filter((option) => !option.was_chosen);

  return (
    <div className="flex flex-col gap-5">
      <div>
        <Link className="meta text-muted hover:text-ink" to="/decisions">
          ← decisions
        </Link>
        <h1 className="prose-content mt-2 text-lg text-ink">{row.question}</h1>
      </div>

      <div className="meta flex flex-wrap items-baseline gap-4 border-b border-rule-strong pb-2">
        <Tag>{row.status}</Tag>
        <Meta label="decided">
          <DateStamp value={row.decided_at} provenance={row.decided_at_source} showProvenance />
        </Meta>
        <Meta label="confidence">
          {row.confidence === null ? (
            <span className="text-faint">not recorded</span>
          ) : (
            /* Labelled "at the time" wherever it appears. The number's whole
               value is that it predates the outcome, and nothing in this app
               can change it. */
            <span>{row.confidence.toFixed(2)} at the time</span>
          )}
        </Meta>
      </div>

      <section className="flex flex-col gap-2">
        <SectionHeading>chosen</SectionHeading>
        <p className="prose-content text-sm text-ink">{chosen?.description ?? row.chosen}</p>
        {row.reasoning ? (
          <p className="prose-content max-w-prose text-sm leading-relaxed text-muted">
            {row.reasoning}
          </p>
        ) : null}
        {row.expected_outcome ? (
          <p className="meta max-w-prose text-faint">
            <span className="meta-label mr-2">expected</span>
            {row.expected_outcome}
          </p>
        ) : null}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${rejected.length} not taken`}>options</SectionHeading>
        {rejected.length === 0 ? (
          <p className="meta text-deny">
            No alternatives recorded. This should be impossible — the capture path refuses it.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {rejected.map((option) => (
              <li key={option.id} className="border-l-2 border-rule pl-3">
                <p className="prose-content text-sm text-ink">{option.description}</p>
                {option.rejected_because ? (
                  <p className="meta mt-0.5 max-w-prose leading-relaxed text-muted">
                    <span className="meta-label mr-2 text-deny">rejected</span>
                    {option.rejected_because}
                  </p>
                ) : (
                  <p className="meta mt-0.5 text-faint">no reason recorded</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${row.assumptions.length}`}>
          assumptions — what had to be true
        </SectionHeading>
        {row.assumptions.length === 0 ? (
          <p className="meta max-w-prose leading-relaxed text-deny">
            None recorded. Outcomes say a decision worked or it did not; assumptions say why,
            and only the why generalises — so there is nothing here for M5.2 to evaluate.
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {row.assumptions.map((assumption) => (
              <AssumptionRow key={assumption.id} assumption={assumption} />
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${row.outcomes.length}`}>
          outcomes — what happened
        </SectionHeading>
        {row.outcomes.length === 0 ? (
          <p className="meta max-w-prose leading-relaxed text-faint">
            Nothing recorded. That is not the same as <code className="kbd">too_early</code>,
            which is a verdict somebody reached by looking — this is a decision nobody has
            looked at. Record one with{" "}
            <code className="kbd">memoryos outcome &lt;id&gt; --verdict …</code>, or propose
            candidates with <code className="kbd">memoryos outcomes suggest</code>.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {/* Oldest first: the sequence is the information. "Worked, then
                failed" and "failed, then worked" are different stories, and
                reverse order would make the second look like the first. */}
            {row.outcomes.map((outcome) => (
              <OutcomeRow key={outcome.id} outcome={outcome} />
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionHeading right={`${row.evidence.length}`}>evidence</SectionHeading>
        {row.evidence.length === 0 ? (
          <p className="meta text-faint">
            Nothing linked. The decision stands on its own — evidence is a pointer into the
            corpus, not a precondition.
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {row.evidence.map((item) => (
              <EvidenceRow key={item.id} item={item} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function AssumptionRow({ assumption }: { assumption: DecisionAssumption }) {
  // Three states, not two. `null` means nobody has judged it yet, which is
  // deliberately distinct from "it broke" — an interface that collapsed them
  // would report every new decision as built on sand.
  const verdict =
    assumption.held === null ? "unevaluated" : assumption.held ? "held" : "broke";
  const tone =
    assumption.held === null
      ? "text-faint"
      : assumption.held
        ? "text-affirm"
        : "text-deny";
  return (
    <li className="flex items-baseline gap-3 border-b border-rule/60 pb-1">
      <span className={`meta-label w-24 shrink-0 ${tone}`}>{verdict}</span>
      <span className="prose-content flex-1 text-sm text-ink">{assumption.statement}</span>
      <span className="meta shrink-0 text-faint">
        {assumption.confidence === null ? "—" : assumption.confidence.toFixed(2)}
      </span>
    </li>
  );
}

/** The four verdicts, and the one that is deliberately not a colour. */
const VERDICT_TONE: Record<string, string> = {
  worked: "text-affirm",
  failed: "text-deny",
  mixed: "text-amber",
  // Neutral on purpose. `too_early` is not a lukewarm result, it is the absence
  // of one, and giving it a verdict colour would make a corpus of unresolved
  // decisions read as a corpus of middling ones.
  too_early: "text-faint",
};

function OutcomeRow({ outcome }: { outcome: Outcome }) {
  const inferred = outcome.evidence_kind === "inferred";
  return (
    <li className="border-l-2 border-rule pl-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <span className={`meta-label ${VERDICT_TONE[outcome.verdict] ?? "text-muted"}`}>
          {outcome.verdict}
        </span>
        {/* Shown on every outcome, not only inferred ones. A reader scanning a
            list has to be able to see which of these somebody watched happen —
            testimony and a model's reading are different kinds of claim, and
            rendering them identically asserts they are not. */}
        <span
          className={`meta border px-1 ${
            inferred ? "border-rule text-faint" : "border-edge text-amber"
          }`}
          title={
            inferred
              ? "inferred: a memory that occurred afterwards, judged by a model and accepted in review"
              : "declared: somebody observed this happen"
          }
        >
          {outcome.evidence_kind}
        </span>
        <DateStamp value={outcome.observed_at} provenance={outcome.observed_at_source} />
        {outcome.confidence !== null ? (
          <span className="meta text-faint">{outcome.confidence.toFixed(2)}</span>
        ) : null}
      </div>
      <p className="prose-content mt-0.5 max-w-prose text-sm text-ink">
        {outcome.description}
      </p>
      {outcome.evidence.map((item) => (
        <Link
          key={item.id}
          className="meta text-amber underline"
          to={`/memory/${item.memory_id}`}
        >
          {item.source_name}:{item.external_key}
        </Link>
      ))}
    </li>
  );
}

function EvidenceRow({ item }: { item: DecisionEvidence }) {
  const where =
    item.chunk_ordinal === null
      ? item.external_key
      : `${item.external_key}#${item.chunk_ordinal}`;
  return (
    <li className="flex items-baseline gap-3">
      {/* `informed` and `records` are kept apart on screen because they are
          different claims: one existed before the decision, one after it. */}
      <span className="meta-label w-24 shrink-0 text-muted">{item.relation}</span>
      <Link className="meta flex-1 text-amber underline" to={`/memory/${item.memory_id}`}>
        {item.source_name}:{where}
      </Link>
    </li>
  );
}
