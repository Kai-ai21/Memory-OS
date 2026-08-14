/**
 * A multi-hop answer with its evidence showing.
 *
 * **The unsupported sentences render at the same weight as the rest**, in the
 * flow of the paragraph, underlined and labelled. Not greyed out, not collapsed,
 * not behind a "show verification" toggle — and that is the whole design of this
 * page rather than a styling preference.
 *
 * A toggle would be read exactly once. The reason a person believes an
 * ungrounded sentence is that it sits in a paragraph of grounded ones and looks
 * identical; the fix has to be visible while they are reading the sentence, not
 * available in a panel they could open afterwards. Making the mark quieter than
 * the text would be the same mistake in a smaller font: it would say the
 * unsupported part matters less, when it is the only part that needs attention.
 *
 * The same argument the surfacing page makes about showing refusals, and the
 * patterns page makes about counter-evidence. A tool that reports its own
 * failures at full volume is the only kind whose successes mean anything.
 *
 * When the verdict is `ungrounded` there is no paragraph to mark: the API
 * withheld the draft and sent a refusal, and this page says so plainly with the
 * support rate that caused it. The trajectory stays on screen either way,
 * because what was retrieved is the evidence for the refusal as much as for an
 * answer.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { api, type AgentAnswer, type AgentClaim } from "../../api/client";
import { Empty, Failure, Loading, Meta, SectionHeading, Tag } from "../../components/primitives";

export function AgentPage() {
  const [question, setQuestion] = useState("");
  const ask = useMutation({
    mutationFn: (asked: string) => api.agentAsk(asked),
  });

  return (
    <div className="space-y-5">
      <form
        className="flex items-baseline gap-2 border-b border-rule-strong pb-2"
        onSubmit={(event) => {
          event.preventDefault();
          const asked = question.trim();
          if (asked) ask.mutate(asked);
        }}
      >
        <label className="meta-label text-muted" htmlFor="agent-question">
          ask
        </label>
        <input
          id="agent-question"
          className="flex-1 bg-transparent text-ink outline-none placeholder:text-faint"
          placeholder="what mistakes have I repeated"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button className="meta-label text-muted hover:text-ink" type="submit">
          run
        </button>
      </form>

      {/* Several model calls and a corpus search each: seconds, not milliseconds,
          and a spinner with no explanation reads as broken. */}
      {ask.isPending ? <Loading rows={6} /> : null}
      {ask.isError ? <Failure error={ask.error} /> : null}
      {!ask.isPending && !ask.isError && !ask.data ? (
        <Empty title="Nothing asked yet">
          A question here is answered over several retrievals, and every sentence
          of the answer is checked against what those retrievals returned.
        </Empty>
      ) : null}

      {ask.data ? <Answer result={ask.data} /> : null}
    </div>
  );
}

function Answer({ result }: { result: AgentAnswer }) {
  const checked = result.verification;
  const refused = checked.refused;

  return (
    <div className="space-y-5">
      <Verdict result={result} />

      {result.answer === null ? (
        <p className="text-ink">
          No answer: {result.error ?? "the run did not complete"}.
        </p>
      ) : refused ? (
        <Refusal result={result} />
      ) : (
        <p className="leading-relaxed text-ink">
          {checked.claims.map((claim) => (
            <Sentence key={claim.sentence_index} claim={claim} />
          ))}
        </p>
      )}

      <Trajectory result={result} />
      <Sources result={result} />
    </div>
  );
}

/**
 * One sentence, marked in place when nothing retrieved supports it.
 *
 * `title` carries the excerpt behind a supported claim, which is the one piece
 * of verification detail that genuinely belongs in a tooltip: it explains a
 * sentence that is *fine*. Everything about a sentence that is not fine is in
 * the text itself.
 */
function Sentence({ claim }: { claim: AgentClaim }) {
  if (!claim.factual) {
    return <span>{claim.text} </span>;
  }
  if (!claim.supported) {
    return (
      <span className="decoration-wavy underline decoration-2 underline-offset-4">
        {claim.text}
        <span className="meta-label ml-1 text-ink">[unsupported]</span>{" "}
      </span>
    );
  }
  return (
    <span
      className={claim.support === "inferred" ? "border-b border-dotted border-rule" : undefined}
      title={claim.support_excerpt ?? undefined}
    >
      {claim.text}
      {claim.support === "inferred" ? (
        <span className="meta-label ml-1 text-muted">[inferred]</span>
      ) : null}{" "}
    </span>
  );
}

function Refusal({ result }: { result: AgentAnswer }) {
  const checked = result.verification;
  return (
    <div className="space-y-2 border-l-2 border-rule-strong pl-3">
      <p className="leading-relaxed text-ink">{result.answer}</p>
      <p className="meta text-muted">
        An answer was drafted and withheld: {percent(checked.support_rate)} of its{" "}
        {checked.factual_claims} factual claims traced to anything retrieved. What
        was retrieved is below.
      </p>
    </div>
  );
}

function Verdict({ result }: { result: AgentAnswer }) {
  const checked = result.verification;
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <Tag>{checked.verdict}</Tag>
      <Meta label="supported">{percent(checked.support_rate)}</Meta>
      <Meta label="direct">{percent(checked.direct_rate)}</Meta>
      <Meta label="claims">
        {checked.factual_claims} factual, {checked.connective_claims} connective
      </Meta>
      <Meta label="hops">{result.hops}</Meta>
      <Meta label="stopped">{result.stopped_because}</Meta>
      <Meta label="tokens">
        {result.cost.prompt_tokens + result.cost.completion_tokens}
      </Meta>
      {/* Never hidden and never rolled into the rate. A hop the run does not
          have is the one failure detectable with certainty. */}
      {checked.invalid_citations.length > 0 ? (
        <span className="meta text-ink">
          cited hop(s) {checked.invalid_citations.join(", ")}, which this run does
          not have
        </span>
      ) : null}
      {checked.unresolved_citations.length > 0 ? (
        <span className="meta text-ink">
          {checked.unresolved_citations.length} citation(s) no longer resolve
        </span>
      ) : null}
    </div>
  );
}

function Trajectory({ result }: { result: AgentAnswer }) {
  const acted = result.steps.filter((step) => step.tool !== null);
  if (acted.length === 0) {
    return (
      <div className="space-y-1">
        <SectionHeading>trajectory</SectionHeading>
        <p className="meta text-muted">
          No tool was called. Nothing in the corpus backs this answer.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <SectionHeading right={`${acted.length} hop(s)`}>trajectory</SectionHeading>
      <ol className="space-y-2">
        {acted.map((step, index) => (
          <li key={index} className="space-y-0.5">
            <div className="flex flex-wrap items-baseline gap-x-3">
              <span className="meta-label text-muted">hop {index + 1}</span>
              <span className="text-ink">{step.tool}</span>
              <span className="meta text-faint">
                {Object.entries(step.args)
                  .map(([name, value]) => `${name}=${JSON.stringify(value)}`)
                  .join(", ")}
              </span>
              {step.truncated ? <Tag>truncated</Tag> : null}
              {/* A hop that returned only things an earlier hop already had.
                  Worth seeing: two in a row is what stops the loop. */}
              {!step.novel ? <Tag>nothing new</Tag> : null}
            </div>
            <p className="meta whitespace-pre-wrap text-muted">{step.result}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Sources({ result }: { result: AgentAnswer }) {
  if (result.citations.length === 0) return null;
  return (
    <div className="space-y-1">
      <SectionHeading right={`${result.citations.length}`}>sources</SectionHeading>
      <ul className="space-y-0.5">
        {result.citations.map((citation, index) => (
          <li key={index} className="meta text-muted">
            {citation.source_name}::{citation.external_key}#{citation.chunk_ordinal} @
            {citation.char_start}-{citation.char_end} (v{citation.version})
          </li>
        ))}
      </ul>
    </div>
  );
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}
