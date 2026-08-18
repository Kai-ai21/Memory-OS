/**
 * What the system has concluded, and — mostly — what it has not.
 *
 * **The empty states are the content.** That is the reference's argument and it
 * is right, and on this corpus it is stronger than the reference knew: the
 * mockup draws six dimensions reading INSUFFICIENT EVIDENCE beside one
 * populated row scoring 0.62. Live, all seven read insufficient. There is no
 * populated row to draw, because nothing in the corpus has reached three
 * distinct observations on any dimension.
 *
 * A page that rendered only the sections with content would be three blank
 * panels, or worse, no page at all — and either says "there is nothing here to
 * think about", which is false. What is actually true is more specific and more
 * useful: there are twelve recorded decisions and a pattern needs three that
 * match; there are thirty-seven assumptions and twenty-five have been
 * evaluated; entity extraction has reached 2% of the corpus, which is why
 * workflows cannot be derived. Every one of those numbers is live, and every
 * gap sentence on the model rows is the API's own words rather than a generic
 * label this page invented.
 *
 * **Nothing here volunteers a reflection.** Reflections are claims about
 * somebody's judgement and this application does not offer them — see
 * `ReflectionsPage`. The section below reports the count and the threshold and
 * stops there, which is the same rule the sidebar follows by not naming the
 * route.
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../../api/client";
import { Failure, Loading } from "../../components/primitives";
import { count, percent } from "../../lib/format";

/**
 * The support a pattern needs before it is a pattern.
 *
 * `DEFAULT_MIN_SUPPORT` in `domain/patterns.py`. Duplicated here rather than
 * fetched because there is no endpoint that reports it, and a sentence saying
 * "a pattern requires at least 3" with no number behind it would be worse than
 * one that can go stale — this one is checked by the test that reads the same
 * copy. Named so that the day it moves, grep finds this line.
 */
const PATTERN_MIN_SUPPORT = 3;

export function InsightsPage() {
  const patterns = useQuery({ queryKey: ["patterns"], queryFn: () => api.patterns() });
  const reflections = useQuery({ queryKey: ["reflections"], queryFn: () => api.reflections() });
  const model = useQuery({ queryKey: ["model"], queryFn: () => api.model() });
  const decisions = useQuery({ queryKey: ["decisions"], queryFn: () => api.decisions() });
  const assumptions = useQuery({
    queryKey: ["assumption-stats"],
    queryFn: api.assumptionStats,
  });

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <h1 className="display-page">Insights</h1>
        <p className="prose-lead">
          Patterns and observations, drawn only from recorded decisions and observed
          behaviour. Nothing here is inferred from a single instance, and nothing is
          volunteered.
        </p>
      </header>

      <Section label="patterns">
        {patterns.isError ? <Failure error={patterns.error} /> : null}
        {patterns.isLoading ? <Loading rows={1} /> : null}
        {patterns.data?.length === 0 ? (
          <Empty title="No patterns yet.">
            <p>
              A pattern requires at least {PATTERN_MIN_SUPPORT} supporting decisions.{" "}
              {decisions.data ? (
                <>
                  You have{" "}
                  <span className="text-ink">
                    {count(decisions.data.length)}{" "}
                    {decisions.data.length === 1 ? "decision" : "decisions"}
                  </span>{" "}
                  recorded, and no group of them has reached {PATTERN_MIN_SUPPORT}.
                </>
              ) : (
                <>The decision count is not loaded.</>
              )}
            </p>
            {/* The reference names the size of the largest matching group. The
                API does not return it: `/patterns` sends patterns that cleared
                the bar and says nothing about the candidates that did not. What
                is known is that none cleared it, which is the sentence above.
                See the milestone report. */}
            {assumptions.data ? (
              <p className="mt-2">
                Separately, {count(assumptions.data.total)} assumptions are recorded across
                those decisions and {count(assumptions.data.evaluated)} have been
                evaluated, holding {percent(assumptions.data.hold_rate ?? 0)} of the time.
              </p>
            ) : null}
            <Action to="/decisions/new">record a decision</Action>
          </Empty>
        ) : null}
        {(patterns.data ?? []).map((pattern) => (
          <div key={pattern.id} className="glass flex flex-col gap-2 p-5">
            <div className="flex items-baseline gap-3">
              <span className="meta-label-on">{pattern.kind}</span>
              <span className="meta text-faint">
                {count(pattern.supporting.length)} supporting ·{" "}
                {count(pattern.contradicting.length)} against
              </span>
            </div>
            <p className="prose-content text-base">{pattern.statement}</p>
          </div>
        ))}
      </Section>

      <Section label="reflections">
        {reflections.isError ? <Failure error={reflections.error} /> : null}
        {reflections.data?.length === 0 ? (
          <Empty title="Nothing to reflect on yet.">
            <p>
              Reflections require patterns above the confidence threshold, and there are no
              patterns at all yet — so nothing has been a candidate.
            </p>
            <p className="mt-2 text-faint">
              This section reports the count and stops. A reflection is a claim about your
              judgement, and a tool that volunteers those is one you stop trusting; they
              are read by going to look, from{" "}
              <Link className="text-cyan underline" to="/decisions/patterns">
                patterns
              </Link>
              .
            </p>
          </Empty>
        ) : reflections.data ? (
          <p className="meta text-muted" data-testid="reflection-count">
            {count(reflections.data.length)}{" "}
            {reflections.data.length === 1 ? "reflection has" : "reflections have"} cleared
            the confidence threshold. They are not shown here — see{" "}
            <Link className="text-cyan underline" to="/decisions/patterns">
              patterns
            </Link>
            .
          </p>
        ) : null}
      </Section>

      <Section
        label="model"
        right={
          model.data
            ? `${model.data.assessments.filter((item) => item.facets > 0).length} of ${
                model.data.assessments.length
              } derived`
            : undefined
        }
      >
        {model.isError ? <Failure error={model.error} /> : null}
        {model.isLoading ? <Loading rows={4} /> : null}
        {model.data ? (
          <ul className="flex flex-col" data-testid="model-dimensions">
            {model.data.assessments.map((assessment) => (
              <Dimension
                key={assessment.dimension}
                dimension={assessment.dimension}
                facets={assessment.facets}
                gap={assessment.gap}
                bestSupport={assessment.best_support}
              />
            ))}
          </ul>
        ) : null}
      </Section>
    </div>
  );
}

function Section({
  label,
  right,
  children,
}: {
  label: string;
  right?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between border-b border-rule pb-2">
        <h2 className="meta-label-on">{label}</h2>
        {right ? <span className="meta text-faint">{right}</span> : null}
      </div>
      {children}
    </section>
  );
}

/**
 * An empty state built to be read rather than skipped.
 *
 * The reference's structure — a headline sentence at display size, the
 * explanation under it, one action — and it is the right structure because the
 * headline is the finding. "No patterns yet." is a fact about the corpus, not
 * an apology for a blank panel.
 */
function Empty({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="glass flex flex-col gap-3 p-6" data-testid="empty">
      <h3 className="display text-2xl">{title}</h3>
      <div className="prose-content max-w-prose text-muted">{children}</div>
    </div>
  );
}

function Action({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      className="meta-label-on mt-4 inline-flex items-center gap-2 hover:underline"
    >
      {children} <span aria-hidden>→</span>
    </Link>
  );
}

/**
 * One dimension of the model: what it holds, or precisely why it holds nothing.
 *
 * The gap sentence is the API's, verbatim. It is the difference between this row
 * and the reference's, which reads INSUFFICIENT EVIDENCE on every empty
 * dimension and therefore says the same thing six times — where the real ones
 * differ in a way that matters. `goals` is empty because goals are never
 * inferred and must be stated; `learning_style` because no deriver exists at
 * all; `workflows` because extraction has covered 2% of the corpus. Three
 * different problems with three different answers, flattened by one label.
 */
function Dimension({
  dimension,
  facets,
  gap,
  bestSupport,
}: {
  dimension: string;
  facets: number;
  gap: string;
  bestSupport: number;
}) {
  const empty = facets === 0;

  return (
    <li
      className="flex flex-col gap-2 border-b border-rule py-4 last:border-b-0"
      data-testid="dimension"
      data-dimension={dimension}
      data-empty={empty}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="prose-content text-base text-ink">
          {dimension.replace(/_/g, " ")}
        </span>
        {empty ? (
          <span className="meta-label shrink-0" data-testid="insufficient">
            insufficient evidence
          </span>
        ) : (
          <span className="meta shrink-0 text-cyan">
            {count(facets)} {facets === 1 ? "facet" : "facets"}
          </span>
        )}
      </div>
      {empty ? (
        <p className="meta max-w-prose text-faint" data-testid="gap">
          {gap}
          {bestSupport > 0 ? (
            <>
              {" "}
              The closest candidate reached{" "}
              <span className="text-muted">
                {bestSupport} distinct observation{bestSupport === 1 ? "" : "s"}
              </span>
              .
            </>
          ) : null}
        </p>
      ) : null}
    </li>
  );
}
