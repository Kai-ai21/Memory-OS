/**
 * What the system believes about you, and — mostly — what it cannot yet say.
 *
 * **The empty dimensions are the page.** On this corpus every derivable
 * dimension is empty, and a view that rendered only the sections with content
 * would be a blank screen implying the feature was broken, or worse, four
 * sections implying the model was complete. So every dimension gets a heading
 * whether or not it has facets, and an empty one carries the reason it is empty
 * and what would fill it — "entity extraction has reached 12 of 253 memories" is
 * something a person can act on; a missing section is not.
 *
 * This is the same argument the surfacing page makes about showing refusals and
 * the patterns page makes about counter-evidence, and it matters more here than
 * in either. A page of claims about somebody's goals and weaknesses is the exact
 * shape of a horoscope, and the thing that separates this from one is that every
 * statement carries its support count and every absence carries its reason.
 *
 * Dismissed facets stay on screen, struck through, with the reason beside them.
 * A rejected claim that vanished would look like one nobody ever made, and the
 * rejection is the more interesting half.
 *
 * Read-only. Deriving, asserting and dismissing are commands somebody types:
 * two of them write claims about a person and the third rejects one, and none
 * should be one click away on a page left open in a tab.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api, type Assessment, type Facet } from "../../api/client";
import { Failure, Loading, Meta, SectionHeading, Tag } from "../../components/primitives";

export function ModelPage() {
  const model = useQuery({ queryKey: ["model"], queryFn: () => api.model() });

  if (model.isLoading) return <Loading rows={7} />;
  if (model.isError) return <Failure error={model.error} />;

  const data = model.data;
  if (!data) return null;

  const derived = data.assessments.filter((item) => item.facets > 0).length;

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <SectionHeading right={`${derived} of ${data.assessments.length} dimensions`}>
          user model
        </SectionHeading>
        <p className="meta text-ink-2">
          Every dimension is listed. One with nothing above the evidence bar says
          so and says why — a stated gap is information, an omitted section looks
          like an oversight.
        </p>
      </div>

      {data.assessments.map((assessment) => (
        <Dimension
          key={assessment.dimension}
          assessment={assessment}
          facets={data.facets[assessment.dimension] ?? []}
        />
      ))}

      {data.dismissed.length > 0 ? (
        <div className="space-y-2">
          <SectionHeading right={`${data.dismissed.length}`}>dismissed</SectionHeading>
          {data.dismissed.map((facet) => (
            <div key={facet.id} className="space-y-0.5">
              <p className="text-ink-2 line-through">{facet.statement}</p>
              <p className="meta text-ink-3">{facet.dismissed_reason}</p>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Dimension({
  assessment,
  facets,
}: {
  assessment: Assessment;
  facets: Facet[];
}) {
  return (
    <div className="space-y-2">
      <SectionHeading
        right={facets.length > 0 ? `${facets.length}` : "insufficient evidence"}
      >
        {assessment.dimension.replace(/_/g, " ")}
      </SectionHeading>
      {facets.length === 0 ? (
        <Gap assessment={assessment} />
      ) : (
        facets.map((facet) => <FacetRow key={facet.id} facet={facet} />)
      )}
    </div>
  );
}

/**
 * An empty dimension, rendered as a gap with its cause.
 *
 * At the same weight as a facet rather than greyed out: this is the majority of
 * the page and the most honest thing on it, and styling it as secondary would
 * say the absences matter less than the one or two claims that cleared the bar.
 */
function Gap({ assessment }: { assessment: Assessment }) {
  return (
    <div className="space-y-1 border-l-2 border-rule pl-3">
      <p className="text-ink">{assessment.gap}</p>
      {assessment.best_support > 0 ? (
        <p className="meta text-ink-2">
          The closest candidate reached {assessment.best_support} distinct
          observation{assessment.best_support === 1 ? "" : "s"}.
        </p>
      ) : null}
    </div>
  );
}

function FacetRow({ facet }: { facet: Facet }) {
  const [history, setHistory] = useState<Facet[] | null>(null);
  const stated = facet.origin === "asserted";

  return (
    <div className="space-y-1 border-l-2 border-rule-strong pl-3">
      <p className="text-ink">{facet.statement}</p>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        {/* A stated goal and a computed weakness must not look the same: showing
            somebody their own words back as a finding is the cheapest way for a
            page like this to lose their trust. */}
        <Tag>{stated ? "stated" : (facet.detector ?? "derived")}</Tag>
        {stated ? null : (
          <Meta label="confidence">
            {facet.confidence === null ? "—" : facet.confidence.toFixed(2)}
          </Meta>
        )}
        <Meta label="support">{facet.support_count}</Meta>
        {/* Never hidden. A facet with nine for and eight against is not a strong
            claim, and a page that showed only the nine would present it as one. */}
        <Meta label="against">{facet.contradiction_count}</Meta>
        <button
          className="meta-label text-ink-2 hover:text-ink"
          onClick={async () =>
            setHistory(history ? null : await api.facetHistory(facet.id))
          }
        >
          {history ? "hide history" : "history"}
        </button>
      </div>
      {facet.evidence.length > 0 ? (
        <p className="meta text-ink-3">
          {facet.evidence
            .map((item) => `${item.kind}:${item.ref_id.slice(0, 8)} (${item.relation})`)
            .join("  ")}
        </p>
      ) : null}
      {history ? <History chain={history} current={facet.id} /> : null}
    </div>
  );
}

/**
 * Every version of one facet, oldest first.
 *
 * The reason `superseded_by` exists: a model that could only show its current
 * state cannot answer what changed and when, which is the question that makes
 * keeping a model of a person worth the risk of having one.
 */
function History({ chain, current }: { chain: Facet[]; current: string }) {
  return (
    <ol className="space-y-1 border-l border-rule pl-3">
      {chain.map((version, index) => (
        <li key={version.id} className="meta text-ink-2">
          <span className="meta-label mr-2">v{index + 1}</span>
          {version.statement}
          {version.id === current ? (
            <span className="meta-label ml-2 text-ink">current</span>
          ) : null}
        </li>
      ))}
      {chain.length === 1 ? (
        <li className="meta text-ink-3">
          One version: this facet has never been revised.
        </li>
      ) : null}
    </ol>
  );
}
