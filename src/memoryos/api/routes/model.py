"""The user model over HTTP: every dimension, including the empty ones.

**The empty dimensions are the payload here, not the omission.** On a corpus this
size most of them are empty, and a response that returned only the dimensions
with facets would let a client render a page that looked complete while saying
almost nothing. So `assessments` carries all seven with the reason each is empty
and what would fill it, and the client renders those as gaps rather than skipping
them.

Read-only. Deriving, asserting and dismissing are CLI operations for now — the
first two write claims about a person and the third rejects one, and none of them
should be reachable from a page somebody has open by accident.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from memoryos.application import user_model
from memoryos.container import Container
from memoryos.domain.values import Dimension

router = APIRouter(tags=["model"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class EvidenceOut(BaseModel):
    kind: str
    ref_id: UUID
    relation: str


class FacetOut(BaseModel):
    id: UUID
    dimension: str
    statement: str
    # Null for an asserted facet, and null rather than 1.0: a goal somebody
    # stated is not a claim with a probability attached.
    confidence: float | None = None
    support_count: int
    contradiction_count: int
    # "derived" or "asserted". A client showing a stated goal beside a computed
    # weakness without distinguishing them would present the user's own words
    # back to them as a finding.
    origin: str
    detector: str | None = None
    superseded_by: UUID | None = None
    # M8.2. **Both, because `superseded_by` alone cannot express a withdrawal.**
    # A facet whose evidence went away is superseded with no replacement, so the
    # pointer is null and the timestamp is not — and a client reading only the
    # pointer would render a retired claim as the live one.
    superseded_at: str | None = None
    superseded_reason: str | None = None
    dismissed_at: str | None = None
    dismissed_reason: str | None = None
    evidence: list[EvidenceOut] = Field(default_factory=list)


class AssessmentOut(BaseModel):
    dimension: str
    facets: int
    # Empty when the dimension has facets; otherwise why it does not, in words
    # specific enough to act on.
    gap: str = ""
    best_support: int = 0


class ModelOut(BaseModel):
    # Keyed by dimension. A dimension with no facets is absent here and present
    # in `assessments`, which is the pairing the page renders.
    facets: dict[str, list[FacetOut]] = Field(default_factory=dict)
    assessments: list[AssessmentOut] = Field(default_factory=list)
    # Kept visible rather than filtered away: a rejected claim that vanished
    # would look like one nobody ever made, and the rejection is the more
    # interesting fact.
    dismissed: list[FacetOut] = Field(default_factory=list)


@router.get("/model", response_model=ModelOut)
async def show(
    container: ContainerDep, dimension: str | None = None
) -> ModelOut:
    wanted: Dimension | None = None
    if dimension is not None:
        try:
            wanted = Dimension(dimension)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in Dimension)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"{dimension!r} is not a dimension. Use one of: {allowed}",
            ) from exc

    view = await user_model.view(container.database.session_factory, dimension=wanted)
    return ModelOut(
        facets={
            name: [_facet_out(facet) for facet in facets]
            for name, facets in view.facets.items()
        },
        assessments=[
            AssessmentOut(
                dimension=item.dimension.value,
                facets=item.facets,
                gap=item.gap,
                best_support=item.best_support,
            )
            for item in view.assessments.values()
        ],
        dismissed=[_facet_out(facet) for facet in view.dismissed],
    )


@router.get("/model/{facet_id}/history", response_model=list[FacetOut])
async def history(facet_id: UUID, container: ContainerDep) -> list[FacetOut]:
    """Every version of one facet, oldest first.

    **This is what `superseded_by` is for.** A model that could only show its
    current state cannot answer the question that makes it worth having, which is
    what changed and when.
    """
    try:
        chain = await user_model.history(container.database.session_factory, facet_id)
    except user_model.UnknownFacet as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return [_facet_out(facet) for facet in chain]


def _facet_out(facet: user_model.FacetRow) -> FacetOut:
    return FacetOut(
        id=facet.id,
        dimension=facet.dimension,
        statement=facet.statement,
        confidence=facet.confidence,
        support_count=facet.support_count,
        contradiction_count=facet.contradiction_count,
        origin=facet.origin,
        detector=facet.detector,
        superseded_by=facet.superseded_by,
        superseded_at=(
            None if facet.superseded_at is None else facet.superseded_at.isoformat()
        ),
        superseded_reason=facet.superseded_reason,
        dismissed_at=None if facet.dismissed_at is None else facet.dismissed_at.isoformat(),
        dismissed_reason=facet.dismissed_reason,
        evidence=[
            EvidenceOut(kind=kind, ref_id=ref_id, relation=relation)
            for kind, ref_id, relation in facet.evidence
        ],
    )
