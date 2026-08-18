/**
 * The radial canvas: one entity at the centre, its neighbours around it.
 *
 * Hand-written SVG and no graph library, deliberately. A force-directed layout
 * is the wrong tool for twelve nodes — it spends a physics simulation to
 * produce a different arrangement every time you look, which makes the picture
 * unmemorable and the screenshots irreproducible. Twelve nodes on a circle,
 * placed by arithmetic, are in the same position every time, and "the same
 * shape as last time" is most of what makes a graph view readable.
 *
 * **Twelve is the cap and it is a legibility limit, not a performance one.**
 * Past a dozen the labels collide, the edges cross enough to stop tracing, and
 * the picture becomes a hairball that says only "there is a lot" — which a
 * count already said, more precisely. When there are more, the twelve
 * best-connected are drawn and the rest are counted in a line beneath.
 *
 * Clicking a neighbour re-centres on it. That is the whole interaction: the
 * graph is walked one hop at a time, and each step is a complete, readable
 * picture rather than a pan across an infinite canvas.
 *
 * This component is pure — it renders the nodes and edges it is handed and owns
 * only which one is centred. It does no fetching, which is what lets it be
 * tested against fixtures while the API that would feed it does not yet exist.
 * See `GraphPage` for what the API currently returns, which is nothing.
 */

import { useState } from "react";

export interface GraphNode {
  id: string;
  name: string;
  /** `concept`, `technology`, `file`, `person` — the extractor's own vocabulary. */
  type: string;
  /** How many memories mention it. Sizes the node. */
  memories: number;
}

export interface GraphEdge {
  from: string;
  to: string;
  /** The predicate, drawn along the edge: IMPLEMENTS, SOLVES, PART_OF. */
  predicate: string;
}

/** See the file header. */
export const MAX_NODES = 12;

const SIZE = 460;
const CENTRE = SIZE / 2;
const RADIUS = 165;

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Which node starts at the centre. Defaults to the first. */
  focusId?: string;
  onFocus?: (id: string) => void;
}

export function EntityGraph({ nodes, edges, focusId, onFocus }: Props) {
  const [internal, setInternal] = useState<string | null>(null);
  const focus = internal ?? focusId ?? nodes[0]?.id ?? null;

  if (nodes.length === 0) return null;

  const centre = nodes.find((node) => node.id === focus) ?? nodes[0];

  // Everything one hop from the centre, in either direction. An edge is a claim
  // about a pair, not about an ordering, so a graph that only walked `from` to
  // `to` would hide half the neighbours of every node.
  const neighbourIds = new Set<string>();
  for (const edge of edges) {
    if (edge.from === centre.id) neighbourIds.add(edge.to);
    if (edge.to === centre.id) neighbourIds.add(edge.from);
  }

  const neighbours = nodes
    .filter((node) => neighbourIds.has(node.id))
    .sort((a, b) => b.memories - a.memories)
    .slice(0, MAX_NODES - 1);

  const hidden = neighbourIds.size - neighbours.length;

  // Placed by arithmetic, starting at the top and going clockwise. Reproducible
  // on every render, which a simulation is not.
  const placed = neighbours.map((node, index) => {
    const angle = (index / neighbours.length) * 2 * Math.PI - Math.PI / 2;
    return {
      node,
      x: CENTRE + RADIUS * Math.cos(angle),
      y: CENTRE + RADIUS * Math.sin(angle),
    };
  });

  function recentre(id: string) {
    setInternal(id);
    onFocus?.(id);
  }

  return (
    <div className="flex flex-col gap-2" data-testid="entity-graph">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="w-full max-w-[460px]"
        role="img"
        aria-label={`${centre.name}, with ${neighbours.length} connected ${
          neighbours.length === 1 ? "entity" : "entities"
        }`}
      >
        {/* Edges first, so nodes sit on top of them. */}
        {placed.map(({ node, x, y }) => {
          const edge = edges.find(
            (candidate) =>
              (candidate.from === centre.id && candidate.to === node.id) ||
              (candidate.to === centre.id && candidate.from === node.id),
          );
          return (
            <g key={`edge-${node.id}`}>
              <line
                x1={CENTRE}
                y1={CENTRE}
                x2={x}
                y2={y}
                /* Darker and thicker than the dark theme's. A 1px line at 35%
                   opacity glowed against a void and vanishes against the light
                   ground — measured by looking at it. `ink-3` at 1.5px is the
                   lightest an edge can be here and still be traceable across
                   the canvas. */
                stroke="var(--color-ink-3)"
                strokeWidth={1.5}
              />
              {edge ? (
                /* The predicate on the edge rather than in a legend. An edge
                   whose meaning is in a key somewhere else is an edge nobody
                   reads the meaning of. */
                <text
                  x={(CENTRE + x) / 2}
                  y={(CENTRE + y) / 2 - 4}
                  textAnchor="middle"
                  className="fill-ink-2 font-mono"
                  fontSize={8}
                  letterSpacing="0.08em"
                >
                  {edge.predicate}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* The neighbours. */}
        {placed.map(({ node, x, y }) => (
          <g
            key={node.id}
            className="cursor-pointer"
            onClick={() => recentre(node.id)}
            data-testid="graph-node"
            data-node-id={node.id}
          >
            <circle
              cx={x}
              cy={y}
              r={14}
              fill="var(--color-surface)"
              stroke="var(--color-ink-3)"
              strokeWidth={1.5}
            />
            <text
              x={x}
              y={y + 30}
              textAnchor="middle"
              className="fill-current font-mono text-ink"
              fontSize={10}
            >
              {node.name.length > 18 ? `${node.name.slice(0, 17)}…` : node.name}
            </text>
          </g>
        ))}

        {/* The centre, lit. */}
        <g data-testid="graph-centre" data-node-id={centre.id}>
          <circle
            cx={CENTRE}
            cy={CENTRE}
            r={22}
            /* The one accent on this canvas. A selected node is the thing the
               reader just clicked and the thing every edge is drawn from, so it
               is a result of an interaction rather than a position in a list —
               which is the line rule 1 draws. No drop shadow: the fill and the
               2px ring are enough on a light ground. */
            fill="var(--color-accent-soft)"
            stroke="var(--color-accent)"
            strokeWidth={2}
          />
          <text
            x={CENTRE}
            y={CENTRE + 44}
            textAnchor="middle"
            className="fill-current font-display text-ink"
            fontSize={15}
            fontWeight={600}
          >
            {centre.name}
          </text>
          <text
            x={CENTRE}
            y={CENTRE + 60}
            textAnchor="middle"
            className="fill-current font-mono text-ink-3"
            fontSize={9}
            letterSpacing="0.08em"
          >
            {centre.type.toUpperCase()}
          </text>
        </g>
      </svg>

      {hidden > 0 ? (
        <p className="meta text-ink-3">
          {hidden} further {hidden === 1 ? "connection" : "connections"} not drawn — the
          canvas holds {MAX_NODES} at a time so the labels stay readable.
        </p>
      ) : null}
    </div>
  );
}
