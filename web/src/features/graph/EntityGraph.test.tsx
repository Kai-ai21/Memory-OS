/**
 * The canvas, and the one interaction it has.
 *
 * **Tested against fixtures rather than against the API, and that is the
 * finding rather than a shortcut.** There is no entity endpoint to render — see
 * `GraphPage` — and there are no relationships in the corpus even if there
 * were. This suite therefore pins the component's behaviour so that the day
 * either of those changes, the canvas is known to work and the only new
 * question is the shape of the response.
 *
 * The fixtures are shaped like the real extractor's output — lowercase concept
 * names, an uppercase predicate per edge — and deliberately do not reproduce
 * the reference's invented content.
 */

import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { EntityGraph, MAX_NODES, type GraphEdge, type GraphNode } from "./EntityGraph";

const NODES: GraphNode[] = [
  { id: "a", name: "job queue", type: "concept", memories: 15 },
  { id: "b", name: "postgres", type: "technology", memories: 128 },
  { id: "c", name: "skip locked", type: "concept", memories: 42 },
  { id: "d", name: "row level locks", type: "concept", memories: 8 },
];

const EDGES: GraphEdge[] = [
  { from: "a", to: "b", predicate: "USES" },
  { from: "a", to: "c", predicate: "REQUIRES" },
  // Deliberately pointing *into* c, so that re-centring on it has to find a
  // neighbour by the `to` side as well as the `from` side.
  { from: "d", to: "c", predicate: "SOLVES" },
];

describe("the radial canvas", () => {
  it("draws the focused entity at the centre and its neighbours around it", () => {
    render(<EntityGraph nodes={NODES} edges={EDGES} focusId="a" />);

    expect(within(screen.getByTestId("graph-centre")).getByText("job queue")).toBeInTheDocument();
    const around = screen.getAllByTestId("graph-node").map((node) => node.dataset.nodeId);
    expect(around.sort()).toEqual(["b", "c"]);
    // `d` connects to `c`, not to `a`, so it is two hops away and not drawn.
    expect(around).not.toContain("d");
  });

  it("re-centres when a neighbour is clicked", async () => {
    // The whole interaction of the view: the graph is walked one hop at a time,
    // and each step is a complete picture rather than a pan across a canvas.
    render(<EntityGraph nodes={NODES} edges={EDGES} focusId="a" />);

    await userEvent.click(
      screen.getAllByTestId("graph-node").find((node) => node.dataset.nodeId === "c")!,
    );

    const centre = screen.getByTestId("graph-centre");
    expect(centre.dataset.nodeId).toBe("c");
    expect(within(centre).getByText("skip locked")).toBeInTheDocument();

    // And the new centre brings its own neighbours, including the one that
    // pointed at it rather than away from it.
    const around = screen.getAllByTestId("graph-node").map((node) => node.dataset.nodeId);
    expect(around.sort()).toEqual(["a", "d"]);
  });

  it("names the predicate on the edge rather than in a legend", () => {
    render(<EntityGraph nodes={NODES} edges={EDGES} focusId="a" />);

    expect(screen.getByText("USES")).toBeInTheDocument();
    expect(screen.getByText("REQUIRES")).toBeInTheDocument();
  });

  it("caps the canvas and counts what it left out", () => {
    // Past a dozen the labels collide and the picture stops being traceable, so
    // the cap is a legibility limit — and a view that silently dropped the rest
    // would misreport how connected the centre is.
    const many: GraphNode[] = [
      { id: "hub", name: "postgres", type: "technology", memories: 128 },
      ...Array.from({ length: 20 }, (_, index) => ({
        id: `n${index}`,
        name: `entity ${index}`,
        type: "concept",
        memories: index,
      })),
    ];
    const manyEdges: GraphEdge[] = many
      .filter((node) => node.id !== "hub")
      .map((node) => ({ from: "hub", to: node.id, predicate: "MENTIONS" }));

    render(<EntityGraph nodes={many} edges={manyEdges} focusId="hub" />);

    expect(screen.getAllByTestId("graph-node")).toHaveLength(MAX_NODES - 1);
    expect(screen.getByText(/9 further connections not drawn/)).toBeInTheDocument();
  });

  it("renders nothing rather than an empty frame when there are no entities", () => {
    const { container } = render(<EntityGraph nodes={[]} edges={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
