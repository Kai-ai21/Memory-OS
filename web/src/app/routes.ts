/**
 * Every view, described once.
 *
 * The sidebar, the command palette and the placeholder pages all read this
 * table. That is the point: three places listing the same routes is three
 * places to forget one, and the symptom is a view that exists and is
 * unreachable — which is exactly the state Phases 3 to 8 left this application
 * in, with six working pages and a masthead that named four of them.
 *
 * `blurb` is not decoration either. It is what the palette shows under a
 * result, what a placeholder page renders, and what tells somebody opening this
 * for the first time what a view is *for* before they click it.
 */

export interface ViewRoute {
  path: string;
  /** The nav label. Lowercase throughout — see the mono label register. */
  label: string;
  /** One sentence: what this view answers. */
  blurb: string;
  group: GroupId;
  /**
   * Nothing is built behind this yet.
   *
   * Marked rather than omitted. A nav that hides what does not exist tells the
   * reader the application is smaller than it is; one that renders a fake
   * screenshot tells them it is bigger. This says "planned", and the page says
   * what will be there and what it is waiting on.
   */
  planned?: boolean;
  /** Extra terms the palette matches on, for things called two names. */
  aliases?: string[];
}

export type GroupId = "retrieval" | "reasoning" | "feedback" | "system";

/** Group headings, in sidebar order. */
export const GROUPS: { id: GroupId; label: string }[] = [
  { id: "retrieval", label: "retrieval" },
  { id: "reasoning", label: "reasoning" },
  { id: "feedback", label: "feedback" },
  { id: "system", label: "system" },
];

/**
 * The overview sits above the groups rather than inside one.
 *
 * It is the only view that is about the whole system rather than one layer of
 * it, and filing it under a heading would make it look like a peer of search.
 */
export const HOME: ViewRoute = {
  path: "/",
  label: "overview",
  blurb: "What this system holds right now, and the one box that searches it.",
  group: "retrieval",
};

export const ROUTES: ViewRoute[] = [
  {
    path: "/search",
    label: "search",
    blurb:
      "Semantic search over every chunk, with the matched span highlighted and the ranking explained.",
    group: "retrieval",
    aliases: ["find", "query", "retrieve"],
  },
  {
    path: "/graph",
    label: "graph",
    blurb:
      "The entity graph: what the corpus talks about, and which claims connect two things.",
    group: "retrieval",
    planned: true,
    aliases: ["entities", "relationships", "neo4j"],
  },
  {
    path: "/timeline",
    label: "timeline",
    blurb:
      "When the corpus happened — activity by period, stacked by kind, with the silences drawn.",
    group: "retrieval",
    aliases: ["activity", "gaps", "history"],
  },
  {
    path: "/agent",
    label: "agent",
    blurb:
      "Ask a question answered over several retrievals, with every sentence checked against what came back.",
    group: "reasoning",
    aliases: ["ask", "answer", "question"],
  },
  {
    path: "/decisions",
    label: "decisions",
    blurb:
      "Decisions with their alternatives, assumptions and outcomes — and the patterns across them.",
    group: "reasoning",
    aliases: ["choices", "outcomes", "assumptions", "patterns"],
  },
  {
    path: "/model",
    label: "model",
    blurb:
      "What the system believes about you, every dimension listed — including the empty ones and why.",
    group: "reasoning",
    aliases: ["user model", "facets"],
  },
  {
    path: "/judgements",
    label: "judgements",
    blurb:
      "The golden set as it grows: which queries have been judged, and how lopsided each one is.",
    group: "feedback",
    aliases: ["golden set", "relevance", "labels"],
  },
  {
    path: "/surfacing",
    label: "surfacing",
    blurb:
      "What the system volunteered without being asked, and every time it decided to stay quiet.",
    group: "feedback",
    aliases: ["proactive", "refusals", "interruptions"],
  },
  {
    path: "/corpus",
    label: "corpus",
    blurb: "Counts, sources and the standing health checks the CLI runs.",
    group: "system",
    aliases: ["stats", "doctor", "health", "sources"],
  },
];

/** Everything the palette and the sidebar walk, home first. */
export const ALL_ROUTES: ViewRoute[] = [HOME, ...ROUTES];

/** The routes in one sidebar group, in table order. */
export function inGroup(group: GroupId): ViewRoute[] {
  return ROUTES.filter((route) => route.group === group);
}

export function findRoute(path: string): ViewRoute | undefined {
  return ALL_ROUTES.find((route) => route.path === path);
}
