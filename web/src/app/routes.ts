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
 * Chat sits above the groups rather than inside one.
 *
 * It was the overview until M10.0, and the swap is the milestone: this is the
 * only view that both *writes* to the corpus and reads from it, so filing it
 * under "retrieval" would describe half of what it does. It is also the front
 * door, and a front door belongs above the map rather than on it.
 */
export const HOME: ViewRoute = {
  path: "/",
  label: "chat",
  blurb:
    "Type a thought and it is kept; ask a question and it is answered from everything kept, with citations.",
  group: "retrieval",
  aliases: ["say", "write", "note", "ask", "message", "front door"],
};

export const ROUTES: ViewRoute[] = [
  {
    path: "/overview",
    label: "overview",
    blurb: "What this system holds right now, in numbers that come from the corpus.",
    group: "retrieval",
    aliases: ["home", "start", "figures", "summary"],
  },
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
    path: "/sources",
    label: "sources",
    blurb:
      "The directories this reads from, and how to add one. Folders are still right for code.",
    group: "system",
    aliases: ["folders", "files", "directories", "sync", "repo", "ingest"],
  },
  {
    path: "/corpus",
    label: "corpus",
    blurb: "Counts and the standing health checks the CLI runs.",
    group: "system",
    aliases: ["stats", "doctor", "health"],
  },
];

/**
 * Real views that the sidebar does not name.
 *
 * Five working pages reachable only from small links in the corner of
 * `/decisions`. They are not in the sidebar because a nav with two levels for
 * one section out of four is a nav that has stopped being scannable — but
 * "jump to any view" has to mean any view, and a palette that cannot reach the
 * outcome queue is a palette with a hole in it.
 *
 * `/decisions/reflections` is deliberately absent from both. It is the one
 * route in this application that nothing is allowed to offer: a reflection is
 * a claim about somebody's judgement, and a tool that volunteers those is one
 * you stop trusting. It is reached from the patterns page, by going to look.
 * See `ReflectionsPage.test.tsx`.
 */
export const SUB_ROUTES: ViewRoute[] = [
  {
    path: "/decisions/new",
    label: "record a decision",
    blurb: "Write down a choice, its alternatives, and the confidence you hold now.",
    group: "reasoning",
    aliases: ["new decision", "decide"],
  },
  {
    path: "/decisions/review",
    label: "review queue",
    blurb: "Decisions the system found in the corpus, waiting to be accepted or rejected.",
    group: "reasoning",
    aliases: ["suggestions", "queue"],
  },
  {
    path: "/decisions/outcomes",
    label: "outcome queue",
    blurb: "How recorded decisions actually turned out, waiting to be confirmed.",
    group: "reasoning",
    aliases: ["results", "worked", "failed"],
  },
  {
    path: "/decisions/assumptions",
    label: "assumptions",
    blurb: "What decisions assumed, grouped, and which assumptions held.",
    group: "reasoning",
    aliases: ["held", "beliefs"],
  },
  {
    path: "/decisions/patterns",
    label: "patterns",
    blurb:
      "Behavioural patterns across decisions, with counter-evidence at equal weight, and calibration.",
    group: "reasoning",
    aliases: ["calibration", "habits"],
  },
];

/** Everything the sidebar walks, home first. */
export const ALL_ROUTES: ViewRoute[] = [HOME, ...ROUTES];

/** Everything the palette can reach, which is more than the sidebar shows. */
export const PALETTE_ROUTES: ViewRoute[] = [...ALL_ROUTES, ...SUB_ROUTES];

/** The routes in one sidebar group, in table order. */
export function inGroup(group: GroupId): ViewRoute[] {
  return ROUTES.filter((route) => route.group === group);
}

export function findRoute(path: string): ViewRoute | undefined {
  return ALL_ROUTES.find((route) => route.path === path);
}
