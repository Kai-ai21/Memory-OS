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
 *
 * **M9.1 split the table in two rather than cutting it down.** The Luminous
 * reference draws a sidebar of six items — search, timeline, graph, decisions,
 * insights, sources — and this application has fifteen views. Rendering only
 * the reference's six would have orphaned five working pages, which is the
 * precise failure this file was written to prevent; rendering all fifteen as
 * one flat list is the scannability problem the reference's six exist to solve.
 * So `PRIMARY` is the reference's nav, `SECONDARY` is everything else that
 * works, and both are in the sidebar with the second group set quieter.
 */

import type { IconName } from "../components/Icon";

export interface ViewRoute {
  path: string;
  /** The nav label. Lowercase throughout — see the mono label register. */
  label: string;
  /** One sentence: what this view answers. */
  blurb: string;
  group: GroupId;
  /** The glyph beside it in the sidebar. See `components/Icon`. */
  icon: IconName;
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

export type GroupId = "primary" | "secondary";

/** Group headings, in sidebar order. The primary group is unlabelled: six
 *  items under the wordmark need no heading, and a heading over them would be
 *  the first thing in the nav and the least useful. */
export const GROUPS: { id: GroupId; label: string }[] = [
  { id: "primary", label: "" },
  { id: "secondary", label: "more" },
];

/**
 * Chat sits above the groups rather than inside one.
 *
 * It was the overview until M10.0, and the swap is the milestone: this is the
 * only view that both *writes* to the corpus and reads from it, so filing it
 * under retrieval would describe half of what it does. It is also the front
 * door, and a front door belongs above the map rather than on it.
 */
export const HOME: ViewRoute = {
  path: "/",
  label: "chat",
  blurb:
    "Type a thought and it is kept; ask a question and it is answered from everything kept, with citations.",
  group: "primary",
  icon: "chat",
  aliases: ["say", "write", "note", "ask", "message", "front door"],
};

export const ROUTES: ViewRoute[] = [
  {
    path: "/search",
    label: "search",
    blurb:
      "Semantic search over every chunk, with the matched span highlighted and the ranking explained.",
    group: "primary",
    icon: "search",
    aliases: ["find", "query", "retrieve"],
  },
  {
    path: "/timeline",
    label: "timeline",
    blurb:
      "When the corpus happened — activity by period, stacked by kind, with the silences drawn.",
    group: "primary",
    icon: "timeline",
    aliases: ["activity", "gaps", "history"],
  },
  {
    path: "/graph",
    label: "graph",
    blurb:
      "The entity graph: what the corpus talks about, and which claims connect two things.",
    group: "primary",
    icon: "graph",
    aliases: ["entities", "relationships", "neo4j"],
  },
  {
    path: "/decisions",
    label: "decisions",
    blurb:
      "Decisions with their alternatives, assumptions and outcomes — and the patterns across them.",
    group: "primary",
    icon: "decisions",
    aliases: ["choices", "outcomes", "assumptions", "patterns"],
  },
  {
    path: "/insights",
    label: "insights",
    blurb:
      "What the system has concluded about how you work — and, for every dimension, whether it has enough evidence to conclude anything.",
    group: "primary",
    icon: "insights",
    planned: true,
    aliases: ["patterns", "reflections", "model", "beliefs", "calibration"],
  },
  {
    path: "/sources",
    label: "sources",
    blurb:
      "The directories this reads from, and how to add one. Folders are still right for code.",
    group: "primary",
    icon: "sources",
    aliases: ["folders", "files", "directories", "sync", "repo", "ingest"],
  },

  /* --- Everything else that works ---------------------------------------- */
  {
    path: "/overview",
    label: "overview",
    blurb: "What this system holds right now, in numbers that come from the corpus.",
    group: "secondary",
    icon: "document",
    aliases: ["home", "start", "figures", "summary"],
  },
  {
    path: "/agent",
    label: "agent",
    blurb:
      "Ask a question answered over several retrievals, with every sentence checked against what came back.",
    group: "secondary",
    icon: "chat",
    aliases: ["ask", "answer", "question"],
  },
  {
    path: "/model",
    label: "model",
    blurb:
      "What the system believes about you, every dimension listed — including the empty ones and why.",
    group: "secondary",
    icon: "insights",
    aliases: ["user model", "facets"],
  },
  {
    path: "/judgements",
    label: "judgements",
    blurb:
      "The golden set as it grows: which queries have been judged, and how lopsided each one is.",
    group: "secondary",
    icon: "decisions",
    aliases: ["golden set", "relevance", "labels"],
  },
  {
    path: "/surfacing",
    label: "surfacing",
    blurb:
      "What the system volunteered without being asked, and every time it decided to stay quiet.",
    group: "secondary",
    icon: "insights",
    aliases: ["proactive", "refusals", "interruptions"],
  },
  {
    path: "/corpus",
    label: "corpus",
    blurb: "Counts and the standing health checks the CLI runs.",
    group: "secondary",
    icon: "sources",
    aliases: ["stats", "doctor", "health"],
  },
];

/**
 * Real views that the sidebar does not name.
 *
 * Working pages reachable only from small links in the corner of `/decisions`
 * and `/insights`. They are not in the sidebar because a nav with two levels
 * for one section is a nav that has stopped being scannable — but "jump to any
 * view" has to mean any view, and a palette that cannot reach the outcome queue
 * is a palette with a hole in it.
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
    group: "secondary",
    icon: "add",
    aliases: ["new decision", "decide"],
  },
  {
    path: "/decisions/review",
    label: "review queue",
    blurb: "Decisions the system found in the corpus, waiting to be accepted or rejected.",
    group: "secondary",
    icon: "decisions",
    aliases: ["suggestions", "queue"],
  },
  {
    path: "/decisions/outcomes",
    label: "outcome queue",
    blurb: "How recorded decisions actually turned out, waiting to be confirmed.",
    group: "secondary",
    icon: "decisions",
    aliases: ["results", "worked", "failed"],
  },
  {
    path: "/decisions/assumptions",
    label: "assumptions",
    blurb: "What decisions assumed, grouped, and which assumptions held.",
    group: "secondary",
    icon: "decisions",
    aliases: ["held", "beliefs"],
  },
  {
    path: "/decisions/patterns",
    label: "patterns",
    blurb:
      "Behavioural patterns across decisions, with counter-evidence at equal weight, and calibration.",
    group: "secondary",
    icon: "insights",
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
