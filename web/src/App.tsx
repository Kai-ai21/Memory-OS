/**
 * The route table, inside the shell.
 *
 * `/` is the chat as of M10.0, and the overview it replaced moved to
 * `/overview`. That is the milestone in one line: typing is how things enter
 * this system, so the box you type into is what you land on.
 *
 * `/?q=…` is still forwarded to search — see `ChatPage` — because the argument
 * for keeping search state in the URL was that those links are worth something,
 * and they do not stop being worth something because the route under them
 * changed a second time.
 *
 * `/ask` became `/agent` for the same reason the sidebar exists: the nav, the
 * palette and the router read one table, and a route whose path and label
 * disagree is a route the palette cannot find. The old path redirects.
 */

import { Navigate, Route, Routes } from "react-router-dom";

import { Shell } from "./app/Shell";
import { GraphPlaceholder } from "./app/Placeholder";
import { findRoute } from "./app/routes";
import { AgentPage } from "./features/agent/AgentPage";
import { ChatPage } from "./features/chat/ChatPage";
import { CorpusPage } from "./features/corpus/CorpusPage";
import { AssumptionsPage } from "./features/decisions/AssumptionsPage";
import { DecisionForm } from "./features/decisions/DecisionForm";
import { DecisionPage } from "./features/decisions/DecisionPage";
import { DecisionsPage } from "./features/decisions/DecisionsPage";
import { OutcomeQueue } from "./features/decisions/OutcomeQueue";
import { PatternsPage } from "./features/decisions/PatternsPage";
import { ReflectionsPage } from "./features/decisions/ReflectionsPage";
import { ReviewQueue } from "./features/decisions/ReviewQueue";
import { JudgementsPage } from "./features/judgements/JudgementsPage";
import { MemoryPage } from "./features/memory/MemoryPage";
import { ModelPage } from "./features/model/ModelPage";
import { OverviewPage } from "./features/overview/OverviewPage";
import { SearchPage } from "./features/search/SearchPage";
import { SourcesPage } from "./features/sources/SourcesPage";
import { SurfacingPage } from "./features/surfacing/SurfacingPage";
import { TimelinePage } from "./features/timeline/TimelinePage";

export function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        {/* Where the overview lived until M10.0. */}
        <Route path="/overview" element={<OverviewPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/memory/:id" element={<MemoryPage />} />

        {/* The only route in this table with nothing behind it. */}
        <Route path="/graph" element={<GraphPlaceholder route={findRoute("/graph")!} />} />

        <Route path="/timeline" element={<TimelinePage />} />
        <Route path="/agent" element={<AgentPage />} />
        {/* Where the agent lived before M9.0. */}
        <Route path="/ask" element={<Navigate to="/agent" replace />} />
        <Route path="/model" element={<ModelPage />} />

        {/* The two literal segments before the parameterised one. React Router
            matches by specificity rather than order, so this is for the
            reader rather than for the router — the grouping should be
            obvious rather than lucky. */}
        <Route path="/decisions" element={<DecisionsPage />} />
        <Route path="/decisions/new" element={<DecisionForm />} />
        <Route path="/decisions/review" element={<ReviewQueue />} />
        <Route path="/decisions/outcomes" element={<OutcomeQueue />} />
        <Route path="/decisions/assumptions" element={<AssumptionsPage />} />
        <Route path="/decisions/patterns" element={<PatternsPage />} />
        {/* Reachable, never delivered. There is deliberately no nav entry for
            this route and nothing links to it from the overview: a tool that
            volunteers claims about your judgement is one you stop trusting.
            You go and look at reflections, from the patterns view. */}
        <Route path="/decisions/reflections" element={<ReflectionsPage />} />
        <Route path="/decisions/:id" element={<DecisionPage />} />

        <Route path="/judgements" element={<JudgementsPage />} />
        <Route path="/surfacing" element={<SurfacingPage />} />
        <Route path="/sources" element={<SourcesPage />} />
        <Route path="/corpus" element={<CorpusPage />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Shell>
  );
}

function NotFound() {
  return (
    <div className="flex max-w-(--width-reading) flex-col gap-3">
      <h1 className="display-page">No such page</h1>
      <p className="prose-lead">
        Nothing is routed here. The sidebar lists every view, and{" "}
        <span className="kbd">⌘K</span> will find one by name.
      </p>
    </div>
  );
}
