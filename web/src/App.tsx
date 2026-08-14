/**
 * The shell: a thin masthead and three routes.
 *
 * No sidebar, no breadcrumbs, no cards. The masthead is one rule deep and gets
 * out of the way, because the content below it is the instrument.
 */

import { NavLink, Route, Routes } from "react-router-dom";

import { API_BASE } from "./api/client";
import { AgentPage } from "./features/agent/AgentPage";
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
import { SearchPage } from "./features/search/SearchPage";
import { SurfacingPage } from "./features/surfacing/SurfacingPage";
import { TimelinePage } from "./features/timeline/TimelinePage";

export function App() {
  return (
    <div className="mx-auto flex min-h-dvh max-w-6xl flex-col px-4 py-3">
      <header className="mb-4 flex items-baseline justify-between border-b border-rule-strong pb-2">
        <div className="flex items-baseline gap-5">
          <span className="meta-label text-ink">memory os</span>
          <nav className="flex items-baseline gap-4">
            <Tab to="/">search</Tab>
            {/* Beside search rather than under it. The two answer the same
                question in different currencies — search returns passages you
                judge, this returns a paragraph somebody else judged — and
                burying the second is how a reader forgets there was a choice. */}
            <Tab to="/ask">ask</Tab>
            <Tab to="/timeline">timeline</Tab>
            <Tab to="/decisions">decisions</Tab>
            <Tab to="/judgements">judgements</Tab>
            <Tab to="/corpus">corpus</Tab>
            {/* A tab, unlike reflections, and the difference is what the page
                contains. A reflection is a claim about somebody's judgement and
                volunteering it is the failure M5.4 was built to avoid. This is a
                record of what the system already interrupted them with, and
                hiding the place to say "that was noise" is how the dismissal
                rate stays flattering. */}
            <Tab to="/surfacing">surfacing</Tab>
          </nav>
        </div>
        {/* Which API is being read. On a tool with several environments, not
            knowing this is how you spend an afternoon confused. */}
        <span className="meta hidden text-faint sm:inline">{API_BASE}</span>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/ask" element={<AgentPage />} />
          <Route path="/timeline" element={<TimelinePage />} />
          <Route path="/memory/:id" element={<MemoryPage />} />
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
          {/* Reachable, never delivered. There is deliberately no nav tab for
              this route and nothing links to it from the home screen: a tool
              that volunteers claims about your judgement is one you stop
              trusting. You go and look at reflections, from the patterns view. */}
          <Route path="/decisions/reflections" element={<ReflectionsPage />} />
          <Route path="/decisions/:id" element={<DecisionPage />} />
          <Route path="/judgements" element={<JudgementsPage />} />
          <Route path="/corpus" element={<CorpusPage />} />
          <Route path="/surfacing" element={<SurfacingPage />} />
          <Route
            path="*"
            element={<p className="meta text-muted">no such page</p>}
          />
        </Routes>
      </main>
    </div>
  );
}

function Tab({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        `meta-label pb-0.5 ${
          isActive
            ? "border-b-2 border-edge text-amber"
            : "border-b-2 border-transparent text-muted hover:text-ink"
        }`
      }
    >
      {children}
    </NavLink>
  );
}
