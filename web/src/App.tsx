/**
 * The shell: a thin masthead and three routes.
 *
 * No sidebar, no breadcrumbs, no cards. The masthead is one rule deep and gets
 * out of the way, because the content below it is the instrument.
 */

import { NavLink, Route, Routes } from "react-router-dom";

import { API_BASE } from "./api/client";
import { CorpusPage } from "./features/corpus/CorpusPage";
import { JudgementsPage } from "./features/judgements/JudgementsPage";
import { MemoryPage } from "./features/memory/MemoryPage";
import { SearchPage } from "./features/search/SearchPage";

export function App() {
  return (
    <div className="mx-auto flex min-h-dvh max-w-6xl flex-col px-4 py-3">
      <header className="mb-4 flex items-baseline justify-between border-b border-rule-strong pb-2">
        <div className="flex items-baseline gap-5">
          <span className="meta-label text-ink">memory os</span>
          <nav className="flex items-baseline gap-4">
            <Tab to="/">search</Tab>
            <Tab to="/judgements">judgements</Tab>
            <Tab to="/corpus">corpus</Tab>
          </nav>
        </div>
        {/* Which API is being read. On a tool with several environments, not
            knowing this is how you spend an afternoon confused. */}
        <span className="meta hidden text-faint sm:inline">{API_BASE}</span>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/memory/:id" element={<MemoryPage />} />
          <Route path="/judgements" element={<JudgementsPage />} />
          <Route path="/corpus" element={<CorpusPage />} />
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
