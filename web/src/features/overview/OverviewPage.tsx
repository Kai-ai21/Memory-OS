/**
 * The first screen, and the one that has to work for two people at once.
 *
 * One of them uses this daily and wants the search box under their cursor. The
 * other has five minutes, no terminal, and no idea what any of this is. Those
 * pull in opposite directions — an explanation is noise to the first, and a bare
 * search box is meaningless to the second — so the page is ordered by what each
 * needs *first*: a paragraph, then the numbers that make the paragraph
 * checkable, then the box. The daily user reads none of it and tabs straight to
 * the input; the evaluator reads three sentences and knows what they are
 * looking at.
 *
 * **Every number on this page comes from the API.** Not one is written in the
 * source. A hardcoded figure on an overview is the worst kind: it is the first
 * thing anybody reads, it looks authoritative, and it starts lying the moment
 * the corpus changes. Where a number cannot be fetched the page says so rather
 * than showing a placeholder — a dash that means "not loaded" is honest, and a
 * zero that means the same thing is not.
 */

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api } from "../../api/client";
import { DateStamp } from "../../components/DateStamp";
import { Failure, SectionHeading } from "../../components/primitives";
import { count, percent } from "../../lib/format";

/** How many recent memories to list. Enough to see a shape, not a page of rows. */
const RECENT = 8;

export function OverviewPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  // Search used to live at `/`, so a bookmarked or pasted `/?q=…` is a real
  // thing that exists in somebody's notes. Forwarded rather than 404'd or
  // silently ignored, because the whole argument for keeping search state in
  // the URL was that those links are worth something.
  const carried = params.get("q");
  useEffect(() => {
    if (carried) navigate(`/search?q=${encodeURIComponent(carried)}`, { replace: true });
  }, [carried, navigate]);

  return (
    <div className="flex flex-col gap-10">
      <Statement />
      <Figures />
      <SearchBox />
      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <Recent />
        <Health />
      </div>
    </div>
  );
}

function Statement() {
  return (
    <header className="flex flex-col gap-4">
      <h1 className="display-page">A corpus that remembers why.</h1>
      <p className="prose-lead">
        MEMO is something you <Link to="/" className="text-accent underline">talk to</Link>,
        which can also read your files. A thought you type is kept as a memory and connected
        to everything already here that talks about the same things; a{" "}
        <Link to="/sources" className="text-accent underline">directory you point it at</Link>{" "}
        goes through the identical pipeline, which is still the right move for code. Nothing
        forgets its provenance: every chunk knows which document it came from and where in
        it, every date carries whether anybody actually stated it or the filesystem guessed,
        and every answer cites the passage it came from — and says so when a sentence is
        not supported by anything retrieved.
      </p>
    </header>
  );
}

/**
 * The size of the thing, in five numbers.
 *
 * Laid on a rule rather than in cards. Five bordered boxes with drop shadows is
 * the universal dashboard gesture and it makes five unrelated numbers look like
 * five buttons; a row of figures over a single hairline reads as what it is,
 * which is a table of contents for the corpus.
 */
function Figures() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, staleTime: 60_000 });
  const decisions = useQuery({
    queryKey: ["decisions", "all"],
    queryFn: () => api.decisions(),
    staleTime: 60_000,
  });

  if (stats.isError) return <Failure error={stats.error} />;

  return (
    <section className="flex flex-col gap-3" data-testid="figures">
      <dl className="grid grid-cols-2 gap-x-8 gap-y-6 border-t border-rule-strong pt-5 sm:grid-cols-3 lg:grid-cols-5">
        <Figure
          label="memories"
          value={stats.data?.current_memories}
          note={
            stats.data
              ? `${count(stats.data.memories)} versions in all`
              : undefined
          }
          to="/search"
        />
        <Figure
          label="chunks"
          value={stats.data?.chunks}
          note={stats.data ? `${percent(stats.data.coverage)} embedded` : undefined}
          to="/corpus"
        />
        <Figure
          label="entities"
          value={stats.data?.entities}
          note="extracted, deduplicated"
          to="/graph"
        />
        <Figure
          label="relationships"
          value={stats.data?.relationships}
          // A zero here is a fact about the corpus, not a missing number, and
          // it is the single most useful thing on this row: it explains why
          // every entity-scoped view is empty. Said in words rather than left
          // as a bare 0 for somebody to misread as a loading state.
          note={
            stats.data?.relationships === 0 ? "none extracted yet" : "distinct claims"
          }
          to="/graph"
        />
        <Figure
          label="decisions"
          value={decisions.data?.length}
          note={
            decisions.data
              ? `${count(decisions.data.filter((row) => row.assumptions > 0).length)} with assumptions`
              : undefined
          }
          to="/decisions"
        />
      </dl>
    </section>
  );
}

function Figure({
  label,
  value,
  note,
  to,
}: {
  label: string;
  /** Undefined until the request resolves. Never defaulted to zero. */
  value: number | undefined;
  note?: string;
  to: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <dd className="figure-value">
        {value === undefined ? (
          // Not a zero and not a spinner. A zero would be a claim about the
          // corpus that has not been checked yet.
          <span className="text-faint" aria-label="loading">
            —
          </span>
        ) : (
          count(value)
        )}
      </dd>
      <dt className="flex flex-col gap-0.5">
        <Link to={to} className="meta-label text-muted hover:text-accent">
          {label}
        </Link>
        {note ? <span className="meta text-faint">{note}</span> : null}
      </dt>
    </div>
  );
}

/**
 * The search box, still prominent, and no longer the primary act.
 *
 * M10.0 moved that to `/`, where the same box both keeps what you type and
 * answers what you ask. This one is search and only search — a query you want
 * ranked results for rather than an answer — which is a narrower and still
 * frequently correct thing to want.
 *
 * A rule and a line of type rather than a bordered control with a filled
 * button. This is the primary act of the application and the only thing on the
 * page that needs to look like it — and on a warm ruled sheet, weight and size
 * say "start here" more clearly than a coloured rectangle does.
 *
 * It does not run the search. It navigates to `/search`, which owns every piece
 * of search state in its URL; running results here would give the page a second
 * implementation of the same view and a query that could not be linked to.
 */
function SearchBox() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);

  return (
    <section className="flex flex-col gap-2">
      <form
        role="search"
        onSubmit={(event) => {
          event.preventDefault();
          const asked = query.trim();
          if (asked) navigate(`/search?q=${encodeURIComponent(asked)}`);
        }}
      >
        <label htmlFor="overview-search" className="meta-label">
          search the corpus
        </label>
        <input
          id="overview-search"
          ref={input}
          // The shell's `/` shortcut looks for this attribute, so the same key
          // focuses the same kind of box on whichever page has one.
          data-search-input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="what did I decide about chunk overlap?"
          className="field-prominent mt-1.5"
          aria-label="Search query"
          spellCheck={false}
          autoComplete="off"
          autoFocus
        />
      </form>
      <p className="meta text-faint">
        Semantic, not keyword — wording matters more than exact terms.{" "}
        <span className="kbd">/</span> focuses this from anywhere,{" "}
        <span className="kbd">⌘K</span> opens the palette.
      </p>
    </section>
  );
}

/**
 * What arrived most recently.
 *
 * By ingestion rather than by occurrence, deliberately, and they are very
 * different orders in this corpus: `occurred_at` is mostly a filesystem mtime,
 * so ordering by it sorts by when files were last touched on this disk. "What
 * did this system learn about most recently" is a question only `ingested_at`
 * answers.
 */
function Recent() {
  const memories = useQuery({
    queryKey: ["memories", RECENT],
    queryFn: () => api.memories(RECENT),
    staleTime: 60_000,
  });

  return (
    <section className="flex min-w-0 flex-col gap-2">
      <SectionHeading right="by ingestion">recent activity</SectionHeading>
      {memories.isError ? <Failure error={memories.error} /> : null}
      {memories.data?.length === 0 ? (
        <p className="meta text-faint">
          Nothing ingested. Register a source and sync it —{" "}
          <code className="kbd">memoryos sync</code>.
        </p>
      ) : null}
      <ul data-testid="recent">
        {(memories.data ?? []).map((memory) => (
          <li
            key={memory.id}
            className="flex items-baseline gap-3 border-b border-rule/60 py-1.5"
          >
            <Link
              to={`/memory/${memory.id}`}
              className="min-w-0 flex-1 truncate font-mono text-xs text-ink hover:text-accent hover:underline"
              title={memory.external_key}
            >
              {memory.external_key}
            </Link>
            <span className="meta shrink-0 text-faint">{memory.kind}</span>
            <span className="hidden shrink-0 sm:inline">
              <DateStamp value={memory.ingested_at} provenance="declared" />
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * Health, quietly — which here means "in the margin, in small type, and only
 * loud when something is wrong".
 *
 * The sidebar already carries a dot. This says what the dot means, because the
 * distinction `/health/ready` draws is a real one somebody will otherwise spend
 * an afternoon on: a graph that is down does not stop search or ingestion, and
 * an orchestrator that treated it as fatal would remove a working instance.
 */
function Health() {
  const ready = useQuery({ queryKey: ["ready"], queryFn: api.ready, staleTime: 30_000 });
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, staleTime: 60_000 });

  const unembedded = stats.data ? stats.data.chunks - stats.data.embedded_chunks : null;

  return (
    <section className="flex flex-col gap-2">
      <SectionHeading>health</SectionHeading>
      <dl className="flex flex-col gap-1.5" data-testid="overview-health">
        <Line
          label="api"
          ok={!ready.isError}
          value={ready.isError ? "unreachable" : "responding"}
        />
        <Line
          label="database"
          ok={ready.data?.database ?? null}
          value={
            ready.data
              ? ready.data.database
                ? `pgvector ${ready.data.pgvector_version ?? "—"}`
                : "unreachable"
              : "—"
          }
        />
        <Line
          label="graph"
          ok={ready.data?.graph ?? null}
          value={ready.data ? (ready.data.graph ? "reachable" : "unreachable") : "—"}
        />
        <Line
          label="embeddings"
          ok={unembedded === null ? null : unembedded === 0}
          value={
            unembedded === null
              ? "—"
              : unembedded === 0
                ? "all chunks embedded"
                : `${count(unembedded)} not embedded`
          }
        />
      </dl>
      <p className="meta text-faint">
        The standing checks live on{" "}
        <Link to="/corpus" className="text-accent underline">
          corpus
        </Link>
        . They tokenize with the real tokenizer, so they run on request.
      </p>
    </section>
  );
}

function Line({
  label,
  ok,
  value,
}: {
  label: string;
  /** Null while unknown — rendered neutral rather than as a failure. */
  ok: boolean | null;
  value: string;
}) {
  const tone = ok === null ? "text-faint" : ok ? "text-muted" : "text-deny";
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="meta-label">{label}</dt>
      <dd className={`meta ${tone}`}>{value}</dd>
    </div>
  );
}
