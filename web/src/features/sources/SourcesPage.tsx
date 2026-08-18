/**
 * Folders, demoted but not diminished.
 *
 * M10.0 made typing the default way things enter this corpus, and moved source
 * management here from the corpus report. Demoted is the right word for what
 * happened and the wrong word for what this page is: pointing at a repository is
 * still the correct move for code, and a directory of a thousand files is not
 * something anybody is going to retype. What changed is which one you reach
 * first.
 *
 * The read half of this page came from `/corpus`, where it was a table nobody
 * could act on. The write half is new and is the reason the move is worth
 * making: registering a source and triggering a sync were API endpoints the CLI
 * used and the interface did not, so "keep it fully working" meant building the
 * two controls that were missing rather than relocating the ones that existed.
 *
 * A sync is enqueued, never run in the request — a directory takes minutes, and
 * a button that blocked on it would time out having done work nobody could
 * resume. The row says when it last finished instead.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type Source } from "../../api/client";
import { Empty, Failure, Loading, SectionHeading } from "../../components/primitives";
import { count, timestamp } from "../../lib/format";

export function SourcesPage() {
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });

  if (sources.isLoading) return <Loading rows={3} />;

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-2">
        <h1 className="display-page">Where else this reads from.</h1>
        <p className="prose-lead">
          A source is a directory this system walks and keeps in step with. Files
          go through exactly the same pipeline a typed message does — hashed,
          parsed, chunked, embedded, and connected through the things they talk
          about — so a repository and a thought end up in the same corpus,
          searchable together.
        </p>
      </header>

      {sources.isError ? <Failure error={sources.error} /> : null}
      {sources.data?.length === 0 ? (
        <Empty title="No sources registered">
          Nothing is being read from disk. That is a perfectly good state — the
          chat box fills the corpus on its own — but pointing at a repository is
          still the right move for code.
        </Empty>
      ) : null}

      {sources.data && sources.data.length > 0 ? (
        <Registered sources={sources.data} />
      ) : null}

      <Register />
    </div>
  );
}

function Registered({ sources }: { sources: Source[] }) {
  return (
    <section className="flex flex-col gap-2">
      <SectionHeading right={`${sources.length} registered`}>sources</SectionHeading>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-rule">
              {[
                "name",
                "kind",
                "memories",
                "chunks",
                "last sync",
                "last full sync",
                "sync",
                "manage",
              ].map((heading) => (
                <th key={heading} className="meta-label py-1 pr-4 font-normal">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sources.map((source) => (
              <Row key={source.id} source={source} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="meta text-ink-3">
        A full sync is the only one that can notice a deletion: a file that is
        gone produces no observation, so the complete set has to be compared
        against the complete known set. That is why the two dates are separate.
      </p>
    </section>
  );
}

function Row({ source }: { source: Source }) {
  const client = useQueryClient();
  const sync = useMutation({
    mutationFn: (full: boolean) => api.syncSource(source.id, full),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["sources"] }),
  });

  // The chat source is not walked and has no root to walk. Syncing it would
  // enqueue a job for a connector that does not exist, so the control is absent
  // rather than present and broken.
  const walkable = source.kind === "filesystem";

  return (
    <tr className="border-b border-rule/60" data-testid="source-row">
      <td className="meta py-1 pr-4 text-ink">{source.name}</td>
      <td className="meta py-1 pr-4 text-ink-3">{source.kind}</td>
      <td className="meta py-1 pr-4">{count(source.memories)}</td>
      <td className="meta py-1 pr-4">{count(source.chunks)}</td>
      <td className="meta py-1 pr-4 text-ink-3">{timestamp(source.last_sync_at)}</td>
      <td className="meta py-1 pr-4 text-ink-3">{timestamp(source.last_full_sync_at)}</td>
      <td className="py-1">
        {walkable ? (
          <span className="flex gap-2">
            <button
              type="button"
              className="btn"
              onClick={() => sync.mutate(false)}
              disabled={sync.isPending}
            >
              {sync.isPending ? "queued…" : "sync"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => sync.mutate(true)}
              disabled={sync.isPending}
              title="Walks everything and reconciles deletions"
            >
              full
            </button>
          </span>
        ) : (
          <span className="meta text-ink-3" title="Messages are pushed, not walked">
            not walked
          </span>
        )}
        {sync.isError ? <Failure error={sync.error} /> : null}
        {sync.isSuccess ? (
          <span className="meta ml-2 text-ink-2">
            {sync.data?.job_id ? "queued" : "already running"}
          </span>
        ) : null}
      </td>
      <td className="py-1">
        <SourceOperations source={source} />
      </td>
    </tr>
  );
}

/**
 * Re-index, export, and the one destructive operation in this table.
 *
 * **Deleting a source is the most destructive thing this product can do**, so it is
 * not a button beside `sync`. It opens a panel that names the exact counts, states
 * what the append-only log keeps, and requires the source's own name to be typed —
 * not `y`, and not a second click, because the name is the only answer that cannot
 * be given by accident.
 */
function SourceOperations({ source }: { source: Source }) {
  const client = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const reindex = useMutation({
    mutationFn: () => api.reindexSource(source.id),
    onSuccess: (result) =>
      setNote(
        `${result.memories} memory(s) queued for re-parsing, re-chunking and ` +
          `re-embedding — ${result.jobs} new job(s). Run the worker to drain them.`,
      ),
  });

  return (
    <span className="flex flex-col gap-1">
      <span className="flex gap-2">
        <button
          type="button"
          className="btn"
          onClick={() => reindex.mutate()}
          disabled={reindex.isPending}
          title="Re-parse, re-chunk and re-embed everything from this source. Nothing is re-read and no event is appended."
        >
          {reindex.isPending ? "queueing…" : "re-index"}
        </button>
        {/* A plain link, not a fetch: the browser's own download is what somebody
            wants here, and streaming a corpus-sized file through JavaScript to
            hand it back to the browser would buy nothing and cost memory. */}
        <a className="btn" href={api.sourceExportUrl(source.id)} download>
          export
        </a>
        <button
          type="button"
          className="btn text-deny"
          onClick={() => setConfirming((was) => !was)}
        >
          delete…
        </button>
      </span>
      {reindex.isError ? <Failure error={reindex.error} /> : null}
      {confirming ? (
        <DeleteSourcePanel
          source={source}
          onCancel={() => setConfirming(false)}
          onDone={(message) => {
            setNote(message);
            setConfirming(false);
            void client.invalidateQueries({ queryKey: ["sources"] });
          }}
        />
      ) : null}
      {note ? (
        <span className="meta text-accent" role="status">
          {note}
        </span>
      ) : null}
    </span>
  );
}

/**
 * The counts, the log note, and the typed confirmation.
 *
 * The counts are read when the panel opens, and the item count is sent back with
 * the deletion so the API can refuse if the corpus moved in between — a sync
 * landing mid-dialog makes this a different operation from the one somebody agreed
 * to.
 */
function DeleteSourcePanel({
  source,
  onDone,
  onCancel,
}: {
  source: Source;
  onDone: (note: string) => void;
  onCancel: () => void;
}) {
  const [typed, setTyped] = useState("");
  const scope = useQuery({
    queryKey: ["source-deletion-scope", source.id],
    queryFn: () => api.sourceDeletionScope(source.id),
    staleTime: 0,
    gcTime: 0,
  });
  const remove = useMutation({
    mutationFn: () => api.deleteSource(source.id, scope.data?.items ?? 0),
    onSuccess: (result) => onDone(result.detail),
  });

  return (
    <div
      className="flex max-w-prose flex-col gap-2 border-l-2 border-deny bg-ground p-3"
      role="alertdialog"
      aria-label={`Delete the source ${source.name}`}
    >
      <p className="text-ink">
        Permanently delete <span className="font-mono">{source.name}</span> and
        everything from it?
      </p>
      {scope.isPending ? (
        <p className="meta text-ink-3">counting what this would remove…</p>
      ) : null}
      {scope.data ? (
        <ul className="flex flex-col gap-0.5" data-testid="source-deletion-scope">
          <li className="meta text-ink">{scope.data.items} item(s)</li>
          <li className="meta text-ink">{scope.data.memories} memory version(s)</li>
          <li className="meta text-ink">
            {scope.data.chunks} chunk(s), {scope.data.embedded_chunks} with vectors
          </li>
          <li className="meta text-ink">
            {scope.data.mentions} entity mention(s)
            {scope.data.orphaned_entities
              ? `, leaving ${scope.data.orphaned_entities} unreachable`
              : ""}
          </li>
          {scope.data.tags ? (
            <li className="meta text-ink">{scope.data.tags} tag(s)</li>
          ) : null}
          {scope.data.turns ? (
            <li className="meta text-ink">
              {scope.data.turns} conversation turn(s)
            </li>
          ) : null}
          {scope.data.evidence ? (
            <li className="meta text-ink">
              {scope.data.evidence} decision evidence link(s)
            </li>
          ) : null}
          <li className="meta text-ink">
            {scope.data.blobs} stored file(s)
            {scope.data.shared_blobs
              ? `, and ${scope.data.shared_blobs} kept because something else uses them`
              : ""}
          </li>
        </ul>
      ) : null}
      {scope.data ? (
        <p className="meta text-ink-2" data-testid="source-log-note">
          {scope.data.log_note}
        </p>
      ) : null}
      <label htmlFor={`confirm-source-${source.id}`} className="meta-label">
        type “{source.name}” to confirm
      </label>
      <input
        id={`confirm-source-${source.id}`}
        className="w-56 border border-deny bg-ground p-2 font-mono text-sm text-ink"
        value={typed}
        onChange={(event) => setTyped(event.target.value)}
        autoComplete="off"
      />
      <span className="flex items-baseline gap-3">
        <button
          type="button"
          className="meta text-deny hover:underline disabled:text-ink-3 disabled:no-underline"
          disabled={typed !== source.name || remove.isPending || !scope.data}
          onClick={() => remove.mutate()}
        >
          {remove.isPending ? "deleting…" : "delete this source"}
        </button>
        <button type="button" className="meta text-ink-3" onClick={onCancel}>
          cancel
        </button>
      </span>
      {remove.isError ? <Failure error={remove.error} /> : null}
    </div>
  );
}

/**
 * Register a directory.
 *
 * A path typed into a box, resolved server-side. There is no directory picker
 * because the path being entered is a path on the *API's* machine, not on the
 * browser's, and a picker would show the wrong filesystem convincingly.
 */
function Register() {
  const client = useQueryClient();
  const [name, setName] = useState("");
  const [root, setRoot] = useState("");

  const create = useMutation({
    mutationFn: () =>
      // `kind` and `follow_symlinks` are sent explicitly rather than left to the
      // API's defaults. They have defaults there, but the generated request type
      // does not mark them optional, and a client that guessed which fields the
      // server would fill in is a client that breaks when it stops.
      api.createSource({
        kind: "filesystem",
        name: name.trim(),
        root: root.trim(),
        follow_symlinks: false,
      }),
    onSuccess: () => {
      setName("");
      setRoot("");
      void client.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  return (
    <section className="flex flex-col gap-2">
      <SectionHeading>register a directory</SectionHeading>
      <form
        className="flex flex-col gap-3 sm:flex-row sm:items-end"
        onSubmit={(event) => {
          event.preventDefault();
          if (name.trim() && root.trim()) create.mutate();
        }}
      >
        <label className="flex flex-1 flex-col gap-1">
          <span className="meta-label">name</span>
          <input
            className="field"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="notes"
            spellCheck={false}
            autoComplete="off"
          />
        </label>
        <label className="flex flex-[2] flex-col gap-1">
          <span className="meta-label">absolute path, on the api's machine</span>
          <input
            className="field"
            value={root}
            onChange={(event) => setRoot(event.target.value)}
            placeholder="/Users/you/notes"
            spellCheck={false}
            autoComplete="off"
          />
        </label>
        <button
          type="submit"
          className="btn"
          disabled={create.isPending || !name.trim() || !root.trim()}
        >
          {create.isPending ? "registering…" : "register"}
        </button>
      </form>
      {create.isError ? <Failure error={create.error} /> : null}
      <p className="meta text-ink-3">
        Registering does not read anything. It records where to look; the first
        sync is a separate act, and it is the one that costs time.
      </p>
    </section>
  );
}
