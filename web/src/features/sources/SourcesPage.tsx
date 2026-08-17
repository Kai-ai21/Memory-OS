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
              {["name", "kind", "memories", "chunks", "last sync", "last full sync", ""].map(
                (heading) => (
                  <th key={heading} className="meta-label py-1 pr-4 font-normal">
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {sources.map((source) => (
              <Row key={source.id} source={source} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="meta text-faint">
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
      <td className="meta py-1 pr-4 text-faint">{source.kind}</td>
      <td className="meta py-1 pr-4">{count(source.memories)}</td>
      <td className="meta py-1 pr-4">{count(source.chunks)}</td>
      <td className="meta py-1 pr-4 text-faint">{timestamp(source.last_sync_at)}</td>
      <td className="meta py-1 pr-4 text-faint">{timestamp(source.last_full_sync_at)}</td>
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
          <span className="meta text-faint" title="Messages are pushed, not walked">
            not walked
          </span>
        )}
        {sync.isError ? <Failure error={sync.error} /> : null}
        {sync.isSuccess ? (
          <span className="meta ml-2 text-muted">
            {sync.data?.job_id ? "queued" : "already running"}
          </span>
        ) : null}
      </td>
    </tr>
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
      <p className="meta text-faint">
        Registering does not read anything. It records where to look; the first
        sync is a separate act, and it is the one that costs time.
      </p>
    </section>
  );
}
