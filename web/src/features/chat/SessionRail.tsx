/**
 * The conversations, newest first.
 *
 * **A rail rather than a dropdown**, because the list is the navigation: the
 * question it answers is "what was I talking about" and that is a question you
 * scan, not one you open a menu to. It sits inside the chat page rather than in
 * the shell's sidebar, which already lists the fourteen *views* — a conversation
 * is not a view, and mixing the two would make the sidebar a list of two
 * different kinds of thing.
 *
 * Archiving is here and deleting is not, anywhere. A conversation you are done
 * with is not a conversation that did not happen, and every message in an archived
 * session is still a memory, still searchable, still connected to everything it
 * shares an entity with. The rail says so rather than leaving somebody to wonder
 * what the button destroyed.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, type ChatSession } from "../../api/client";
import { Button, Failure } from "../../components/primitives";
import { RelativeTime } from "../../components/RelativeTime";
import { count } from "../../lib/format";
import { useToast } from "../../lib/toast";

export function SessionRail({
  current,
  onSelect,
  onNew,
}: {
  current: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const [showArchived, setShowArchived] = useArchivedToggle();
  const sessions = useQuery({
    queryKey: ["chat-sessions", showArchived],
    queryFn: () => api.chatSessions(showArchived),
  });

  return (
    // `min-w-0` on the column and `truncate` on the title are a pair, and only
    // the pair works: a grid track sized `16rem` still lets its content set a
    // larger min-content width, so without this the longest session title
    // overflows the rail and lands on top of the conversation beside it.
    <aside className="flex min-w-0 flex-col gap-2" aria-label="Conversations">
      <div className="flex items-baseline justify-between border-b border-rule-strong pb-1">
        <h2 className="meta-label text-ink-2">conversations</h2>
        <button type="button" className="btn" onClick={onNew}>
          new
        </button>
      </div>

      {sessions.isError ? <Failure error={sessions.error} /> : null}
      {sessions.data?.length === 0 ? (
        <p className="meta text-ink-3">
          Nothing yet. A conversation starts on the first thing you type, and a new
          one starts by itself after thirty minutes of silence.
        </p>
      ) : null}

      <ul className="flex flex-col" data-testid="sessions">
        {(sessions.data ?? []).map((session) => (
          <Row
            key={session.id}
            session={session}
            active={session.id === current}
            onSelect={onSelect}
          />
        ))}
      </ul>

      <button
        type="button"
        className="meta self-start text-ink-3 hover:text-accent"
        onClick={() => setShowArchived(!showArchived)}
      >
        {showArchived ? "hide archived" : "show archived"}
      </button>
    </aside>
  );
}

function Row({
  session,
  active,
  onSelect,
}: {
  session: ChatSession;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  const client = useQueryClient();
  const toast = useToast();
  const archiving = session.archived_at === null;
  const archive = useMutation({
    /* The mutation takes the state to move *to*, so undo is the same call with
       the flag inverted. That is the whole reason archiving is undoable and
       deletion is not: one is a boolean on a row that already exists, and the
       other has nothing to call. */
    mutationFn: (archived: boolean) => api.archiveSession(session.id, archived),
    onSuccess: (_data, archived) => {
      void client.invalidateQueries({ queryKey: ["chat-sessions"] });
      // The row leaves the list, which is a change you notice only if you were
      // watching that corner of the screen.
      toast.show(archived ? "Conversation archived" : "Conversation restored", {
        undo: () => archive.mutate(!archived),
      });
    },
  });

  return (
    <li
      className={`flex min-w-0 items-baseline gap-2 border-b border-rule/60 py-1.5 ${
        active ? "bg-surface-tint" : ""
      }`}
      data-testid="session-row"
    >
      <button
        type="button"
        className="flex min-w-0 flex-1 flex-col items-start gap-0.5 text-left"
        aria-current={active ? "true" : undefined}
        onClick={() => onSelect(session.id)}
      >
        <span
          className={`w-full truncate text-sm ${active ? "text-ink" : "text-ink-2"}`}
          title={session.title ?? undefined}
        >
          {/* Null rather than "Conversation 4". A name that says nothing cannot be
              searched for, and the honest version of "we could not derive one" is
              saying so. */}
          {session.title ?? <span className="text-ink-3 italic">untitled</span>}
        </span>
        <span className="meta text-ink-3">
          <RelativeTime value={session.last_activity} /> · {count(session.message_count)}{" "}
          {session.message_count === 1 ? "message" : "messages"}
          {session.archived_at ? " · archived" : ""}
        </span>
      </button>
      <Button
        className="meta inline-flex shrink-0 items-center gap-1.5 text-ink-3 hover:text-accent"
        title={
          session.archived_at
            ? "Bring this conversation back into the list"
            : "Hide this conversation. Every message stays a memory and stays searchable."
        }
        onClick={() => archive.mutate(archiving)}
        loading={archive.isPending}
      >
        {session.archived_at ? "restore" : "archive"}
      </Button>
    </li>
  );
}

/**
 * Whether archived conversations are shown.
 *
 * Local state rather than a URL parameter, unlike search: this is a preference
 * about the rail rather than a description of what is on screen, and nobody
 * bookmarks "the session list with the archived ones showing".
 */
function useArchivedToggle(): [boolean, (value: boolean) => void] {
  const client = useQueryClient();
  const state = useQuery({
    queryKey: ["chat-sessions-archived-toggle"],
    queryFn: () => false,
    initialData: false,
    staleTime: Infinity,
  });
  return [
    state.data,
    (value) => client.setQueryData(["chat-sessions-archived-toggle"], value),
  ];
}
