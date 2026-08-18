/**
 * Files on a message, and the honest account of where each one is.
 *
 * **Progress is per file and per stage, because a PDF is not searchable the
 * moment its upload finishes.** Bytes land, then a worker reads them, then chunks
 * them, then embeds them, and on a real document that is tens of seconds. An
 * interface that said "done" at upload would look broken thirty seconds later
 * when a search found nothing — so each row says which of the four stages it is
 * in, polled until it settles.
 *
 * **A failure is drawn at full weight, in the parser's own words.** A scanned PDF
 * is a document somebody dropped in good faith; the pipeline is right to refuse it
 * and the only outcome worse than refusing is refusing quietly, because then they
 * believe it was filed. The sentence shown is the one the parser wrote — it names
 * the file, counts the characters it found, counts the pages it looked at, and says
 * what the file probably is. Nothing here paraphrases that into "processing
 * failed".
 */

import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api, type Attachment, type MessageStatus } from "../../api/client";
import { fileSize } from "../../lib/format";

/** How often a file that has not settled is asked about. */
const POLL_MS = 2_000;

export function AttachmentList({
  attachments = [],
}: {
  attachments?: Attachment[];
}) {
  // Defaulted rather than assumed. The API always sends the field, so an empty
  // list is the normal case and `undefined` means somebody is rendering a
  // hand-built message — and a missing optional field must not take the whole
  // conversation down with it.
  if (attachments.length === 0) return null;
  return (
    <ul className="flex flex-col gap-1" data-testid="attachments">
      {attachments.map((attachment) => (
        <Row key={attachment.id} attachment={attachment} />
      ))}
    </ul>
  );
}

function Row({ attachment }: { attachment: Attachment }) {
  const status = useQuery({
    queryKey: ["message-status", attachment.memory_id],
    queryFn: () => api.messageStatus(attachment.memory_id!),
    enabled: attachment.memory_id !== null,
    // Stops on a terminal stage, in both directions. `failed` is terminal too —
    // polling a dead-lettered job forever is a request per two seconds for an
    // answer that will never change.
    refetchInterval: (query) => (settled(query.state.data) ? false : POLL_MS),
    refetchIntervalInBackground: true,
  });

  const data = status.data;
  const failed = data?.stage === "failed";

  return (
    <li
      className={`flex flex-col gap-0.5 border-l-2 pl-3 ${
        failed ? "border-deny" : "border-rule-strong"
      }`}
      data-testid="attachment"
    >
      <div className="flex flex-wrap items-baseline gap-x-3">
        {attachment.memory_id ? (
          // Links out to the memory, which is where the file sits beside
          // everything it connects to regardless of which conversation it arrived
          // in. A session is a view; the memory is the thing.
          <Link
            to={`/memory/${attachment.memory_id}`}
            className="font-mono text-sm text-ink hover:text-accent hover:underline"
          >
            {attachment.filename}
          </Link>
        ) : (
          <span className="font-mono text-sm text-ink">{attachment.filename}</span>
        )}
        <span className="meta text-ink-3">{fileSize(attachment.byte_size)}</span>
        {attachment.media_type ? (
          <span className="meta text-ink-3">{attachment.media_type}</span>
        ) : null}
        {attachment.deduplicated ? (
          // Said out loud. A silent success looks identical to a re-upload that
          // did nothing, and content addressing means the second upload of a file
          // genuinely stores no new bytes — which is worth knowing rather than
          // hiding.
          <span className="meta text-ink-2">already in memory, linked</span>
        ) : null}
      </div>
      <p className={`meta ${failed ? "text-deny" : "text-ink-3"}`}>
        {describe(data)}
      </p>
    </li>
  );
}

function settled(status: MessageStatus | undefined): boolean {
  return status?.stage === "indexed" || status?.stage === "failed";
}

function describe(status: MessageStatus | undefined): string {
  if (!status) return "uploaded · checking…";
  switch (status.stage) {
    case "failed":
      // Verbatim. See the module note.
      return status.failure ?? "this could not be read, and no reason was recorded";
    case "stored":
      return "uploaded · waiting for a worker";
    case "parsing":
      return "uploaded · reading it…";
    case "chunking":
      return "parsed · chunking and embedding…";
    case "indexed":
      return "indexed · searchable";
  }
}
