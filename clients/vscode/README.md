# Memory OS — VS Code extension

Context for the file you are looking at, drawn from your own corpus.

Two features and no third one: it posts a `file_focused` event when you change
tabs, and it shows a sidebar panel with the context M6.1 assembled for that file.
No commands, no status bar item, and **no notifications** — nothing here is
allowed to interrupt you. Anything proactive is M6.3.

## There is no authentication, and that is not an oversight

This talks to `http://localhost:8000` with no token, no header and no TLS. Any
process on your machine can read the same API, and so can any page you visit if
you have started the API with a permissive `MEMOS_CORS_ORIGINS`.

That is acceptable for exactly one deployment — the API running on your own
machine, bound to localhost, holding your own corpus — and it is the only one
this extension supports. It is stated here rather than papered over with a token
field that would be theatre: a shared secret in a settings file, read by an
extension, sent over plain HTTP to a process on the same machine, protects
against nothing that localhost does not already protect against.

**Do not point `memoryos.apiUrl` at a remote host.** If the API is ever hosted,
the extension needs real auth first, and this README is where that will be
written down.

## Installing it in development mode

Not published to the marketplace, and not intended to be.

```bash
cd clients/vscode
npm install
npm run compile
```

Then either:

- **Run it from the editor.** Open `clients/vscode` in VS Code and press `F5`.
  That opens an Extension Development Host window with the extension loaded.
  Open your own repository in that window.
- **Install it as a folder.** Symlink or copy this directory into
  `~/.vscode/extensions/memoryos` and restart the editor. `npm run compile` has
  to have been run, because `main` points at `out/extension.js`.

`npm run watch` recompiles on save while you are working on the extension
itself.

## What it needs running

| | |
| --- | --- |
| the API | `uv run uvicorn "memoryos.api.app:create_app" --factory` |
| a worker | `uv run memoryos worker` — this is what actually assembles context |
| the web UI, optionally | `make web`, for the links to open something |

**The worker is the part people forget.** `GET /context` never assembles: it
serves the cache and enqueues a job on a miss, because assembly loads an embedder
and a cross-encoder and takes a second warm and twenty cold. With no worker
running, the panel says "Assembling context…" indefinitely and is telling the
truth — the job is queued and nothing is draining it.

With none of it running, the panel says the API is unavailable and tries again
when you switch files. That is the intended behaviour rather than a failure
state: the API being stopped is the normal condition of a laptop.

## Settings

| setting | default | |
| --- | --- | --- |
| `memoryos.apiUrl` | `http://localhost:8000` | Localhost only. See above. |
| `memoryos.webUrl` | `http://localhost:5173` | Where clicking an item opens it. |
| `memoryos.emitEvents` | `true` | Post `file_focused` on tab change. |
| `memoryos.tokenBudget` | `4000` | Tokens of context to ask for. |

## Tests

```bash
npm test
```

Plain `node --test`, no editor host. `client.ts` and `panel.ts` deliberately do
not import `vscode`, which is what makes the failure behaviour testable at all —
"does it throw when the API is down" is otherwise answered by watching for a
notification that should never appear.
