# Deploying MEMO

M12.3. One server, Docker Compose, a real hostname with HTTPS. Everything here
was rehearsed end to end on a laptop with the production files unchanged, using
`MEMO_DOMAIN=localhost` (see [What was verified](#what-was-verified)). It has
**not** yet been run on a public server.

## Where it runs, and what that gives up

**Recommendation: a single VPS running `docker-compose.prod.yml`.** About 4 GB of
RAM is the floor and 8 GB is comfortable. Measured resident memory on the
rehearsal: API ~460 MB and worker ~410 MB (each loads the embedding and
reranking models), Neo4j ~470 MB, Postgres ~40 MB plus its cache, Caddy ~30 MB.

| Option | What it costs you |
| --- | --- |
| **Single VPS + Compose** (chosen) | You run it: OS updates, disk space, watching that backups land. One host is one point of failure, with no failover. Neo4j is just another container with a volume, which is the main reason to choose this. |
| Fly.io / Railway | Postgres with pgvector is available. Neo4j Community needs a persistent volume of its own, or a paid Aura instance. API, worker and Neo4j become three billed machines that each need memory for the models, so it costs more than a VPS for the same thing. |
| Split: Vercel UI + API elsewhere | Two sites. The session cookie is `SameSite=Lax` and the live-events stream is an `EventSource` without credentials, and both stop working cross-site. Making them work means `SameSite=None` and CORS with credentials: more to get wrong, for no gain at one user. |

**The models are the real constraint, and they turned out not to be a problem at
this size.** They are baked into the image (217 MB in `/app/models`) and the
running container has `HF_HUB_OFFLINE=1`, so nothing is downloaded at request
time. On the rehearsal machine (Apple silicon, 10 cores), the API container went
from restart to healthy in 6.4 s, and the first search after it took 0.34–0.50 s.
A warm search took 0.04–0.09 s. A 2-vCPU VPS will be slower at embedding and
reranking. That has not been measured, and it is the first number to take after a
real deploy (`deploy/smoke.py` prints it). If it is too slow, the options are a
bigger instance, a hosted embedding API (which means re-embedding the corpus,
because vectors from two models do not mix), or accepting the latency.

**Cost:** the VPS, roughly €6–12 a month for a 4–8 GB Hetzner cloud server after
its June 2026 price change (check hetzner.com/cloud; third-party figures
disagree). Add a domain, about €1 a month, and optionally off-host backup storage.
Groq's free tier costs nothing and limits how much you can ask.

## Deploy from scratch

On a fresh Ubuntu or Debian server with Docker Engine and the compose plugin:

1. **DNS first.** Point an A (and AAAA) record for your hostname at the server,
   and open ports 80 and 443. Caddy needs both to obtain a certificate on first
   start.
2. **Code and secrets:**
   ```bash
   git clone https://github.com/Kai-ai21/Memory-OS.git memo && cd memo
   cp deploy/env.production.example .env.production
   chmod 600 .env.production
   ```
   Fill it in (variables below), generating every secret. **Copy
   `MEMOS_MASTER_KEY` somewhere that is neither this server nor its backups.**
3. **Start:**
   ```bash
   docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
   ```
   Order is enforced by the compose file: Postgres and Neo4j become healthy,
   `migrate` runs `alembic upgrade head` and exits, and only then do `api` and
   `worker` start. `web` starts once the API is healthy.
4. **Create the account**, then restart the worker. It works for the account
   that exists when it starts, and before this there was none:
   ```bash
   docker compose --env-file .env.production -f docker-compose.prod.yml exec api memoryos auth create-user --email you@example.com
   ```
   ```bash
   docker compose --env-file .env.production -f docker-compose.prod.yml restart worker
   ```
5. **Verify from another machine** (any machine with this repository and `uv`):
   ```bash
   MEMO_SMOKE_PASSWORD='…' uv run python deploy/smoke.py --url https://memo.example.com --email you@example.com
   ```

**Updating** is `git pull`, then the same `up -d --build`. The new image's
migrations run before the new API starts. If a migration fails, `migrate` exits
non-zero and Compose stops there, so the new `api` and `worker` are never created
and the running ones should be left alone. That failure path has not been
exercised. Each update is a short restart of the API, not a rolling deploy.

### Environment variables

Set in `.env.production`. The compose file builds the application's settings
from these, and every one marked required stops `docker compose` if it is missing.

| Variable | Required | What it is |
| --- | --- | --- |
| `MEMO_DOMAIN` | yes | The public hostname. Caddy gets its certificate. |
| `MEMOS_MASTER_KEY` | yes | 32 random bytes, base64. Every memory is encrypted under it, so losing it loses the data. |
| `POSTGRES_PASSWORD` | yes | The database superuser: migrations and backups. |
| `MEMOS_APP_DB_PASSWORD` | yes | The application's own role, which is not a superuser, so row-level security applies to it. Used once, by the migration that creates the role. |
| `MEMOS_NEO4J_PASSWORD` | yes | Neo4j's. At least 8 characters. |
| `MEMOS_ALLOW_REGISTRATION` | no, default `false` | Leave it off, or anyone who finds the URL can make an account. |
| `MEMOS_GROQ_API_KEY`, `MEMOS_GROQ_MODEL`, `MEMOS_LLM_PROVIDER`, `MEMOS_GEMINI_API_KEY` | no | Answers need a language model; search does not. |
| `MEMO_BACKUP_DIR`, `BACKUP_INTERVAL_SECONDS`, `BACKUP_KEEP` | no | Default `./backups`, daily, 14 kept. |

**The application refuses to start in production** (`MEMOS_ENV=production`, which
the compose file sets) if:
- the master key is missing;
- either database URL is a development default or points at localhost;
- the Neo4j password is the default;
- `MEMOS_CORS_ORIGINS` is empty or not https;
- SQL echo is on.

It lists every problem at once. In production it also logs JSON, sends `Secure`
cookies, and serves no `/docs` or `/openapi.json`.

There is no session secret to set: sessions are random tokens stored hashed, and
nothing signs them.

## Backups, and getting them back

The `backup` service writes a `pg_dump` and a tarball of the blob store into
`MEMO_BACKUP_DIR` once a day. It keeps 14 of each and writes under a `.partial`
name until a file is complete. It needs both: the database holds the log, accounts,
decisions and the wrapped data keys, and the blobs hold the bytes every replay
rebuilds from. To take one now:

```bash
docker compose --env-file .env.production -f docker-compose.prod.yml exec backup sh /deploy/backup.sh --once
```

**Copy them off the server.** Backups on the same disk survive a mistake, but not
the loss of the server. They are useless without `MEMOS_MASTER_KEY`, which is
deliberately not in them.

To restore (it asks you to type `restore`):

```bash
deploy/restore.sh backups/memos-<stamp>.dump backups/blobs-<stamp>.tar.gz
```

It stops everything that writes, recreates the database from the dump, extracts
the blobs (content-addressed, so nothing has to be deleted first), and starts the
stack, migrations first. Run it with the same `.env.production` the backup was
taken under.

## What this deployment protects, and what it does not

**Protected:**
- **Every route but `/health/*` and `/auth/*` needs a session.** A test walks every
  path and method the API serves and requires a 401 from each one.
- **Accounts are isolated by Postgres row-level security** (M11.1). The application
  connects as a role that is not a superuser, so it cannot skip the policies. More
  than one account is supported. Registration is off unless enabled.
- **Passwords are Argon2id. Session tokens are random and stored only as hashes.**
  Logout and password reset revoke them.
- **Cookies are `Secure`, `HttpOnly` and `SameSite=Lax`, over HTTPS only, with
  HSTS.**
- **Login is rate limited** to five failures per fifteen minutes per client. The
  API trusts `X-Forwarded-For` from Caddy's address only. Without that, every
  request would arrive from the proxy and five bad logins from anyone would lock
  out everyone. The rehearsal confirmed the API sees the real client and the sixth
  bad login gets a 429.
- **Memory and chunk text is encrypted at rest** (M11.2, as wired by M12.1). It
  is encrypted on write, so a stolen database or backup does not contain your
  sentences.
- **Only ports 80 and 443 are open.** Postgres, Neo4j and the API are on a private
  network.

**Not protected:**
- **The blob store is plaintext on disk.** These are the original bytes of
  everything ingested, in the `blobs` volume and in every backup tarball. Anyone
  with the server's disk or a blob backup has the corpus.
- **Chat messages are stored unencrypted**, as are embeddings, the keyword index's
  vocabulary, entity names, file paths and titles.
- **The master key is in the server's environment.** Anyone who can run
  `docker inspect` on the host can read it. Encryption at rest protects a stolen
  disk or backup, not a compromised server.
- **Your questions and the retrieved passages go to the language model provider**
  (Groq by default) whenever you ask something.
- **There is one server.** Nothing fails over. Backups are only as current as the
  last night, and only as safe as the last copy off the server.
- **There is no second factor and no email password reset.** A reset needs a shell
  on the server (`memoryos auth reset-password`).
- **The rate limiter lives in one process.** Run more than one API container and
  each has its own count.

## What was verified

Rehearsed on a laptop with the production compose file, the production images and
generated secrets, `MEMO_DOMAIN=localhost` and Caddy's internal CA, ports 8080 and
8443. `deploy/smoke.py` checked it through Caddy, with TLS verified against that
CA:

| Check | Result |
| --- | --- |
| Page over HTTPS, HSTS sent | pass |
| Anonymous `GET /api/memories` | 401 |
| `/api/openapi.json` | 404 |
| Login; cookie `Secure` and `HttpOnly` | pass |
| Ingest a statement; worker makes it searchable | pass, 2.0 s |
| Search: first request after an API restart / warm | 0.34–0.50 s / 0.04–0.09 s |
| Ask a question | answered from the passage |
| Ask "what is our AWS billing setup" | refused: "The passages do not contain any information about the AWS billing setup." |
| Logout; the old cookie afterwards | 204; 401 |
| Six bad logins through the proxy | 401 ×5, then 429 |
| Stack from empty volumes to API healthy | 13 s (images built) |

**The restore was tested by destroying it.** I took a backup, ran
`docker compose down -v` on the rehearsal (every volume gone), brought it up empty
(0 accounts, 0 memories), and ran `deploy/restore.sh`. Afterwards the counts matched
the backup exactly (1 account, 2 memories, 2 chunks, 2 data keys, 2 chat messages,
2 blobs). The restored account signed in with its original password, and search
returned both pre-disaster memories, decrypted, beside a new one.

Not verified: a public hostname with a Let's Encrypt certificate, a machine other
than the one running the stack, and timings on a small VPS.
