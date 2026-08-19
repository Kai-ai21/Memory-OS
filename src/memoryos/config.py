from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memoryos.adapters.embedding.sentence_transformers import (
    DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL,
)
from memoryos.adapters.llm.gemini import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from memoryos.adapters.llm.groq import DEFAULT_MODEL as DEFAULT_GROQ_MODEL
from memoryos.adapters.reranking.cross_encoder import (
    DEFAULT_MODEL as DEFAULT_RERANKER_MODEL,
)


# Where `./var/blobs` and `./var/hf` are relative *to*.
#
# **Not the working directory, and M10.3 is why.** Starting the API from `web/`
# while the worker ran from the repo gave the two processes different blob roots:
# the API wrote artifacts into `web/var/blobs`, the worker looked for them in
# `var/blobs`, and a normalization job dead-lettered on a blob that had in fact
# been stored — a hundred metres from where anything went looking. Nothing errored
# at the point of the mistake. M1.7 hit the same edge from the other side, where
# running `replay` from a subdirectory resolved this default to an empty path and
# truncated the corpus before failing on the first document.
#
# So a relative path is anchored to the tree rather than to the shell. Walking up
# from this module rather than from `cwd`, because that is the one location that
# does not depend on how the process was started — which is the entire failure
# being fixed. An installed-not-editable copy has no `pyproject.toml` above it, so
# the walk falls back to `cwd` and the old behaviour, and an absolute
# `MEMOS_BLOB_ROOT` is untouched either way: naming a path outright is the one
# case where nobody has to guess.
def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _project_root()


class Settings(BaseSettings):
    # `env_file` is anchored for the same reason the paths below are, and the two
    # halves of the M10.3 failure were both here: run from `web/`, this file was
    # not found at all, so `MEMOS_BLOB_ROOT` went unread *and* the default it fell
    # back to resolved somewhere else.
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="MEMOS_", extra="ignore"
    )

    environment: str = "local"

    # **The application connects as `memos_app`, which is not a superuser, and
    # that is not cosmetic.** Row-level security is skipped entirely for
    # superusers and for any role with BYPASSRLS — `FORCE ROW LEVEL SECURITY`
    # does not change that. Until M11.1 this pointed at `memos`, the owning
    # superuser, which means every policy would have been enabled, visible in
    # `pg_policies`, and enforcing nothing. The role is created by migration
    # 0032 and holds DML rights and nothing else.
    database_url: str = "postgresql+asyncpg://memos_app:memos_app@localhost:5433/memos"

    # The owner, used by Alembic and by nothing else. DDL needs privileges the
    # application must not have, and a migration that ran under the policies it
    # is in the middle of creating would be unable to see the rows it is
    # backfilling.
    database_admin_url: str = "postgresql+asyncpg://memos:memos@localhost:5433/memos"

    # The password migration 0032 gives `memos_app` when it creates it. A fixed
    # default because a local Postgres bound to localhost with a `memos:memos`
    # superuser beside it is not made safer by a random one; set it for anything
    # that is not a laptop.
    app_db_password: str = "memos_app"

    # **Registration is off unless somebody turns it on.** M11.1 makes a second
    # account safe, which is not the same as wanting one: the deployment this is
    # written for is one person on one laptop, and an open registration endpoint
    # on it is a way for anybody who can reach the port to help themselves to an
    # account. Off by default, and the accounts that do exist are made with
    # `memoryos auth create-user` from a shell.
    allow_registration: bool = False
    # Where the test suite writes. Its own database rather than its own
    # isolation strategy: the integration tests truncate every table, which is
    # the only strategy that survives code under test committing, and pointing
    # that at the development database destroys a working corpus. It did, three
    # times during M2.0a.
    #
    # Compose creates it; `MEMOS_ENVIRONMENT=test` selects it. CI sets neither
    # and is unaffected — its database is disposable, and a second one there
    # would only be a second thing to migrate.
    # `memos_app`, like `database_url`, and for the same reason: a suite that
    # ran as the superuser would pass every isolation test by not having any
    # policies applied to it.
    test_database_url: str = "postgresql+asyncpg://memos_app:memos_app@localhost:5433/memos_test"
    db_echo: bool = False
    # Where artifact bytes live. Local directory for now; the BlobStore port is
    # what lets this become object storage without a use case changing.
    blob_root: Path = Path("./var/blobs")
    # Where HuggingFace caches model weights. Pinned explicitly so that a local
    # run and a CI run agree on the location, which is what makes the CI cache
    # key mean anything.
    hf_home: Path = Path("./var/hf")
    # Sourced from the adapter rather than repeated here. Two copies of a
    # model name is how the CLI ended up on a different model from the tests.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = 32
    # HNSW search width per query. Higher recall, higher latency; measured
    # rather than guessed by `memoryos eval-recall`.
    hnsw_ef_search: int = 100
    # How much each ranking counts in the fusion.
    #
    # **Both signals default to zero, and that is the M2.3b result rather than a
    # placeholder.** A 97-combination grid search over the 41-query golden set
    # found that recency monotonically *lowers* nDCG at every importance level
    # it was tried with — 0.735 at weight 0, 0.731 at 0.15, 0.721 at 0.30, 0.707
    # at 0.60 — and that the best importance weight, 0.10, gains 0.0109, which
    # is below the 0.0122 resolution floor M2.3a measured. A gain under the
    # floor is not evidence, so shipping it would be shipping noise with a
    # decimal point.
    #
    # The machinery stays and the weights stay tunable: `MEMOS_WEIGHT_RECENCY`
    # and `MEMOS_WEIGHT_IMPORTANCE` turn them on for a corpus where the answer
    # may differ. This one is a repository of explanatory prose, where when a
    # file was last edited says almost nothing about whether it answers a
    # question about the design.
    weight_vector: float = 1.0
    weight_keyword: float = 1.0
    weight_recency: float = 0.0
    weight_importance: float = 0.0
    # M3.5's graph expansion: the one ranking that *introduces* candidates rather
    # than reordering them.
    #
    # **Zero, and that is the measurement rather than a placeholder.** The
    # milestone specified 0.5; two results overrode it, and both are in the README.
    #
    # At 0.5 the ranking is arithmetically *inert*. RRF's curve is flat by design,
    # so a graph-only candidate at rank 1 contributes 0.5/61 = 0.0082 while a
    # vector-only candidate at rank 30 contributes 1/90 = 0.0111 — and the vector
    # leg returns fifty. Measured over 46 golden queries: expansion produced
    # candidates for 18 of them and *not one* reached the top ten. It cost 30-140ms
    # per query to change nothing.
    #
    # At 1.0 it places candidates and they are worse than what they displace:
    # recall@10 falls 0.029 and nDCG@10 falls 0.019, both larger than the 0.0122
    # resolution floor, so that is real harm rather than noise. Per query, 1 of the
    # 5 written for the graph improved (+0.064 nDCG), 1 fell sharply (-0.243), and
    # 3 saw no contribution at all.
    #
    # The machinery stays and the knob stays, exactly as M2.3b left recency and
    # importance: this corpus is one person's prose about one system, where
    # structural relatedness and semantic relatedness are nearly the same relation.
    # A corpus of meetings, commits and invoices sharing a person might answer
    # differently — and extraction here has reached only 13% of the corpus, which
    # bounds what the number can mean. See the README.
    weight_graph: float = 0.0
    # Entity hops the expansion traverses. Two, because depth 3 on a graph this
    # connected reaches most of the corpus and a ranking that contains everything
    # is not a ranking.
    graph_depth: int = 2
    # The share of reachable memories an entity may appear in before it is treated
    # as carrying no information. An entity in a tenth of the corpus is a bridge
    # every path can cross, and at depth 2 a bridge connects everything to
    # everything — which is what hub suppression exists to prevent.
    graph_hub_ratio: float = 0.10
    # How many of hybrid retrieval's memories seed the expansion. A precision
    # bound rather than a cost one: expanding from a memory retrieval ranked
    # fortieth expands from something retrieval was not sure about.
    graph_seed_memories: int = 10
    # The cross-encoder that rescores the shortlist. Sourced from the adapter
    # for the same reason the embedder is: two copies of a model name is how the
    # CLI ended up running a different model from the tests.
    reranker_model: str = DEFAULT_RERANKER_MODEL
    # How many fused chunks reach the expensive model. The whole retrieve-then-
    # rerank trade lives in this number: every candidate costs a forward pass,
    # and the shortlist is the only thing standing between a 30ms search and a
    # half-second one.
    #
    # **25 rather than the obvious 50, and it is faster *and* more accurate.**
    # Measured on the 41-query golden set: nDCG 0.761 at 50, 0.788 at 25, 0.781
    # at 15, against 0.718 with reranking off — while p95 latency falls from
    # 473ms to 280ms. A deeper shortlist lets the cross-encoder promote a chunk
    # fusion ranked fortieth into the top ten, and at that depth the model is
    # not reliable enough to be trusted over fusion's own judgement. Bounding
    # the shortlist bounds how far a candidate can jump.
    rerank_candidates: int = 25
    rerank_enabled: bool = True
    # M4.3. Whether a query is inspected for temporal intent at all.
    #
    # An off switch rather than only a weight, and it earns its place twice. It
    # is the control arm of this milestone's A/B — the same code and corpus with
    # parsing off is the only honest baseline, since the committed
    # `var/baseline-hybrid.json` predates M2.4's reranking — and it is the escape
    # hatch if the parser ever fires on a query it should not: retrieval falls
    # straight back to M3.5 rather than needing a deployment.
    temporal_intent_enabled: bool = True
    # How much recency counts *for a query that asked for it*.
    #
    # Distinct from `weight_recency`, which is the global signal M2.3b measured
    # and switched off. That measurement said recency does not help a conceptual
    # question about a codebase, and it is not evidence about a question that
    # says "recently" — those are different hypotheses and this is the second
    # one. 0.5 is a starting point, not a tuned value: `tune-weights` searches
    # the global grid and has no way to vary a weight that only exists for the
    # subset of queries a parser fires on.
    temporal_recency_weight: float = 0.5
    # Which provider answers. Both implement the same `LanguageModel` port, so
    # this is the only thing that changes between them — which is what M2.6
    # claimed a port would buy and what M2.6a spent to check.
    #
    # Groq by default because Gemini's free tier began returning `limit: 0` on
    # this account, which blocks answering entirely. Gemini stays selectable
    # rather than being removed: the quota is an account condition, not a defect
    # in the adapter.
    llm_provider: Literal["groq", "gemini"] = "groq"
    # Answering only. Absent, retrieval and search work exactly as before and
    # `ask` reports that it needs a key rather than failing obscurely. Only the
    # *selected* provider's key is required; the other may be empty.
    gemini_api_key: str | None = None
    llm_model: str = DEFAULT_LLM_MODEL
    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_MODEL
    # Tokens of *passages*. The system prompt and question add a few hundred
    # more. Deliberately well inside the model's window: a fuller prompt is not
    # a better one, because the instruction to refuse competes with every extra
    # passage that looks vaguely relevant.
    answer_token_budget: int = 6000
    # M6.0. How many events one source may deliver per window before it is
    # refused. Per source rather than global: the point is to stop one
    # misbehaving plugin, and a global limit would let that plugin lock out
    # every well-behaved client.
    #
    # Sixty a minute is generous for anything a person does and far below what a
    # plugin firing on keystrokes produces, which is the gap the number needs to
    # sit in. Deduplication is the first defence and catches the well-behaved
    # case; this catches the client that sets no dedupe key, or a fresh one each
    # time.
    # Where the local clients post to and read from. Not a deployment setting —
    # M6.2's watcher and editor extension are explicitly localhost-only and
    # unauthenticated — and a setting anyway, because a person running the API
    # on a second port should not have to edit source to point the watcher at
    # it.
    api_url: str = "http://localhost:8000"
    event_rate_limit: int = 60
    event_rate_window_seconds: int = 60
    answer_max_tokens: int = 1024
    # M7.1. Retrievals one question may make before the loop is stopped.
    #
    # Six is a bound on cost, not a target: it is roughly the number of dependent
    # steps the hardest question in this milestone's set actually decomposes into
    # ("find poor outcomes, read their assumptions, group them, check
    # independence, gather evidence"), and it is also six model calls plus tool
    # work, which on a free tier is minutes and a measurable share of a daily
    # cap. Raising it makes drift cheaper before it makes answers better.
    agent_max_hops: int = 6
    # Tokens the compacted findings from earlier hops may occupy. Two verbatim
    # search results already cost ~1,400, so this roughly doubles the history the
    # prompt carries rather than replacing it.
    agent_finding_budget: int = 1200
    # M7.2. The share of an answer's factual claims that must trace to something
    # the trajectory retrieved before the answer is shown at all.
    #
    # A setting rather than a constant because it is the one number in this
    # milestone a deployment might legitimately want to move: half of an answer
    # unsupported is the line here, and somebody running this over a corpus that
    # genuinely contains their whole history could reasonably demand more. The
    # two *similarity* thresholds are not settings — see `agent/verify.py`, they
    # are properties of a specific embedding model and moving one by hand
    # without re-running the calibration would be worse than not having the knob.
    agent_min_support: float = 0.5
    # Tokens one agent turn may generate. M7.0 chose 700 for a two-sentence
    # answer over one tool result; a final answer over six hops is longer than
    # that, and a *reasoning* model spends this budget on thinking before it
    # writes anything at all — `openai/gpt-oss-20b` returned neither text nor a
    # tool call at 700, which the adapter correctly reports as an empty turn and
    # which is really a budget that was never enough.
    agent_max_tokens: int = 1400
    # Chunks per extraction request (M3.1). The free tier's binding constraint
    # is requests per day, not tokens per request, so batching is what decides
    # whether a corpus of this size can be extracted at all: 1,308 chunks one at
    # a time exceeds the daily cap before anything else goes wrong.
    extraction_batch_size: int = 8
    # Room for the JSON of a whole batch. Too small and the response is
    # truncated mid-object, which the parser reports as malformed rather than as
    # what it is.
    extraction_max_tokens: int = 2048
    # The graph projection. Bolt rather than HTTP: it is what the official
    # driver speaks, and the browser on 7474 is for reading traversals by eye.
    #
    # No key here is optional the way `gemini_api_key` is, and that difference is
    # deliberate: an absent Gemini key means answering cannot work, while an
    # unreachable Neo4j means only that the graph is unavailable. So these carry
    # working local defaults and the *reachability* — not the configuration — is
    # what everything downstream degrades on.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memoryos"
    log_level: str = "INFO"
    log_json: bool = False
    # Browser origins allowed to call this API. Empty by default, which means no
    # CORS middleware is installed at all — a browser cannot reach the API from a
    # page unless somebody deliberately says which page.
    #
    # A list rather than a string, and never `["*"]`: a wildcard on an API that
    # reads a private corpus means any page the operator visits can search it.
    # `create_app` refuses a wildcard outright rather than trusting this comment.
    cors_origins: list[str] = []

    # M11.0. Whether the session cookie carries `Secure`.
    #
    # Derived from `environment` rather than configured separately, because the
    # two cannot disagree usefully: browsers silently drop `Secure` cookies sent
    # over plain HTTP, so hardcoding it would make a local deployment unable to
    # log in at all, and hardcoding it off would send a session token in clear
    # over any real network. Anything that is not `local` gets it.
    #
    # A property rather than a field: it is not something to set, it is
    # something that follows, and a settable one is a way to be wrong.
    @property
    def session_cookie_secure(self) -> bool:
        return self.environment not in ("local", "test")

    @model_validator(mode="after")
    def _anchor_relative_paths(self) -> "Settings":
        """Resolve `blob_root` and `hf_home` against the tree, not the shell.

        Here rather than at the two call sites, so that every reader of these
        fields — the container, the replay preflight, a script somebody writes
        next week — sees the same absolute path without knowing the rule exists.
        That is the property the M10.3 failure lacked: the API and the worker read
        the same setting and disagreed about what it named.

        An absolute path is returned unchanged, which is what makes the test
        suite's temporary directories and any real deployment unaffected.
        """
        if not self.blob_root.is_absolute():
            self.blob_root = (PROJECT_ROOT / self.blob_root).resolve()
        if not self.hf_home.is_absolute():
            self.hf_home = (PROJECT_ROOT / self.hf_home).resolve()
        return self

    @model_validator(mode="after")
    def _redirect_the_test_environment(self) -> "Settings":
        """Under `MEMOS_ENVIRONMENT=test`, `database_url` *is* the test database.

        Resolved here rather than at each call site so that everything reading
        `database_url` — the container, Alembic's env.py, the shadow workspace —
        agrees without knowing the rule exists. A test run that reached the
        development database through one forgotten call site would truncate it,
        which is the failure this exists to prevent.
        """
        if self.environment == "test":
            self.database_url = self.test_database_url
            # The admin URL follows it. Alembic runs against the test database
            # too, and a migration pointed at the development one would be the
            # exact failure this hook exists to prevent — with DDL rights.
            self.database_admin_url = self.test_database_url.replace(
                "memos_app:memos_app@", "memos:memos@"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
