-- A second database for the test suite.
--
-- The integration tests truncate every table, because truncation is the only
-- isolation strategy that survives code under test committing its own
-- transactions. Pointed at the development database, that is `pytest` deleting
-- a working corpus — which is what it did, three times, during M2.0a.
--
-- Created here rather than by a fixture so that it exists before anything
-- connects, and so `docker compose down -v && docker compose up -d` reproduces
-- the whole arrangement. `MEMOS_ENVIRONMENT=test` is what points the suite at it.
CREATE DATABASE memos_test;

\connect memos_test

-- The initial migration creates this too, but a database that has the extension
-- before Alembic runs is one fewer ordering dependency.
CREATE EXTENSION IF NOT EXISTS vector;
