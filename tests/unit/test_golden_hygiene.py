"""The benchmark must not be written into the thing it measures.

This repository is its own corpus, so a tracked file quoting a golden query
makes that file a literal match for it. M2.2 measured the cost: the keyword
retriever ranked the acceptance test first for one query — the test exists
*because of* that query — and fusion promoted the file that names the question
above anything that answers it.

Two mechanisms, and this is the second one. `eval_exclude` drops files whose
whole purpose is to hold those strings; this test stops new contamination
appearing anywhere else. It fails in CI rather than quietly inflating a score,
which is the only way a benchmark stays honest as the repository grows.
"""

import json
import subprocess
from pathlib import Path

import pytest

from memoryos.application.golden import excluded_by, is_prose_query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET = PROJECT_ROOT / "var" / "golden-set.json"


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def test_no_prose_golden_query_appears_verbatim_in_a_tracked_file() -> None:
    """A question somebody typed has no business being in the source.

    **Only prose queries.** `SKIP LOCKED`, `prefix_chars` and `ON DELETE CASCADE`
    are all over this corpus and have to be: they are the literals those queries
    exist to find, and a repository that did not contain them would make the
    query meaningless. The contamination that matters is a natural-language
    question appearing in a file, which can only happen by writing the benchmark
    into the thing being measured. `is_prose_query` draws that line on function
    words, because no identifier contains one.

    Files named by `eval_exclude` are exempt, and that is the deal: a file is
    allowed to hold a query verbatim *or* to be scored against, never both.
    """
    if not GOLDEN_SET.exists():
        pytest.skip("no exported golden set to check")

    payload = json.loads(GOLDEN_SET.read_text())
    exclude = payload.get("eval_exclude") or []
    queries = [
        query["query_text"]
        for query in payload["queries"]
        if is_prose_query(query["query_text"])
    ]
    assert queries, "the golden set should contain natural-language queries"

    found: dict[str, list[str]] = {}
    for name in tracked_files():
        if name.startswith("var/") or excluded_by(exclude, name) is not None:
            continue
        try:
            text = (PROJECT_ROOT / name).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for query in queries:
            if query in text:
                found.setdefault(query, []).append(name)

    assert not found, (
        "golden queries are quoted in tracked files, which makes those files "
        "lexical matches for the benchmark:\n"
        + "\n".join(f"  {query!r} in {files}" for query, files in sorted(found.items()))
        + "\nEither rephrase the file to describe the query instead of quoting it, "
        "or add it to eval_exclude if holding the string is its purpose."
    )


AGENT_GOLDEN = PROJECT_ROOT / "var" / "agent-golden.json"


def test_no_agent_golden_question_appears_verbatim_in_a_tracked_file() -> None:
    """**M2.1's rule, applied to M7.3's answer key.**

    The corpus is this repository, so an agent question written into a tracked
    file is a passage that matches itself — and the agent's first hop is a search
    over exactly that corpus. Where retrieval contamination inflated a ranking,
    this would hand the agent the answer key as its top hit and make a trajectory
    look brilliant for reading the benchmark.

    Every one of these is prose, so unlike the retrieval set there is no
    `is_prose_query` exemption to make: none of them is a literal a file could
    legitimately contain.

    `var/` is skipped for the same reason it is skipped above — `var/**` is in
    `eval_exclude`, and that exclusion is the entire licence for writing the
    questions down anywhere at all.
    """
    if not AGENT_GOLDEN.exists():
        pytest.skip("no agent golden set to check")

    payload = json.loads(AGENT_GOLDEN.read_text())
    questions = [entry["question"] for entry in payload["questions"]]
    assert questions, "the agent golden set should contain questions"

    exclude: list[str] = []
    if GOLDEN_SET.exists():
        exclude = json.loads(GOLDEN_SET.read_text()).get("eval_exclude") or []

    found: dict[str, list[str]] = {}
    for name in tracked_files():
        if name.startswith("var/") or excluded_by(exclude, name) is not None:
            continue
        try:
            text = (PROJECT_ROOT / name).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for question in questions:
            if question in text:
                found.setdefault(question, []).append(name)

    assert not found, (
        "agent golden questions are quoted in tracked files, which makes those "
        "files retrievable answers to the benchmark:\n"
        + "\n".join(f"  {q!r} in {files}" for q, files in sorted(found.items()))
        + "\nDescribe the question rather than quoting it — the README refers to "
        "these by id for exactly this reason."
    )
