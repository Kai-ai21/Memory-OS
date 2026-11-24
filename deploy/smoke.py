"""Verify a deployed MEMO from outside it (M12.3).

    MEMO_SMOKE_PASSWORD=... uv run python deploy/smoke.py \\
        --url https://memo.example.com --email you@example.com

Runs the checklist a deployment has to pass, against the public URL, in the
order a person would: the page loads over HTTPS; the API refuses an anonymous
caller; login works and the cookie is Secure and HttpOnly; something ingested
becomes searchable; a question is answered and an unanswerable one is refused;
and logging out kills the session. Times the cold and the warm request.

The password comes from the environment, never from argv, where it would be in
shell history and visible in `ps`. It writes one memory (marked as a smoke test)
and leaves it, because deleting is not something a check should do.

Exit status is the number of failed checks.
"""

import argparse
import os
import sys
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "pass" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def timed(call: Callable[[], httpx.Response]) -> tuple[httpx.Response, float]:
    started = time.monotonic()
    response = call()
    return response, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", required=True, help="the deployment, e.g. https://memo.example.com"
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--cacert", help="a CA bundle, for a rehearsal with a private CA")
    parser.add_argument(
        "--unanswerable", default="what is our AWS billing setup", help="must be refused"
    )
    args = parser.parse_args()
    password = os.environ.get("MEMO_SMOKE_PASSWORD")
    if not password:
        print("set MEMO_SMOKE_PASSWORD", file=sys.stderr)
        return 2

    base = args.url.rstrip("/")
    api = f"{base}/api"
    verify: Any = args.cacert or True
    marker = f"smoke-{uuid.uuid4().hex[:8]}"
    fact = (
        f"Deployment check {marker}: the lighthouse keeper logs the tide at dawn "
        "in a green ledger kept in the boathouse."
    )

    with httpx.Client(verify=verify, timeout=120, follow_redirects=False) as http:
        print(f"MEMO at {base}")

        page, page_s = timed(lambda: http.get(f"{base}/"))
        check(
            "the page loads over HTTPS",
            page.status_code == 200 and "<html" in page.text.lower(),
            f"{page.status_code}, {page_s:.2f}s",
        )
        check("HSTS is sent", "strict-transport-security" in page.headers)

        anon = http.get(f"{api}/memories")
        check("an anonymous API request is 401", anon.status_code == 401, str(anon.status_code))
        schema = http.get(f"{api}/openapi.json")
        check(
            "the API description is not published",
            schema.status_code in (401, 404),
            str(schema.status_code),
        )

        login = http.post(f"{api}/auth/login", json={"email": args.email, "password": password})
        check("login", login.status_code == 200, str(login.status_code))
        cookie = login.headers.get("set-cookie", "").lower()
        check(
            "the session cookie is Secure and HttpOnly", "secure" in cookie and "httponly" in cookie
        )
        if login.status_code != 200:
            return len(FAILED)

        stored, stored_s = timed(lambda: http.post(f"{api}/chat", json={"text": fact}))
        memory_id = (
            next(
                (
                    m.get("memory_id")
                    for m in stored.json().get("messages", [])
                    if m.get("memory_id")
                ),
                None,
            )
            if stored.status_code == 201
            else None
        )
        check(
            "ingest: a statement is stored",
            memory_id is not None,
            f"{stored.status_code}, {stored_s:.2f}s",
        )

        searchable = False
        waited = 0.0
        if memory_id:
            started = time.monotonic()
            while time.monotonic() - started < 180:
                status = http.get(f"{api}/chat/messages/{memory_id}/status").json()
                if status.get("searchable") or status.get("stage") == "failed":
                    searchable = bool(status.get("searchable"))
                    break
                time.sleep(1)
            waited = time.monotonic() - started
        check("the worker made it searchable", searchable, f"{waited:.1f}s")

        query = "where is the tide ledger kept"
        cold, cold_s = timed(lambda: http.get(f"{api}/search", params={"q": query}))
        _warm, warm_s = timed(lambda: http.get(f"{api}/search", params={"q": query}))
        found = marker in cold.text
        check("search finds it", cold.status_code == 200 and found, str(cold.status_code))
        print(f"        search: first {cold_s:.2f}s, second {warm_s:.2f}s")

        answer, answer_s = timed(lambda: http.post(f"{api}/answer", json={"q": query}))
        if answer.status_code == 501:
            check("ask a question", False, "501: no language model configured")
        else:
            body = answer.json()
            check(
                "ask a question",
                answer.status_code == 200
                and not body["verification"]["is_refusal"]
                and "boathouse" in body["answer"].lower(),
                f"{answer.status_code}, {answer_s:.2f}s",
            )

        refusal, refusal_s = timed(
            lambda: http.post(f"{api}/answer", json={"q": args.unanswerable})
        )
        if refusal.status_code == 501:
            check("an unanswerable question is refused", False, "501: no language model configured")
        else:
            refused = refusal.status_code == 200 and refusal.json()["verification"]["is_refusal"]
            check(
                "an unanswerable question is refused",
                refused,
                f"{refusal.status_code}, {refusal_s:.2f}s: "
                f"{refusal.json().get('answer', '')[:80]!r}",
            )

        session_cookie = http.cookies.get("memos_session")
        out = http.post(f"{api}/auth/logout")
        check("logout", out.status_code in (200, 204), str(out.status_code))
        replay = httpx.get(
            f"{api}/memories", cookies={"memos_session": session_cookie or ""}, verify=verify
        )
        check("the old session cookie is dead", replay.status_code == 401, str(replay.status_code))

    print(f"\n{len(FAILED)} failed" + (f": {', '.join(FAILED)}" if FAILED else ""))
    return len(FAILED)


if __name__ == "__main__":
    sys.exit(main())
