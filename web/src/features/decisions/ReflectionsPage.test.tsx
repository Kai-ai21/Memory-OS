/**
 * Three properties this page exists to hold, and one the app holds around it.
 *
 * **Citations are links.** A behavioural claim whose `[2]` is punctuation rather
 * than a route into the decision it came from is a horoscope with a footnote.
 *
 * **An uncited sentence is visible as one.** The server flags rather than
 * removes; a client that rendered the flagged text identically to the cited text
 * would undo that entirely.
 *
 * **The counts sit beside the prose.** Six decisions with two arguing back is a
 * different claim from six with none, and the paragraph does not say which.
 *
 * And the one the app holds: **nothing reaches this page by itself.** The
 * masthead has no reflections tab, so a claim about your judgement is something
 * you go and look at rather than something the tool volunteers.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../../App";
import { ReflectionsPage } from "./ReflectionsPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

const REFLECTION = {
  id: "44444444-4444-7444-8444-444444444444",
  pattern_id: "11111111-1111-7111-8111-111111111111",
  pattern_statement: "A recurring assumption breaks more often than it holds.",
  text:
    "You have underestimated deployment work four times, most clearly in Which deploy path? [1]. " +
    "It held once, in Which migration path? [2]. " +
    "You are an optimist by temperament.",
  citation_rate: 0.667,
  model_id: "groq/llama@1",
  generated_at: "2026-08-13T12:00:00Z",
  acknowledged_at: null,
  dismissed_at: null,
  dismissed_reason: null,
  support_count: 4,
  contradiction_count: 1,
  citations: [
    {
      marker: 1,
      decision_id: "22222222-2222-7222-8222-222222222222",
      decision_question: "Which deploy path?",
      relation: "supports",
    },
    {
      marker: 2,
      decision_id: "33333333-3333-7333-8333-333333333333",
      decision_question: "Which migration path?",
      relation: "contradicts",
    },
  ],
  uncited: ["You are an optimist by temperament."],
};

afterEach(() => vi.unstubAllGlobals());

describe("the reflections page", () => {
  it("renders every citation as a link to the decision it came from", async () => {
    stubFetch([{ match: "/reflections", body: [REFLECTION] }]);
    renderWithProviders(<ReflectionsPage />);

    const supporting = await screen.findByRole("link", { name: "1" });
    expect(supporting).toHaveAttribute(
      "href",
      "/decisions/22222222-2222-7222-8222-222222222222",
    );
    // The counter-evidence citation is as clickable as the supporting one, and
    // says which side it is on.
    const contradicting = screen.getByRole("link", { name: "2" });
    expect(contradicting).toHaveAttribute(
      "href",
      "/decisions/33333333-3333-7333-8333-333333333333",
    );
    expect(contradicting).toHaveAttribute(
      "title",
      "argues against: Which migration path?",
    );
  });

  it("marks the sentence that carries no citation", async () => {
    stubFetch([{ match: "/reflections", body: [REFLECTION] }]);
    const { container } = renderWithProviders(<ReflectionsPage />);
    await screen.findByRole("link", { name: "1" });

    const flagged = container.querySelector("[title^='no citation']");
    expect(flagged).not.toBeNull();
    expect(flagged?.textContent).toContain("optimist");
  });

  it("shows the citation rate and both evidence counts beside the prose", async () => {
    stubFetch([{ match: "/reflections", body: [REFLECTION] }]);
    const { container } = renderWithProviders(<ReflectionsPage />);
    await screen.findByText("4 supporting");

    // One decimal place, as `percent` renders everywhere else: a rate shown as
    // "67%" and one shown as "66.7%" are the same number, and the interface
    // should not have two ways of saying it.
    expect(container.textContent).toContain("cited 66.7%");
    expect(screen.getByText("4 supporting")).toBeInTheDocument();
    expect(screen.getByText("1 contradicting")).toBeInTheDocument();
  });

  it("refuses to dismiss without a reason, and sends it when given", async () => {
    const calls = stubFetch([
      { match: "/reflections/44444444", status: 204 },
      { match: "/reflections", body: [REFLECTION] },
    ]);
    renderWithProviders(<ReflectionsPage />);
    await screen.findByText("4 supporting");

    const dismiss = screen.getByRole("button", { name: "dismiss" });
    expect(dismiss).toBeDisabled();

    await userEvent.type(
      screen.getByLabelText(`dismiss reason for ${REFLECTION.id}`),
      "that is not why I did any of that",
    );
    await userEvent.click(dismiss);

    const posted = calls.find((call) => call.url.includes("/dismiss"));
    expect(posted?.method).toBe("POST");
    expect(posted?.body).toEqual({ reason: "that is not why I did any of that" });
  });

  it("reads the empty state as a result rather than a failure", async () => {
    stubFetch([{ match: "/reflections", body: [] }]);
    renderWithProviders(<ReflectionsPage />);

    expect(await screen.findByText("no reflections")).toBeInTheDocument();
    expect(
      screen.getByText(/refused before a model is called rather than hedged/),
    ).toBeInTheDocument();
  });
});

describe("where reflections are not", () => {
  it("has no nav tab, so nothing volunteers a claim about you", async () => {
    // The home screen renders search; it must not ask for reflections at all.
    stubFetch([{ match: "/search", body: { hits: [], total: 0 } }]);
    renderWithProviders(<App />, { route: "/" });

    const nav = screen.getByRole("navigation");
    expect(nav.textContent).not.toMatch(/reflection/i);
    // Every tab that does exist, named, so adding one here is a deliberate act
    // rather than something a refactor does quietly. `surfacing` joins in M6.3
    // and the two are not in tension: a reflection is a claim about somebody's
    // judgement, and volunteering one is the failure M5.4 was built to avoid,
    // while that page is a record of what the system has *already* interrupted
    // them with. Hiding the place to say "that was noise" is how a dismissal
    // rate stays flattering.
    //
    // `ask` joins in M7.2, and it is the tab that comes closest to the line this
    // test guards. It produces prose about the corpus, which is what reflections
    // do — and it is here because the person asked for it and because every
    // sentence arrives with its support marked in place. Volunteering a claim is
    // still the thing that does not happen: nothing on this page speaks first.
    //
    // `model` joins in M8.0 and is closer still: it is a page of claims about
    // the person, which is the register a reflection is written in. It is here
    // for the same two reasons — the person navigated to it, and every statement
    // carries its support count while every absence carries its cause. The rule
    // this test protects is about *volunteering*, and nothing on that page
    // speaks unasked either.
    expect(nav.textContent).toBe(
      "searchasktimelinedecisionsjudgementscorpussurfacingmodel",
    );
  });
});
