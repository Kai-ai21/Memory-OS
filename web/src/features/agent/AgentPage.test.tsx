/**
 * Three properties, and all three are about the unsupported part being visible.
 *
 * **The flagged sentence is in the paragraph.** Not removed, not collapsed, not
 * behind a toggle — a reader believes an ungrounded sentence because it sits in
 * a paragraph of grounded ones and looks the same, so the mark has to be there
 * while they are reading it.
 *
 * **A withheld answer is withheld on screen too.** The refusal is what renders,
 * and the draft is nowhere in the DOM. A guardrail the browser could reveal is
 * not a guardrail.
 *
 * **The trajectory renders either way**, because what was retrieved is the
 * evidence for a refusal as much as for an answer.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AgentPage } from "./AgentPage";
import { renderWithProviders, stubFetch } from "../../test/harness";

const SUPPORTED = "The lease expires after thirty seconds.";
const INVENTED = "The team changed its deployment process in March.";

function claim(text: string, index: number, supported: boolean, support: string) {
  return {
    text,
    sentence_index: index,
    cited_step: 1,
    supported,
    support_excerpt: supported ? "worker.py: lease = 30s" : null,
    support: support,
    similarity: supported ? 0.71 : 0.53,
    steps: supported ? [1] : [],
    factual: true,
    from_truncated: false,
  };
}

const STEP = {
  thought: "",
  tool: "search_memories",
  args: { query: "lease" },
  result: "worker.py: a worker holds a lease on the job it claimed",
  citations: 1,
  truncated: false,
  novel: true,
  tokens: 400,
  duration_ms: 900,
};

const ANSWERED = {
  question: "how long is a lease",
  answer: `${SUPPORTED} ${INVENTED}`,
  raw_answer: `${SUPPORTED} ${INVENTED}`,
  verification: {
    support_rate: 0.5,
    direct_rate: 1.0,
    verdict: "partial",
    factual_claims: 2,
    connective_claims: 0,
    claims: [claim(SUPPORTED, 0, true, "direct"), claim(INVENTED, 1, false, "unsupported")],
    invalid_citations: [],
    truncated_citations: [],
    unresolved_citations: [],
    refused: false,
  },
  stopped_because: "confidence",
  hops: 1,
  steps: [STEP],
  citations: [
    {
      memory_id: "11111111-1111-7111-8111-111111111111",
      source_name: "self",
      external_key: "src/worker.py",
      chunk_ordinal: 0,
      char_start: 0,
      char_end: 40,
      prefix_chars: 0,
      excerpt: "a worker holds a lease",
      version: 1,
    },
  ],
  cost: { model_calls: 2, prompt_tokens: 900, completion_tokens: 40, duration_ms: 4000 },
  truncated: false,
  error: null,
  retry_after: null,
};

const REFUSED = {
  ...ANSWERED,
  answer: "I could not answer that from what I retrieved. The searches I ran did not return material that supports an answer.",
  raw_answer: null,
  verification: {
    ...ANSWERED.verification,
    support_rate: 0.0,
    verdict: "ungrounded",
    factual_claims: 3,
    claims: [claim(INVENTED, 0, false, "unsupported")],
    refused: true,
  },
};

async function ask(body: unknown) {
  stubFetch([{ match: "/agent/ask", body }]);
  renderWithProviders(<AgentPage />);
  await userEvent.type(screen.getByLabelText("ask"), "how long is a lease");
  await userEvent.click(screen.getByRole("button", { name: "run" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("the ask page", () => {
  it("marks the unsupported sentence in place rather than removing it", async () => {
    await ask(ANSWERED);

    // Both sentences on screen. The flagged one is present, in order, with its
    // mark beside it — the reader sees the claim and the doubt together.
    expect(await screen.findByText(SUPPORTED)).toBeInTheDocument();
    const flagged = screen.getByText(INVENTED);
    expect(flagged).toBeInTheDocument();
    expect(flagged.className).toMatch(/underline/);
    expect(screen.getByText("[unsupported]")).toBeInTheDocument();
    // And the verdict is stated rather than implied by the marking alone.
    expect(screen.getByText("partial")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("shows the refusal and not the draft when the answer was withheld", async () => {
    await ask(REFUSED);

    expect(await screen.findByText(/could not answer that/)).toBeInTheDocument();
    expect(screen.getByText(/drafted and withheld/)).toBeInTheDocument();
    // The withheld sentence is nowhere on the page. The API did not send it and
    // this page has nothing to reconstruct it from, which is the point.
    expect(screen.queryByText(INVENTED)).not.toBeInTheDocument();
    expect(screen.getByText("ungrounded")).toBeInTheDocument();
  });

  it("shows the hops behind an answer and behind a refusal alike", async () => {
    await ask(REFUSED);

    expect(await screen.findByText("search_memories")).toBeInTheDocument();
    expect(screen.getByText(/a worker holds a lease/)).toBeInTheDocument();
    expect(screen.getByText("hop 1")).toBeInTheDocument();
  });
});
