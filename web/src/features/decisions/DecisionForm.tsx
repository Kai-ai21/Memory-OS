/**
 * Capture, in the order the questions are worth asking.
 *
 * Alternatives before reasoning, because naming what lost changes what somebody
 * writes about why the winner won. Assumptions last and with their own add
 * button, because they are the hardest field and asking for them first stops
 * people finishing the form at all — the same ordering `decide --interactive`
 * uses, for the same reason.
 *
 * **Nothing here is prefilled and nothing defaults.** An empty confidence stays
 * empty rather than becoming 0.5: a number the form invented would be measured
 * by M5.2 as though somebody had held it. The submit button refuses a decision
 * with no alternative and says why, rather than disabling itself silently —
 * a disabled button with no explanation is a dead end.
 */

import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api, type DecisionIn } from "../../api/client";
import { Failure, SectionHeading } from "../../components/primitives";

interface OptionDraft {
  description: string;
  rejected_because: string;
}

/**
 * What the review queue hands over when a reviewer chooses to edit a draft.
 *
 * Edit-then-accept goes through this form rather than an inline editor in the
 * queue, and that is deliberate: a reviewer who has read the passage usually
 * knows a confidence and at least one assumption the model could not have, and
 * this is the screen that asks for them properly. `acceptSuggestionId` makes the
 * submit an accept rather than a create, so one act both writes the decision and
 * closes the queue entry — accepting first and editing afterwards would leave a
 * record in the table that nobody stands behind, however briefly.
 */
export interface PrefilledDraft {
  acceptSuggestionId?: string;
  question?: string;
  chosen?: string;
  reasoning?: string;
  options?: OptionDraft[];
  assumptions?: string[];
}

export function DecisionForm() {
  const navigate = useNavigate();
  const client = useQueryClient();
  const prefill = (useLocation().state ?? {}) as PrefilledDraft;

  const [question, setQuestion] = useState(prefill.question ?? "");
  const [chosen, setChosen] = useState(prefill.chosen ?? "");
  const [reasoning, setReasoning] = useState(prefill.reasoning ?? "");
  // Never prefilled, even when the draft carried one. The queue reports what the
  // model said; this field is the reviewer's own claim about what they believe
  // now, and starting it from a model's number would make it the model's.
  const [confidence, setConfidence] = useState("");
  const [expected, setExpected] = useState("");
  const [options, setOptions] = useState<OptionDraft[]>(
    prefill.options?.length ? prefill.options : [{ description: "", rejected_because: "" }],
  );
  const [assumptions, setAssumptions] = useState<string[]>(
    prefill.assumptions?.length ? prefill.assumptions : [""],
  );
  const [refusal, setRefusal] = useState<string | null>(null);

  const submit = useMutation({
    mutationFn: (body: DecisionIn) =>
      prefill.acceptSuggestionId
        ? api.acceptSuggestion(prefill.acceptSuggestionId, body)
        : api.decide(body),
    onSuccess: async (created) => {
      await client.invalidateQueries({ queryKey: ["decisions"] });
      await client.invalidateQueries({ queryKey: ["suggestions"] });
      navigate(`/decisions/${created.id}`);
    },
  });

  function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    const alternatives = options
      .map((option) => ({
        description: option.description.trim(),
        rejected_because: option.rejected_because.trim() || null,
      }))
      .filter((option) => option.description.length > 0);

    if (alternatives.length === 0) {
      // Stated in the UI as well as refused by the API. The rule is the point of
      // the form, so it should not take a round trip to learn it.
      setRefusal(
        "A decision needs at least one alternative that was considered and not taken. " +
          "Without one this is a description of what happened, and no later outcome can " +
          "be read against it.",
      );
      return;
    }
    setRefusal(null);

    submit.mutate({
      question: question.trim(),
      chosen: chosen.trim(),
      reasoning: reasoning.trim() || null,
      // Parsed only when something was typed. An empty box stays null.
      confidence: confidence.trim() === "" ? null : Number(confidence),
      expected_outcome: expected.trim() || null,
      options: alternatives,
      assumptions: assumptions
        .map((statement) => statement.trim())
        .filter(Boolean)
        .map((statement) => ({ statement, confidence: null })),
      evidence: [],
      decided_at: null,
    });
  }

  return (
    <form className="flex max-w-3xl flex-col gap-5" onSubmit={onSubmit}>
      <SectionHeading right={prefill.acceptSuggestionId ? "from a suggestion" : undefined}>
        record a decision
      </SectionHeading>
      {prefill.acceptSuggestionId ? (
        <p className="meta max-w-prose leading-relaxed text-faint">
          Editing a draft. Submitting accepts the suggestion and writes this, with the
          passage it came from attached as <code className="kbd">records</code> evidence.
          Confidence and assumptions are blank on purpose — they are yours, not the
          model&apos;s.
        </p>
      ) : null}

      <Field label="what was being decided">
        <input
          className="field"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          required
          aria-label="question"
        />
      </Field>

      <Field label="what was chosen">
        <input
          className="field"
          value={chosen}
          onChange={(event) => setChosen(event.target.value)}
          required
          aria-label="chosen"
        />
      </Field>

      <section className="flex flex-col gap-2">
        <SectionHeading right="required">what else was on the table</SectionHeading>
        {options.map((option, index) => (
          <div key={index} className="flex flex-col gap-1 border-l-2 border-rule pl-3">
            <input
              className="field"
              placeholder="option"
              aria-label={`option ${index + 1}`}
              value={option.description}
              onChange={(event) =>
                setOptions((current) =>
                  current.map((item, position) =>
                    position === index
                      ? { ...item, description: event.target.value }
                      : item,
                  ),
                )
              }
            />
            <input
              className="field"
              placeholder="rejected because"
              aria-label={`rejected because ${index + 1}`}
              value={option.rejected_because}
              onChange={(event) =>
                setOptions((current) =>
                  current.map((item, position) =>
                    position === index
                      ? { ...item, rejected_because: event.target.value }
                      : item,
                  ),
                )
              }
            />
          </div>
        ))}
        <button
          type="button"
          className="meta self-start text-amber underline"
          onClick={() =>
            setOptions((current) => [...current, { description: "", rejected_because: "" }])
          }
        >
          another option
        </button>
      </section>

      <Field label="why this one">
        <textarea
          className="field h-24"
          value={reasoning}
          onChange={(event) => setReasoning(event.target.value)}
          aria-label="reasoning"
        />
      </Field>

      <Field label="confidence, right now (0–1)">
        <input
          className="field w-24"
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={confidence}
          onChange={(event) => setConfidence(event.target.value)}
          aria-label="confidence"
        />
        <p className="meta mt-1 text-faint">
          Recorded as of now and never updated. A number revised after the outcome measures
          nothing.
        </p>
      </Field>

      <Field label="what you expect to happen">
        <input
          className="field"
          value={expected}
          onChange={(event) => setExpected(event.target.value)}
          aria-label="expected outcome"
        />
      </Field>

      <section className="flex flex-col gap-2">
        <SectionHeading right="none is an answer">
          what has to be true for this to be right
        </SectionHeading>
        {assumptions.map((statement, index) => (
          <input
            key={index}
            className="field"
            placeholder="assumption"
            aria-label={`assumption ${index + 1}`}
            value={statement}
            onChange={(event) =>
              setAssumptions((current) =>
                current.map((item, position) =>
                  position === index ? event.target.value : item,
                ),
              )
            }
          />
        ))}
        <button
          type="button"
          className="meta self-start text-amber underline"
          onClick={() => setAssumptions((current) => [...current, ""])}
        >
          another assumption
        </button>
      </section>

      {refusal ? (
        <p className="meta max-w-prose border-l-2 border-deny bg-raised p-3 leading-relaxed text-ink">
          {refusal}
        </p>
      ) : null}
      {submit.isError ? <Failure error={submit.error} /> : null}

      <button
        type="submit"
        className="meta-label self-start border border-edge px-3 py-1 text-amber"
        disabled={submit.isPending}
      >
        {submit.isPending
          ? "recording…"
          : prefill.acceptSuggestionId
            ? "accept with these edits"
            : "record"}
      </button>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="meta-label text-muted">{label}</span>
      {children}
    </label>
  );
}
