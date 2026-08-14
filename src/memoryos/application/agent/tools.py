"""What a tool is, and the registry that dispatches to one.

**A tool here is an existing use case with a schema around it.** That is the
claim this milestone tests: six phases of capability, each already behind a
function that takes domain types and returns domain types, and exposing them to a
model should be declaration rather than implementation. Where it is not — where a
tool needs logic that does not already exist — the use case was written for a
transport rather than for the domain, and `library.py` says so at the two places
it happened.

### Arguments are declared once, as a pydantic model

`ToolSpec.parameters` is a JSON Schema, as the milestone requires, and it is
*generated* rather than written. The same model that produces the schema the
provider reads validates the arguments the model sends back, so the two cannot
disagree — a hand-written schema beside a hand-written validator is two
descriptions of one contract, and the failure mode is a tool that advertises a
parameter it then rejects.

It also means the error a model gets back is pydantic's, which is already the
shape Step 4 asks for: `limit: Input should be a valid integer, unable to parse
string as an integer`. That is correctable. A traceback is not.

### Every result carries citations

**A tool result the model can read but not attribute is how the no-fabrication
guardrail dies quietly in Phase 7.** The model reads facts, loses the source, and
writes a fluent uncited claim — and every check M2.6 built operates on the
answer's citations, so an uncited claim passes them by not being covered.

`Citation` is M2.5's type and it is chunk-shaped: it names a span of a specific
version of a memory, and `verify-citations` asserts the offsets still resolve.
Two of the six tools return facts that are not spans — a decision is a row
somebody wrote, a gap is an absence — and neither gets a fabricated offset. They
cite the corpus evidence they are linked to, and their content says plainly when
there is none. See `library.py`.

### Truncation is part of the result, not a detail of it

A tool that silently returns the first five of fifty leaves the model reasoning
about a complete picture it never had, and the answer will be confidently wrong
in a way nothing downstream can detect. `truncated` is on the result, the content
says so in words the model reads, and the cap is the tool's own.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ValidationError

from memoryos.domain.citation import Citation
from memoryos.domain.jobs import PermanentError

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What a model is told about a tool.

    `description` is the interface. A model picks a tool by reading it and
    nothing else — it cannot see the implementation, the tests, or this
    docstring — so a description that says what the tool *does* without saying
    what it is *for* routes questions to whichever tool sounds nearest.
    """

    name: str
    description: str
    # JSON Schema. Generated from a pydantic model by `spec_for`, never written
    # by hand, so the schema and the validation cannot drift apart.
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool gives back: prose for the model, provenance for the guardrail.

    `content` is what the model reads. It is deliberately text rather than JSON:
    the model has to reason about it, and a JSON blob spends tokens on syntax
    that a sentence spends on facts.
    """

    content: str
    citations: list[Citation] = field(default_factory=list)
    # True when the tool had more to give than its cap allowed. The content says
    # so as well, because the model reads the content and not this field.
    truncated: bool = False


@runtime_checkable
class Tool(Protocol):
    """Something a model can call.

    `spec` is a property rather than a class attribute so a tool can build its
    description from what it is wired to.

    `arguments` is the pydantic model the spec's schema was generated from, and
    it is on the protocol so the *registry* can validate rather than each tool
    validating itself. A tool that checked its own arguments could be called
    without the check by a caller that forgot, and the sixth tool written is the
    one that forgets.
    """

    @property
    def spec(self) -> ToolSpec: ...

    @property
    def arguments(self) -> type[BaseModel]: ...

    async def call(self, **kwargs: Any) -> ToolResult: ...


class UnknownTool(PermanentError):
    """A name no tool is registered under.

    **Permanent, and that is the whole classification.** Retrying cannot make a
    tool appear: the registry is fixed at startup, so the second attempt reads
    the same dictionary as the first. A `TransientError` here would burn the
    worker's attempt budget on a name that will never resolve, and dead-letter
    it five attempts later with the same message it had at the first.
    """

    def __init__(self, name: str, known: Sequence[str]) -> None:
        listed = ", ".join(sorted(known)) or "none"
        super().__init__(f"no tool named {name!r}; registered tools are: {listed}")
        self.name = name


class InvalidArguments(ValueError):
    """Arguments that do not fit the schema. **Returned, never raised at a model.**

    A model that passed a string where an integer belongs has made a mistake it
    can correct, and the only thing it needs in order to correct it is a sentence
    saying which field and what was expected. A traceback is a sentence it cannot
    act on wrapped in four hundred characters of this system's internals.

    So the registry catches this and hands the message back as a `ToolResult`.
    The exception type exists for the tests and for callers that want to
    distinguish "the model got it wrong" from "the tool broke".
    """

    def __init__(self, tool: str, problems: str) -> None:
        super().__init__(f"{tool}: {problems}")
        self.tool = tool
        self.problems = problems


def spec_for(
    model: type[BaseModel], *, name: str, description: str
) -> ToolSpec:
    """A spec whose schema is the model's own.

    `model_json_schema()` emits `additionalProperties: false` for a model
    configured `extra="forbid"`, which is what makes an unexpected argument an
    error rather than something silently dropped. One provider refuses that key;
    stripping it is the adapter's job, not this function's — see
    `adapters/llm/gemini.py`.
    """
    schema = model.model_json_schema()
    # The title is the *class* name, which is an implementation detail leaking
    # into a prompt. The tool's name is on the spec already.
    schema.pop("title", None)
    return ToolSpec(name=name, description=description, parameters=schema)


def validate(model: type[BaseModel], tool: str, arguments: dict[str, Any]) -> BaseModel:
    """Arguments as a validated model, or `InvalidArguments` naming what is wrong.

    Every message pydantic produces is joined, rather than only the first. A
    model that passed two bad arguments and was told about one would fix it,
    call again, and be told about the other — two turns and two tool calls to
    learn something one sentence could have said.
    """
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors()
        )
        raise InvalidArguments(tool, problems) from exc


class ToolRegistry:
    """Which tools exist, and dispatch by name.

    A dictionary and a lookup. It is a class rather than a dict because two
    behaviours belong with it and neither is a dict's: an unknown name is a
    `PermanentError` with the known names in it, and bad arguments come back as
    a result the model can read rather than as an exception the loop must catch.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool. Registering a name twice is refused.

        Refused rather than replaced, for the reason `EventBus.subscribe` gives:
        two tools under one name means the model's choice no longer identifies
        which code runs, and it would show up as the wrong tool answering rather
        than as an error at startup.
        """
        if tool.spec.name in self._tools:
            raise ValueError(f"a tool named {tool.spec.name!r} is already registered")
        self._tools[tool.spec.name] = tool

    def specs(self) -> list[ToolSpec]:
        """Every spec, in registration order.

        Order is the order they were registered rather than alphabetical,
        because it is the order they appear in the prompt and the first one
        listed is the one a model reaches for when the question is vague. The
        container registers search first for exactly that reason.
        """
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run one tool. Validation happens here, before the tool is touched.

        **Before, not inside.** Every tool's `call` can therefore assume its
        arguments are the shape it declared, which is what keeps the six of them
        free of argument handling entirely.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownTool(name, self.names())

        try:
            validated = validate(tool.arguments, name, args)
        except InvalidArguments as exc:
            # Handed back rather than raised. This is the model's mistake and
            # the model is the one that can fix it; a raise here would abort a
            # turn over something a sentence resolves.
            logger.info("tool.invalid_arguments", tool=name, problems=exc.problems)
            return ToolResult(
                content=(
                    f"The arguments for {name} were not valid: {exc.problems}. "
                    "Call it again with corrected arguments."
                ),
            )

        result = await tool.call(**validated.model_dump())
        logger.info(
            "tool.called",
            tool=name,
            arguments=sorted(args),
            citations=len(result.citations),
            truncated=result.truncated,
            content_chars=len(result.content),
        )
        return result
