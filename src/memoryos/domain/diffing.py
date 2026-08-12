"""What changed between two texts, as spans.

Pure Python over `difflib`. No dependency, and none is warranted: these diffs are
read by a person and summarised by a model, never applied as a patch. A real
patch format would have to be exact about context lines and offsets in a way
nothing here consumes.

**Line-level, not character-level.** `SequenceMatcher` over characters on a
50,000-character README finds thousands of one-character runs and reports
something technically correct and completely unreadable. Lines are the unit a
person diffs in, and the char offsets are recovered afterwards by summing line
lengths — so the spans are exact even though the matching is not fine-grained.

**The texts are normalized, never raw bytes.** That is what makes a line-ending
change produce an empty diff, and it is a live check on M1.4: if normalization
ever stops collapsing CRLF, this starts reporting every line of the file as
changed and the emptiness test fails loudly.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from enum import StrEnum, auto


class ChangeKind(StrEnum):
    ADDED = auto()
    REMOVED = auto()
    CHANGED = auto()


@dataclass(frozen=True, slots=True)
class Span:
    """One contiguous run of change, located in both texts.

    Both sets of offsets are carried because the two are needed for different
    things and neither can be derived from the other: `b_*` locates the change
    in the version that still has chunks, and `a_*` is what lets the removed
    text be shown beside it rather than described.

    An `ADDED` span has `a_start == a_end` — an empty range at the point in the
    old text where the new material was inserted — and `REMOVED` is the mirror.
    That is more useful than a null, because it says *where* in the old text the
    insertion landed.
    """

    kind: ChangeKind
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    a_text: str
    b_text: str

    @property
    def added_chars(self) -> int:
        return len(self.b_text)

    @property
    def removed_chars(self) -> int:
        return len(self.a_text)


def diff_spans(before: str, after: str) -> list[Span]:
    """Every changed run between two texts, with character offsets into each.

    `autojunk` is off, and that is not a tuning knob. `SequenceMatcher`'s
    heuristic treats any element appearing in more than 1% of a sequence longer
    than 200 as junk and refuses to anchor on it — and in source code the
    commonest lines are blank lines and lines that are only indentation. With it
    on, a diff of two versions of a 1,500-line Python file loses most of its
    anchors and reports one enormous replacement instead of the three functions
    that actually changed. The result is not wrong so much as useless, and it
    degrades silently as files get longer.
    """
    a_lines = before.splitlines(keepends=True)
    b_lines = after.splitlines(keepends=True)
    a_offsets = _offsets(a_lines)
    b_offsets = _offsets(b_lines)

    spans: list[Span] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, a_lines, b_lines, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        a_start, a_end = a_offsets[i1], a_offsets[i2]
        b_start, b_end = b_offsets[j1], b_offsets[j2]
        spans.append(
            Span(
                kind=_KINDS[tag],
                a_start=a_start,
                a_end=a_end,
                b_start=b_start,
                b_end=b_end,
                a_text=before[a_start:a_end],
                b_text=after[b_start:b_end],
            )
        )
    return spans


def unified(
    before: str,
    after: str,
    *,
    a_label: str = "before",
    b_label: str = "after",
    context: int = 2,
) -> str:
    """A unified diff, for a person to read and a model to be shown.

    Two lines of context rather than three. This text is the *whole* evidence a
    change summary is allowed to draw on, so every unchanged line in it is a line
    the model can describe as though it changed — context is necessary to make
    the change legible and it is not free.
    """
    return "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=a_label,
            tofile=b_label,
            n=context,
        )
    )


_KINDS = {
    "replace": ChangeKind.CHANGED,
    "delete": ChangeKind.REMOVED,
    "insert": ChangeKind.ADDED,
}


def _offsets(lines: list[str]) -> list[int]:
    """Character offset of each line, plus the end of the text.

    One longer than `lines`, so `offsets[i2]` is valid for the exclusive end of
    the last opcode without a special case.
    """
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets
