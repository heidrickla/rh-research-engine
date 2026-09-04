"""Where an indexed formula came from.

A structural match in the formula index answers "something like this is already
recorded". Without provenance it cannot answer the question that follows, which
is the one that matters: *which theorem, in which source, under which
hypotheses?* Two formulas can be structurally identical and mean different
things because one carries an unstated assumption the other does not.

WHAT A CITATION IS NOT. It confers no epistemic status. "Appears in a paper" is
not "proved", and it is not even "known" in the sense
:mod:`rh_research_engine.contracts.epistemic` means -- that requires the
statement to be established external mathematics, which a citation alone does
not establish. A paper can be wrong, withdrawn, or misread by whoever indexed
it, and a formula transcribed out of one loses whatever hypotheses the
surrounding text carried.

So :class:`Citation` is deliberately inert: descriptive fields only, no verdict,
no status, and nothing downstream promotes on the strength of one. Deciding that
a citation establishes something is
:class:`~rh_research_engine.contracts.artifacts.LiteratureMatch`'s job, and that
model demands the comparison be spelled out -- source identifiers *and* the
theorem's assumptions -- before any decisive verdict is allowed.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..contracts.hashing import canonical_json


class SourceKind(StrEnum):
    """Where a formula was read from."""

    #: An ingested document: paper, preprint, notes.
    PAPER = "paper"
    #: A record in this project's durable mathematical memory.
    KNOWLEDGE = "knowledge"
    #: Entered by hand at the CLI. Carries the least provenance of the three,
    #: and says so rather than looking like the others.
    MANUAL = "manual"


class Citation(BaseModel):
    """A pointer to the exact place a formula was read from.

    Every field is descriptive. There is no status here, on purpose -- see the
    module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    source_kind: SourceKind
    #: Where it was read: a file path, a knowledge record ID, or a note.
    source_id: str
    #: DOI, arXiv ID, ISBN -- whatever identifies the source outside this
    #: repository. Optional, because notes and durable-memory records have none,
    #: and inventing one would be worse than leaving it empty.
    identifier: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    #: The label the source itself uses: "Theorem 3.2", "Lemma 1", "(4.7)".
    #: This is what turns "it is in that paper somewhere" into a citation.
    theorem_label: str | None = None
    section: str | None = None
    line: int | None = None
    #: The surrounding statement as written, so the hypotheses travel with the
    #: formula instead of being left behind in the source.
    statement: str | None = None
    #: Logical tick, not wall-clock: nothing here may observe time, or the same
    #: replay produces different records on different days.
    retrieved_at_tick: int | None = None
    notes: list[str] = Field(default_factory=list)

    def citation_hash(self) -> str:
        """Content identity, so the same citation indexes to the same value."""
        return hashlib.sha256(
            canonical_json(self.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()

    def describe(self) -> str:
        """One line a researcher can act on, naming what is missing.

        A citation that cannot be followed back to a specific claim is worth
        saying so about, rather than rendering as a tidy-looking reference.
        """
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.authors:
            parts.append(", ".join(self.authors))
        if self.identifier:
            parts.append(self.identifier)
        if self.theorem_label:
            parts.append(self.theorem_label)
        location = self.section or (f"line {self.line}" if self.line else None)
        if location:
            parts.append(location)
        if not parts:
            return f"{self.source_kind.value}:{self.source_id} (no citation detail recorded)"
        rendered = " — ".join(parts)
        if not self.theorem_label:
            rendered += " [no theorem label: the exact result is not pinned]"
        return rendered


def knowledge_citation(item, *, formula_index: int | None = None) -> Citation:
    """Cite a durable-memory record.

    The statement travels with it. A formula lifted out of a knowledge record
    without its statement loses the conditions the record attached to it, and
    `conditional_on_RH_standard` is exactly the status where that matters.
    """
    return Citation(
        source_kind=SourceKind.KNOWLEDGE,
        source_id=item.id,
        title=item.title,
        theorem_label=f"formula {formula_index}" if formula_index is not None else None,
        statement=item.statement,
        notes=[f"durable-memory status: {item.status.value}"],
    )
