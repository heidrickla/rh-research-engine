from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sympy as sp
from pydantic import BaseModel, Field
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .citations import Citation, knowledge_citation
from .equivalence import fingerprint
from .functions import SREPR_NAMESPACE
from .models import PaperIngestionResult
from .parser import prepare_for_parsing

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


class FormulaRecord(BaseModel):
    id: str
    expression: str
    canonical_hash: str
    canonical: str
    kind: str = "expression"
    lhs: str | None = None
    rhs: str | None = None
    source: str | None = None
    line: int | None = None
    tags: list[str] = Field(default_factory=list)
    #: Where this formula was read from. Optional because records predating
    #: citations exist on disk; a structural match without one can say "this
    #: looks familiar" and nothing more.
    #:
    #: Deliberately not a status. See `citations.py`: appearing in a paper is
    #: not being proved, and this index has no epistemic field for a citation
    #: to raise even if one were tempted to let it.
    citation: Citation | None = None


class FormulaMatch(BaseModel):
    record: FormulaRecord
    exact: bool
    structural_score: float
    shared_tokens: list[str] = Field(default_factory=list)

    @property
    def provenance(self) -> str:
        """Where the matched formula came from, or that nothing is recorded.

        A match with no provenance is still a useful signal and a useless
        citation, and the difference should be visible at the point of reading
        rather than discovered later.
        """
        if self.record.citation is None:
            return "no source recorded"
        return self.record.citation.describe()


def _tokens(expression: str) -> set[str]:
    text, locals_ = prepare_for_parsing(expression)
    expr = parse_expr(
        text, transformations=_TRANSFORMS, local_dict=locals_, evaluate=False
    )
    tokens: set[str] = set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Symbol):
            tokens.add("SYM")
        elif isinstance(node, sp.Integer):
            tokens.add("INT")
        elif isinstance(node, sp.Rational):
            tokens.add("RAT")
        elif getattr(node, "is_Function", False):
            tokens.add(f"FN:{node.func.__name__}")
        else:
            tokens.add(type(node).__name__)
    return tokens


def _exact_expression(equation) -> sp.Basic | None:
    """The object the extractor built, not a re-reading of how it printed.

    `sympy_srepr` round-trips exactly; `normalized` is display text, and the
    two are not interchangeable. Returns None when there is no srepr to use,
    leaving the caller on the string path.

    The srepr is produced in this process by the extractor immediately before
    this call. It is never read back from the stored index, which is why
    `sympify` is acceptable here and would not be on untrusted input.
    """
    if not equation.sympy_srepr:
        return None
    try:
        return sp.sympify(equation.sympy_srepr, locals=dict(SREPR_NAMESPACE))
    except Exception:
        return None


def _relation_fingerprint(lhs: str | sp.Basic, rhs: str | sp.Basic) -> tuple[str, str]:
    left = fingerprint(lhs).canonical
    right = fingerprint(rhs).canonical
    canonical = f"Eq({left},{right})"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return canonical, digest


class FormulaIndex:
    def __init__(self, path: Path = Path("research_state/formula_index.json")) -> None:
        self.path = path

    def load(self) -> list[FormulaRecord]:
        if not self.path.exists():
            return []
        return [FormulaRecord.model_validate(item) for item in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self, records: list[FormulaRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.model_dump(mode="json") for item in sorted(records, key=lambda r: r.id)]
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self.path.write_text(text, encoding="utf-8", newline="")

    def _replace(self, record: FormulaRecord) -> FormulaRecord:
        records = [r for r in self.load() if r.id != record.id]
        records.append(record)
        self.save(records)
        return record

    def add(self, expression: str, *, record_id: str, source: str | None = None, line: int | None = None, tags: list[str] | None = None, citation: Citation | None = None) -> FormulaRecord:
        fp = fingerprint(expression)
        return self._replace(FormulaRecord(id=record_id, expression=expression, canonical_hash=fp.sha256, canonical=fp.canonical, kind="expression", source=source, line=line, tags=tags or [], citation=citation))

    def add_equation(self, lhs: str, rhs: str, *, record_id: str, source: str | None = None, line: int | None = None, tags: list[str] | None = None, citation: Citation | None = None) -> FormulaRecord:
        canonical, digest = _relation_fingerprint(lhs, rhs)
        return self._replace(FormulaRecord(id=record_id, expression=f"{lhs} = {rhs}", canonical_hash=digest, canonical=canonical, kind="equation", lhs=lhs, rhs=rhs, source=source, line=line, tags=tags or [], citation=citation))

    def add_ingestion(
        self,
        ingestion: PaperIngestionResult,
        *,
        tags: list[str] | None = None,
        skipped: list[str] | None = None,
        citation: Citation | None = None,
    ) -> list[FormulaRecord]:
        """Index an ingested document, isolating per-record failures.

        A single malformed equation used to raise out of the loop. Because each
        record was saved as it was added, that left the first *k* records on
        disk and silently dropped the rest -- a partial index that looks
        complete. Records are now built in memory, failures are collected into
        ``skipped``, and the index is written once at the end.
        """
        report: list[str] = [] if skipped is None else skipped
        existing = {r.id: r for r in self.load()}
        added: list[FormulaRecord] = []
        for item in ingestion.equations:
            equation = item.equation
            # One citation per document, narrowed to where each equation sits.
            # A paper-level reference that cannot say *where* in the paper is
            # the thing this feature exists to stop producing.
            entry_citation = None
            if citation is not None:
                entry_citation = citation.model_copy(
                    update={"section": item.section, "line": item.line}
                )
            if equation.parse_error is not None:
                report.append(f"{item.equation_id[:12]}: parse error: {equation.parse_error}")
                continue
            try:
                exact = _exact_expression(equation)
                if (
                    equation.kind.value == "equation"
                    and equation.lhs is not None
                    and equation.rhs is not None
                ):
                    if isinstance(exact, sp.Equality):
                        left, right = exact.lhs, exact.rhs
                    else:
                        left, right = equation.lhs, equation.rhs
                    canonical, digest = _relation_fingerprint(left, right)
                    record = FormulaRecord(
                        id=item.equation_id, expression=f"{equation.lhs} = {equation.rhs}",
                        canonical_hash=digest, canonical=canonical, kind="equation",
                        lhs=equation.lhs, rhs=equation.rhs, source=item.source,
                        line=item.line, tags=tags or [], citation=entry_citation,
                    )
                else:
                    fp = fingerprint(exact if exact is not None else equation.normalized)
                    record = FormulaRecord(
                        id=item.equation_id, expression=equation.normalized,
                        canonical_hash=fp.sha256, canonical=fp.canonical, kind="expression",
                        source=item.source, line=item.line, tags=tags or [],
                        citation=entry_citation,
                    )
            except Exception as exc:
                report.append(
                    f"{item.equation_id[:12]}: could not fingerprint "
                    f"({type(exc).__name__}: {str(exc).splitlines()[0][:100]})"
                )
                continue
            added.append(record)
        if added:
            # Drop what this document no longer says.
            #
            # Record ids are content hashes, so CORRECTING a formula does not
            # overwrite its record -- it files a second one beside the first,
            # and the wrong version stays indexed forever under its own id.
            # The functional equation was in here twice, once with the
            # `sin(pi*s/2)` factor and once without, and a structural search
            # would have returned the false one as indexed knowledge.
            #
            # The document is the truth and the index is a projection of it, so
            # a record sourced from this document that this ingestion did not
            # produce is stale by definition. Only sources this run actually
            # indexed are pruned: a document that suddenly parses to nothing
            # should raise a parse error, not silently empty its own section of
            # the index.
            fresh = {record.id for record in added}
            sources = {record.source for record in added if record.source}
            superseded = [
                record
                for record in existing.values()
                if record.source in sources and record.id not in fresh
            ]
            for record in superseded:
                report.append(
                    f"{record.id[:12]}: superseded, removed "
                    f"({record.source}:{record.line}) {record.expression[:60]}"
                )
                del existing[record.id]
            for record in added:
                existing[record.id] = record
            self.save(list(existing.values()))
        return added

    def add_knowledge(self, items, *, skipped: list[str] | None = None) -> list[FormulaRecord]:
        """Index the formulas durable memory *declares*, with their citations.

        Reads `KnowledgeItem.formulas` and not the prose statement. The
        extractor deliberately only recognises delimited mathematics, and
        loosening it to scrape any prose containing an `=` would fill the index
        with fragments of sentences -- which is worse than an empty index,
        because a structural match against a sentence fragment looks like a
        result.

        Note that the shipped durable memory declares no formulas today, so this
        indexes nothing until those fields are populated. Finding nothing
        because there is nothing is a different state from finding nothing
        because the reader is broken, and the returned list distinguishes them
        for a caller that reports the count.
        """
        report: list[str] = [] if skipped is None else skipped
        existing = {r.id: r for r in self.load()}
        added: list[FormulaRecord] = []
        for item in items:
            for position, formula in enumerate(item.formulas):
                record_id = hashlib.sha256(
                    f"knowledge:{item.id}:{position}:{formula}".encode()
                ).hexdigest()
                citation = knowledge_citation(item, formula_index=position)
                try:
                    fp = fingerprint(formula)
                except Exception as exc:
                    report.append(
                        f"{item.id}[{position}]: could not fingerprint "
                        f"({type(exc).__name__}: {str(exc).splitlines()[0][:100]})"
                    )
                    continue
                added.append(
                    FormulaRecord(
                        id=record_id,
                        expression=formula,
                        canonical_hash=fp.sha256,
                        canonical=fp.canonical,
                        kind="expression",
                        source=f"knowledge:{item.id}",
                        tags=[f"knowledge:{item.status.value}"],
                        citation=citation,
                    )
                )
        if added:
            for record in added:
                existing[record.id] = record
            self.save(list(existing.values()))
        return added

    def query(self, expression: str, *, limit: int = 10) -> list[FormulaMatch]:
        fp = fingerprint(expression)
        wanted = _tokens(expression)
        out: list[FormulaMatch] = []
        for record in self.load():
            if record.kind != "expression":
                continue
            try:
                theirs = _tokens(record.expression)
            except Exception:
                theirs = set()
            union = wanted | theirs
            shared = wanted & theirs
            score = 1.0 if not union else len(shared) / len(union)
            exact = record.canonical_hash == fp.sha256
            if exact:
                score = 1.0
            out.append(FormulaMatch(record=record, exact=exact, structural_score=score, shared_tokens=sorted(shared)))
        out.sort(key=lambda m: (not m.exact, -m.structural_score, m.record.id))
        return out[:limit]

    def query_equation(self, lhs: str, rhs: str, *, limit: int = 10) -> list[FormulaMatch]:
        _, digest = _relation_fingerprint(lhs, rhs)
        wanted = _tokens(lhs) | _tokens(rhs)
        out: list[FormulaMatch] = []
        for record in self.load():
            if record.kind != "equation" or record.lhs is None or record.rhs is None:
                continue
            theirs = _tokens(record.lhs) | _tokens(record.rhs)
            union = wanted | theirs
            shared = wanted & theirs
            score = 1.0 if not union else len(shared) / len(union)
            exact = record.canonical_hash == digest
            if exact:
                score = 1.0
            out.append(FormulaMatch(record=record, exact=exact, structural_score=score, shared_tokens=sorted(shared)))
        out.sort(key=lambda m: (not m.exact, -m.structural_score, m.record.id))
        return out[:limit]
