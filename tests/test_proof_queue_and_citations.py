"""The formal-proof queue, formula-index citations, and durable-memory indexing.

Three features that meet at one boundary: a formula can be indexed, cited, and
handed to Lean without any of that making it true. The tests below spend most of
their effort on that boundary rather than on the happy paths, because the happy
paths are the ones that look right when they are wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pydantic
import pytest

from rh_research_engine.contracts.artifacts import FormalizationReport
from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.core.knowledge import KnowledgeItem, KnowledgeStatus
from rh_research_engine.symbolic import (
    Citation,
    FormulaIndex,
    ProofQueueVerdict,
    SourceKind,
    build_proof_queue,
    ingest_text,
    knowledge_citation,
)
from rh_research_engine.symbolic.proof_queue import REFUSED_VERDICTS


@pytest.fixture
def index(tmp_path: Path) -> FormulaIndex:
    return FormulaIndex(tmp_path / "formula_index.json")


def _paper_citation(**overrides) -> Citation:
    fields = dict(
        source_kind=SourceKind.PAPER,
        source_id="notes.md",
        identifier="arXiv:2601.00001",
        title="A Short Note",
        authors=["A. Author"],
        theorem_label="Lemma 1",
    )
    fields.update(overrides)
    return Citation(**fields)


# ---------------------------------------------------------------------------
# 6 — citations carry provenance and confer nothing
# ---------------------------------------------------------------------------


def test_a_citation_carries_no_epistemic_status():
    """Appearing in a paper is not being proved, and not even being known.

    A paper can be wrong, withdrawn, or misread by whoever indexed it, and a
    formula transcribed out of one loses whatever hypotheses the surrounding
    text carried. So there is no status field here for a citation to raise.
    """
    fields = set(Citation.model_fields)
    for forbidden in ("status", "epistemic_status", "confidence", "verdict", "proved"):
        assert forbidden not in fields
    from rh_research_engine.symbolic.formula_index import FormulaRecord

    assert not ({"status", "epistemic_status", "confidence"} & set(FormulaRecord.model_fields))


def test_a_citation_says_when_it_cannot_pin_the_theorem():
    """A reference that cannot be followed back should not render as a tidy one."""
    complete = _paper_citation().describe()
    assert "Lemma 1" in complete
    assert "no theorem label" not in complete

    vague = _paper_citation(theorem_label=None).describe()
    assert "no theorem label" in vague

    bare = Citation(source_kind=SourceKind.MANUAL, source_id="scratch").describe()
    assert "no citation detail recorded" in bare


def test_citation_identity_is_content_addressed():
    assert _paper_citation().citation_hash() == _paper_citation().citation_hash()
    assert (
        _paper_citation().citation_hash()
        != _paper_citation(theorem_label="Theorem 2").citation_hash()
    )


def test_a_citation_may_not_carry_unknown_fields():
    with pytest.raises(pydantic.ValidationError):
        Citation(source_kind=SourceKind.PAPER, source_id="p", proved=True)


def test_indexing_a_document_narrows_the_citation_to_each_equation(index):
    """A paper-level reference that cannot say *where* is what this replaces."""
    text = "# Notes\n\n## Algebra\n\nThe square expands: $(x+1)^2 = x^2 + 2x + 1$.\n"
    added = index.add_ingestion(ingest_text(text, source="notes.md"), citation=_paper_citation())
    assert len(added) == 1
    citation = added[0].citation
    assert citation is not None
    assert citation.section == "Algebra"
    assert citation.line is not None
    assert citation.identifier == "arXiv:2601.00001"


def test_a_match_reports_its_provenance_or_says_there_is_none(index):
    index.add_equation("(x+1)**2", "x**2+2*x+1", record_id="cited", citation=_paper_citation())
    index.add_equation("(y+1)**2", "y**2+2*y+1", record_id="uncited")

    matches = {m.record.id: m for m in index.query_equation("(x+1)**2", "x**2+2*x+1")}
    assert "Lemma 1" in matches["cited"].provenance
    assert matches["uncited"].provenance == "no source recorded"


def test_citations_survive_a_round_trip_through_disk(index):
    index.add_equation("(x+1)**2", "x**2+2*x+1", record_id="r", citation=_paper_citation())
    reloaded = FormulaIndex(index.path).load()[0]
    assert reloaded.citation is not None
    assert reloaded.citation.citation_hash() == _paper_citation().citation_hash()


# ---------------------------------------------------------------------------
# 5 — durable-memory and ingestion indexing
# ---------------------------------------------------------------------------


def _knowledge_item(**overrides) -> KnowledgeItem:
    fields = dict(
        id="K999",
        title="A worked identity",
        status=KnowledgeStatus.EXACT_ALGEBRA,
        domain="algebra",
        statement="Under no extra hypotheses, the square expands.",
        formulas=["(x+1)**2", "x**2 - 1"],
    )
    fields.update(overrides)
    return KnowledgeItem(**fields)


def test_durable_memory_formulas_are_indexed_with_their_record_cited(index):
    added = index.add_knowledge([_knowledge_item()])
    assert len(added) == 2
    citation = added[0].citation
    assert citation is not None
    assert citation.source_kind is SourceKind.KNOWLEDGE
    assert citation.source_id == "K999"
    # The statement travels with the formula, so the record's conditions are not
    # left behind in durable memory.
    assert citation.statement == "Under no extra hypotheses, the square expands."
    assert any("exact_algebra" in note for note in citation.notes)
    assert added[0].tags == ["knowledge:exact_algebra"]


def test_the_conditions_of_a_conditional_record_travel_with_its_formula(index):
    """`conditional_on_RH_standard` is exactly where losing the statement hurts."""
    item = _knowledge_item(
        id="K998",
        status=KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD,
        statement="Under RH, Theta <= 1/2.",
        formulas=["x/2"],
    )
    [record] = index.add_knowledge([item])
    assert record.citation is not None
    assert "Under RH" in record.citation.statement
    assert record.tags == ["knowledge:conditional_on_RH_standard"]


def test_indexing_durable_memory_is_idempotent(index):
    first = index.add_knowledge([_knowledge_item()])
    second = index.add_knowledge([_knowledge_item()])
    assert [r.id for r in first] == [r.id for r in second]
    assert len(index.load()) == 2


def test_an_unparseable_declared_formula_is_reported_not_dropped(index):
    skipped: list[str] = []
    added = index.add_knowledge(
        [_knowledge_item(formulas=["(x+1)**2", "))not algebra(("])], skipped=skipped
    )
    assert len(added) == 1
    assert len(skipped) == 1
    assert "K999[1]" in skipped[0]


def test_prose_statements_are_not_scraped(index):
    """Only the `formulas` field is read, and this pins that.

    The shipped durable memory declares no formulas while its statements are
    full of prose like `z=s-1/2`. Loosening the reader to scrape those would
    fill the index with sentence fragments, and a structural match against a
    sentence fragment looks like a result.
    """
    item = _knowledge_item(statement="Use z=s-1/2 and Xi(z)=xi(1/2+z).", formulas=[])
    assert index.add_knowledge([item]) == []
    assert index.load() == []


def test_the_shipped_durable_memory_declares_no_formulas():
    """Recorded so "indexed 0" is known to be a data gap, not a broken reader."""
    from rh_research_engine.core.knowledge import KnowledgeBase

    assert sum(len(item.formulas) for item in KnowledgeBase().load()) == 0


# ---------------------------------------------------------------------------
# 4 — the formal-proof queue
# ---------------------------------------------------------------------------


def _seeded(index: FormulaIndex) -> FormulaIndex:
    index.add_equation("(x+1)**2", "x**2+2*x+1", record_id="true-identity")
    index.add_equation("x**2", "x**3", record_id="false-identity")
    index.add_equation("sin(x)", "sin(x)", record_id="transcendental")
    index.add("x+1", record_id="bare-expression")
    return index


@pytest.mark.parametrize(
    "record_id,verdict",
    [
        ("true-identity", ProofQueueVerdict.EXPORT_READY),
        ("false-identity", ProofQueueVerdict.NOT_AN_IDENTITY),
        ("transcendental", ProofQueueVerdict.UNSUPPORTED_FRAGMENT),
        ("bare-expression", ProofQueueVerdict.NOT_AN_EQUATION),
    ],
)
def test_the_queue_sorts_each_kind_of_formula(index, record_id, verdict):
    queue = build_proof_queue(_seeded(index).load())
    entry = next(e for e in queue.entries if e.record_id == record_id)
    assert entry.verdict is verdict


def test_a_refused_formula_keeps_its_place_in_the_queue(index):
    """"400 formulas, 3 provable" is the useful number; filtering hides it."""
    queue = build_proof_queue(_seeded(index).load())
    assert len(queue.entries) == 4
    assert len(queue.ready) == 1
    assert len(queue.refused) == 3
    assert "1 of 4 ready to export" in queue.summary()
    assert "not verified" in queue.summary()


def test_every_refusal_verdict_leaves_an_open_obligation():
    assert ProofQueueVerdict.EXPORT_READY not in REFUSED_VERDICTS
    assert len(REFUSED_VERDICTS) == len(ProofQueueVerdict) - 1


def test_a_non_identity_is_refused_rather_than_exported(index):
    """`ring` discharges identities; this is not one."""
    queue = build_proof_queue(_seeded(index).load())
    entry = next(e for e in queue.entries if e.record_id == "false-identity")
    assert entry.lean is None
    assert entry.reason is not None


def test_export_ready_means_emitted_and_explicitly_not_verified(index):
    """The whole point of the queue, and the easiest thing to get wrong."""
    queue = build_proof_queue(_seeded(index).load())
    source = queue.ready[0].lean_source()
    assert "NOT verified" in source
    assert "has not been compiled" in source
    assert ":= by\n  ring" in source


def test_emitted_lean_carries_its_provenance(index):
    index.add_equation(
        "(x+1)**2", "x**2+2*x+1", record_id="cited", citation=_paper_citation()
    )
    queue = build_proof_queue(index.load())
    source = queue.ready[0].lean_source()
    assert "Lemma 1" in source
    assert "cited" in source


def test_emitted_lean_says_so_when_no_source_is_recorded(index):
    queue = build_proof_queue(_seeded(index).load())
    assert "-- source: none recorded" in queue.ready[0].lean_source()


def test_a_formalization_report_from_an_export_is_not_fully_formalized(index):
    """Emitting Lean is a step toward a proof, not one.

    `remaining_obligations` names the compilation that has not happened, which
    keeps `fully_formalized` false and the contract validator refusing
    FORMALLY_VERIFIED.
    """
    queue = build_proof_queue(_seeded(index).load())
    report = queue.ready[0].to_formalization_report()
    assert report.fully_formalized is False
    assert report.epistemic_status is Confidence.SYMBOLIC_DERIVED
    assert report.epistemic_status not in RIGOROUS
    assert any("compile" in item for item in report.remaining_obligations)


def test_a_refusal_records_why_in_its_obligations(index):
    queue = build_proof_queue(_seeded(index).load())
    entry = next(e for e in queue.entries if e.record_id == "false-identity")
    report = entry.to_formalization_report()
    assert report.steps_formalized == []
    assert any("not_an_identity" in item for item in report.remaining_obligations)


def test_an_export_cannot_be_relabelled_as_formally_verified(index):
    """Two independent guards refuse it, and both are worth pinning.

    The queue is a worker, so the first refusal is that a worker may not assert
    `formally_verified` at all. Granting it `external-verifier` gets past that
    and straight into the second: the report still has an open obligation, so it
    is not fully formalized. Either alone would be enough; having both means a
    change to one does not quietly open the path.
    """
    queue = build_proof_queue(_seeded(index).load())
    payload = queue.ready[0].to_formalization_report().model_dump()

    with pytest.raises(pydantic.ValidationError, match="may not assert"):
        FormalizationReport.model_validate(
            payload | {"epistemic_status": "formally_verified"}
        )

    with pytest.raises(pydantic.ValidationError, match="compiles with an axiom"):
        FormalizationReport.model_validate(
            payload
            | {
                "epistemic_status": "formally_verified",
                "created_by": "external-verifier",
            }
        )


def test_only_export_ready_entries_write_files(index, tmp_path):
    queue = build_proof_queue(_seeded(index).load())
    written = queue.write(tmp_path / "lean")
    assert len(written) == 1
    assert written[0].name.endswith(".lean")
    assert written[0].read_bytes().count(b"\r") == 0


def test_writing_the_queue_is_deterministic(index, tmp_path):
    queue = build_proof_queue(_seeded(index).load())
    first = queue.write(tmp_path / "a")[0].read_bytes()
    second = queue.write(tmp_path / "b")[0].read_bytes()
    assert first == second


def test_an_empty_index_yields_an_honest_empty_queue():
    queue = build_proof_queue([])
    assert queue.entries == []
    assert "no indexed equations" in queue.summary()
    assert queue.write(Path.cwd() / "unused") == []


def test_the_queue_serializes_without_losing_its_verdicts(index):
    queue = build_proof_queue(_seeded(index).load())
    payload = json.loads(json.dumps(queue.model_dump(mode="json")))
    verdicts = {e["record_id"]: e["verdict"] for e in payload["entries"]}
    assert verdicts["true-identity"] == "export_ready"
    assert verdicts["false-identity"] == "not_an_identity"


def test_knowledge_citation_names_the_formula_position():
    citation = knowledge_citation(_knowledge_item(), formula_index=1)
    assert citation.theorem_label == "formula 1"
    assert citation.source_kind is SourceKind.KNOWLEDGE


def test_a_constraint_is_not_reported_as_refuted(index):
    """`Theta = 1/2` is RH. The engine must not appear to have disproved it.

    A constraint on a symbol and a wrong identity both fail
    `expand(lhs - rhs) == 0`, and the exporter cannot tell them apart. So the
    verdict says only what is known -- this is not an identity, so `ring` cannot
    discharge it -- and never that it is false. The earlier name,
    `not_symbolically_true`, put "Theta = 1/2 -- not symbolically true" in a
    generated report, which reads as a refutation of the Riemann Hypothesis.
    """
    index.add_equation("Theta", "1/2", record_id="rh")
    entry = build_proof_queue(index.load()).entries[0]

    assert entry.verdict is ProofQueueVerdict.NOT_AN_IDENTITY
    assert entry.verdict.value == "not_an_identity"

    wording = (entry.verdict.value + " " + (entry.reason or "")).lower()
    for overclaim in ("false", "refuted", "disproved", "untrue", "wrong"):
        assert overclaim not in wording, f"verdict wording claims {overclaim!r}"

    report = entry.to_formalization_report()
    assert report.epistemic_status is not Confidence.REFUTED
