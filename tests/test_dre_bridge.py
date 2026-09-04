from pathlib import Path

from rh_research_engine.dre import ClaimEffect, DreEvidenceEnvelope, EvidenceClass
from rh_research_engine.dre.export import render_dre_experiment, write_dre_experiment


def _env(version: str = "0.4.0") -> DreEvidenceEnvelope:
    return DreEvidenceEnvelope(
        experiment_name="correlation-lab",
        claim_id="C005",
        claim_effect=ClaimEffect.SUPPORTS,
        evidence_class=EvidenceClass.NUMERICAL,
        method_family="python-numpy",
        worker_version=version,
        parameters={"X": 20000, "q": 4.0},
        metrics={"screening_remainder": 1.25},
        primary_metric_name="screening_remainder",
        primary_metric_value=1.25,
    )


def test_envelope_hash_is_deterministic() -> None:
    assert _env().result_hash == _env().result_hash


def test_same_method_version_is_one_independence_group() -> None:
    a = _env()
    b = _env()
    assert a.independence_group == b.independence_group
    assert _env("0.4.1").independence_group != a.independence_group


def test_export_uses_scaled_integer_and_no_raw_float_metric() -> None:
    text = render_dre_experiment(_env())
    assert "primary_metric_scaled" in text
    assert "1250000000" in text
    assert "independence_group: python-numpy:0.4.0" in text
    assert "predicate: evidence_class" in text
    assert "value: numerical" in text


def test_write_export(tmp_path: Path) -> None:
    out = write_dre_experiment(_env(), tmp_path / "e.yaml")
    assert out.exists()
    assert "pack: model-packs/riemann-research" in out.read_text(encoding="utf-8")
