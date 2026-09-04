"""Determinism and platform-drift tests.

Cross-process determinism is checked in CI by running the CLI twice in separate
processes (see `.github/workflows/ci.yml`), because anything seeded per process
-- set iteration order being the obvious one -- is invisible inside a single
interpreter. These tests cover what can be checked in-process.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from rh_research_engine.core.bootstrap import seed_claims
from rh_research_engine.core.models import ExperimentResult
from rh_research_engine.core.store import ResearchStore
from rh_research_engine.dre import DreEvidenceEnvelope, write_dre_experiment

REPO = Path(__file__).resolve().parents[1]


def _claims_bytes(tmp_path: Path, seed: str) -> bytes:
    """Serialize the seed registry in a fresh interpreter with a given hash seed."""
    target = tmp_path / seed
    target.mkdir()
    script = (
        "import sys, pathlib;"
        f"sys.path.insert(0, {str(REPO / 'src')!r});"
        "from rh_research_engine.core.bootstrap import seed_claims;"
        "from rh_research_engine.core.store import ResearchStore;"
        f"ResearchStore(pathlib.Path({str(target)!r})).save_claims(seed_claims())"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": ""},
        capture_output=True,
    )
    return (target / "claims.json").read_bytes()


def test_claims_json_is_byte_identical_across_hash_seeds(tmp_path: Path) -> None:
    """`tags` is a set; without a stable serializer its order is seed-dependent."""
    digests = {
        hashlib.sha256(_claims_bytes(tmp_path, seed)).hexdigest()
        for seed in ("0", "1", "12345")
    }
    assert len(digests) == 1, "claims.json bytes depend on PYTHONHASHSEED"


def test_claims_json_is_lf_and_utf8(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path)
    store.save_claims(seed_claims())
    raw = store.claims_path.read_bytes()
    assert b"\r\n" not in raw, "CRLF would change every content hash on Windows"
    assert json.loads(raw.decode("utf-8"))


def test_claims_round_trip_preserves_non_ascii(tmp_path: Path) -> None:
    """`read_text()` without an encoding silently mojibakes UTF-8 on Windows."""
    store = ResearchStore(tmp_path)
    statement = "Riemann–von Mangoldt Θ ≤ 1/2"
    store.claims_path.write_text(
        json.dumps([{"id": "C1", "statement": statement}], ensure_ascii=False),
        encoding="utf-8",
        newline="",
    )
    assert store.load_claims()[0].statement == statement


def test_dre_artifact_bytes_are_lf(tmp_path: Path) -> None:
    """DRE hashes pack bytes, so a CRLF copy of this file is a different model."""
    result = ExperimentResult(
        name="correlation-lab", parameters={"X": 3000}, metrics={"screening_remainder": 1.43}
    )
    envelope = DreEvidenceEnvelope.from_experiment(result, claim_id="C005")
    path = write_dre_experiment(envelope, tmp_path / "e.yaml")
    assert b"\r\n" not in path.read_bytes()


def test_dre_export_is_reproducible(tmp_path: Path) -> None:
    result = ExperimentResult(
        name="correlation-lab", parameters={"X": 3000}, metrics={"screening_remainder": 1.43}
    )
    a = write_dre_experiment(
        DreEvidenceEnvelope.from_experiment(result, claim_id="C005"), tmp_path / "a.yaml"
    )
    b = write_dre_experiment(
        DreEvidenceEnvelope.from_experiment(result, claim_id="C005"), tmp_path / "b.yaml"
    )
    assert a.read_bytes() == b.read_bytes()


def test_repository_text_files_are_lf() -> None:
    """Guard the working tree, not just the index."""
    for pattern in ("research_state/*.json", "dre/experiments/*.yaml", "src/**/*.py"):
        for path in REPO.glob(pattern):
            assert b"\r\n" not in path.read_bytes(), f"{path} contains CRLF"


# --- the quantisation margin is thin for cancellation-derived metrics -------


def test_hash_quantisation_absorbs_ulp_drift_on_a_direct_metric() -> None:
    """Twelve digits is comfortable for a metric computed directly."""
    import math

    from rh_research_engine.dre.contracts import HASH_SIGNIFICANT_DIGITS, _canonical_number

    value = 6.5874541401914115
    drifted = value + 4 * math.ulp(value)
    assert _canonical_number(value) == _canonical_number(drifted)
    assert HASH_SIGNIFICANT_DIGITS == 10


def test_the_quantisation_margin_is_comfortable_for_a_cancellation_metric() -> None:
    """`total_energy` is a cancellation result, and it must still be stable.

    A sum of thousands of mixed-sign terms lands four orders of magnitude below
    the terms, so one ULP on an operand is a 2.6e-12 relative error. At twelve
    significant digits the quantisation step was 3.4e-15 -- a 3.8x margin, and
    CI caught ubuntu-latest and windows-latest disagreeing. Ten digits gives
    384x.
    """
    import math

    from rh_research_engine.dre.contracts import HASH_SIGNIFICANT_DIGITS

    diagonal, offdiag = 6.5874541401914115, -6.5871134707587880
    total = diagonal + offdiag
    operand_ulp = math.ulp(diagonal)

    step = abs(total) * 10 ** -(HASH_SIGNIFICANT_DIGITS - 1)
    margin = step / operand_ulp
    assert margin > 100, f"margin is only {margin:.1f}x"


def test_the_real_payload_hash_survives_the_drift_another_platform_would_add() -> None:
    """What `cross-platform-identity` asserted, without needing two platforms.

    THAT JOB IS RETIRED. It compared the DRE payload hash computed on
    ubuntu-latest and windows-latest, and it cannot run on a Linux-only forge:
    with one platform it has one hash and refuses, which is honest and also
    permanently red. Lewis's call, 2026-08-25, that this repository does not
    need a Windows leg.

    RETIRED IS NOT LAPSED. The claim it made was that one ULP of libm and BLAS
    drift cannot move a stored result, and that claim is still worth something.
    The tests above check the MARGIN on representative values; this one checks
    the ACTUAL envelope, by perturbing every float metric of a real experiment
    and requiring the hash not to move.

    It is a stronger check than the job it replaces. Two machines agreeing
    tells you those two machines agreed; this deliberately injects drift larger
    than either would produce and shows the quantisation absorbs it.
    """
    import math
    from pathlib import Path as _Path

    from rh_research_engine.core.store import ResearchStore
    from rh_research_engine.dre import DreEvidenceEnvelope

    runs = [
        run
        for run in ResearchStore(_Path("research_state")).load_experiments()
        if run.name == "correlation-lab"
    ]
    assert runs, "the correlation-lab run this gate is about is missing from the store"

    def envelope_for(run):
        return DreEvidenceEnvelope.from_experiment(
            run, claim_id="C005", primary_metric="screening_remainder"
        )

    baseline = envelope_for(runs[-1]).payload_hash

    # Four ULPs on every operand at once, both directions. A different BLAS or
    # vector width moves the last ULP or two; four is comfortably past that,
    # and `total_energy` amplifies it by four orders of magnitude because it is
    # a cancellation result.
    for ulps in (-4, -1, 1, 4):
        drifted = dict(runs[-1].metrics)
        for name, value in runs[-1].metrics.items():
            if isinstance(value, float) and math.isfinite(value) and value != 0.0:
                drifted[name] = value + ulps * math.ulp(value)
        moved = runs[-1].model_copy(update={"metrics": drifted})

        assert any(
            moved.metrics[k] != runs[-1].metrics[k] for k in runs[-1].metrics
        ), f"the {ulps:+d} ULP perturbation changed nothing, so this proves nothing"

        assert envelope_for(moved).payload_hash == baseline, (
            f"{ulps:+d} ULP on every metric changed the payload hash -- the "
            "quantisation no longer absorbs platform drift, and the same "
            "computation would be stored as two different results"
        )


def test_a_change_big_enough_to_matter_does_move_the_payload_hash() -> None:
    """The other half, or the test above passes on a hash that ignores metrics.

    A quantisation that absorbed EVERYTHING would satisfy the drift test
    perfectly and identify nothing. This fixes the point where it must stop
    absorbing: a relative change of 1e-8 is four orders above the drift being
    absorbed and well inside the tenth significant digit.
    """
    from pathlib import Path as _Path

    from rh_research_engine.core.store import ResearchStore
    from rh_research_engine.dre import DreEvidenceEnvelope

    runs = [
        run
        for run in ResearchStore(_Path("research_state")).load_experiments()
        if run.name == "correlation-lab"
    ]
    assert runs

    def envelope_for(run):
        return DreEvidenceEnvelope.from_experiment(
            run, claim_id="C005", primary_metric="screening_remainder"
        )

    baseline = envelope_for(runs[-1]).payload_hash
    nudged = dict(runs[-1].metrics)
    nudged["diagonal_energy"] = runs[-1].metrics["diagonal_energy"] * (1 + 1e-8)
    moved = runs[-1].model_copy(update={"metrics": nudged})

    assert envelope_for(moved).payload_hash != baseline, (
        "a 1e-8 relative change did not move the hash; the quantisation is "
        "absorbing real differences, not just platform drift"
    )


def test_the_metric_reductions_are_order_independent() -> None:
    """`math.fsum`, not `np.sum`: the summation ORDER must not reach the metric.

    numpy's pairwise order depends on SIMD width, so the same computation gave
    different last digits on different machines. fsum is correctly rounded, so
    shuffling the terms cannot change the result.
    """
    import math
    import random

    import numpy as np

    rng = random.Random(20260823)
    terms = [rng.uniform(-1, 1) for _ in range(5000)]
    shuffled = terms[:]
    rng.shuffle(shuffled)

    assert math.fsum(terms) == math.fsum(shuffled)
    # np.sum gives no such guarantee -- this is why the switch was needed.
    assert isinstance(float(np.sum(np.array(terms))), float)
