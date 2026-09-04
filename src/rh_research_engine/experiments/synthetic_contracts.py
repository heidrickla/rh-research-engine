"""Express synthetic-adversary output as canonical contract artifacts.

Adaptation only: the mathematics in ``adversarial_synthetic`` is untouched. What
this adds is the declaration the contract layer requires -- which genuine zeta
properties a synthetic family satisfies and which it violates -- because results
from an undeclared model cannot be interpreted.

Nothing here can advance the frontier. ``Confidence.SYNTHETIC`` is not rigorous,
so ``assess`` denies advancement for every artifact this module produces, and
``CounterexampleArtifact`` separately refuses to be relabelled into a theorem.
"""

from __future__ import annotations

from ..contracts.artifacts import (
    CounterexampleArtifact,
    CounterexampleVerdict,
    ModelFamilySpec,
    SyntheticModel,
)
from ..contracts.epistemic import Confidence
from ..contracts.roles import Role
from .adversarial_synthetic import CriterionResult, SyntheticSystem

METHOD_FAMILY = "python-synthetic-adversary"

#: Properties every synthetic system here has by construction.
SATISFIED_PROPERTIES = (
    "conjugation symmetry",
    "functional-equation symmetry rho -> 1-rho",
    "discrete zero set",
)

#: Properties no synthetic system here has, ever. Stated so a result against one
#: of these models is read as what it is: evidence about the model.
VIOLATED_PROPERTIES = (
    "Euler product over primes",
    "analytic continuation of an actual Dirichlet series",
    "zeros are zeros of the Riemann zeta function",
    "Riemann-von Mangoldt zero counting",
)


def model_family_spec(*, artifact_id: str, method_version: str) -> ModelFamilySpec:
    """Declare the family's relationship to real zeta."""
    return ModelFamilySpec(
        artifact_id=artifact_id,
        created_by=METHOD_FAMILY,
        method_family=METHOD_FAMILY,
        method_version=method_version,
        family_name="zeta-like-synthetic",
        satisfied_properties=list(SATISFIED_PROPERTIES),
        violated_properties=list(VIOLATED_PROPERTIES),
        epistemic_status=Confidence.SYNTHETIC,
        mathematical_role=Role.CONSTRUCTION,
        assumptions=[
            "zeta-like zeros are synthetic fixtures, not zeros of the Riemann zeta function",
        ],
    )


def synthetic_model(
    system: SyntheticSystem, *, artifact_id: str, family_ref: str, method_version: str
) -> SyntheticModel:
    return SyntheticModel(
        artifact_id=artifact_id,
        created_by=METHOD_FAMILY,
        method_family=METHOD_FAMILY,
        method_version=method_version,
        family_ref=family_ref,
        instance_parameters={
            "critical_line_zeros": [float(g) for g in system.critical_line_zeros],
            "off_line_zeros": [z.model_dump(mode="json") for z in system.off_line_zeros],
            "q": system.q,
            "expected_rh": system.expected_rh,
            "max_off_line_deviation": system.max_off_line_deviation(),
        },
        epistemic_status=Confidence.SYNTHETIC,
        mathematical_role=Role.CONSTRUCTION,
        dependencies=[family_ref],
    )


def verdict_for(result: CriterionResult) -> CounterexampleVerdict:
    """Map one criterion outcome onto a contract verdict.

    A *correct* prediction on a single system is ``INCONCLUSIVE``, not
    ``SEPARATES_SYNTHETIC_CLASSES``. One data point cannot establish that a
    criterion distinguishes the classes; that needs a run across both, which is
    the property-separator engine's job rather than this adapter's.
    """
    if result.false_positive:
        return CounterexampleVerdict.FALSE_POSITIVE
    if result.false_negative:
        return CounterexampleVerdict.FALSE_NEGATIVE
    return CounterexampleVerdict.INCONCLUSIVE


def counterexample_artifacts(
    results: list[CriterionResult],
    *,
    model_ref: str,
    method_version: str,
    artifact_prefix: str = "cex",
) -> list[CounterexampleArtifact]:
    out: list[CounterexampleArtifact] = []
    for index, result in enumerate(results, start=1):
        verdict = verdict_for(result)
        out.append(
            CounterexampleArtifact(
                artifact_id=f"{artifact_prefix}:{result.criterion}:{index:03d}",
                created_by=METHOD_FAMILY,
                method_family=METHOD_FAMILY,
                method_version=method_version,
                criterion_ref=result.criterion,
                model_ref=model_ref,
                verdict=verdict,
                detail=(
                    f"predicted_rh={result.predicted_rh}, expected_rh={result.expected_rh}"
                ),
                epistemic_status=Confidence.SYNTHETIC,
                mathematical_role=Role.CRITERION,
                dependencies=[model_ref],
                assumptions=[
                    "outcome holds for a synthetic fixture, not for zeta",
                ],
            )
        )
    return out
