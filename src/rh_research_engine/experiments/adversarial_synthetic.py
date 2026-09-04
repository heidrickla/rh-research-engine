from __future__ import annotations

import math
from collections.abc import Callable

import mpmath as mp
from pydantic import BaseModel, Field, model_validator

from ..core.models import EvidenceClass, ExperimentResult


class SyntheticZero(BaseModel):
    beta: float
    gamma: float
    multiplicity: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _valid_zero(self):
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in the critical strip")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        return self

    @property
    def off_line_deviation(self) -> float:
        return abs(self.beta - 0.5)


class SyntheticSystem(BaseModel):
    critical_line_zeros: list[float] = Field(default_factory=list)
    off_line_zeros: list[SyntheticZero] = Field(default_factory=list)
    q: float = 20.0

    @model_validator(mode="after")
    def _valid_system(self):
        if self.q <= 0:
            raise ValueError("q must be positive")
        return self

    @property
    def expected_rh(self) -> bool:
        return not self.off_line_zeros

    def symmetric_zeros(self) -> list[SyntheticZero]:
        zeros: dict[tuple[float, float], SyntheticZero] = {}
        for gamma in self.critical_line_zeros:
            z = SyntheticZero(beta=0.5, gamma=abs(gamma))
            zeros[(z.beta, z.gamma)] = z
        for zero in self.off_line_zeros:
            for beta in {zero.beta, 1.0 - zero.beta}:
                z = SyntheticZero(beta=beta, gamma=zero.gamma, multiplicity=zero.multiplicity)
                zeros[(round(z.beta, 15), round(z.gamma, 15))] = z
        return [zeros[key] for key in sorted(zeros)]

    def max_off_line_deviation(self) -> float:
        return max((zero.off_line_deviation for zero in self.symmetric_zeros()), default=0.0)

    def gamma_energy_slope(self) -> float:
        # Synthetic analogue of the Gamma-filter response: an off-line pair with
        # beta=1/2+eta contributes energy growth exp(2 eta t).
        return 2.0 * self.max_off_line_deviation()

    def functional_equation_symmetric(self) -> bool:
        zeros = {(round(z.beta, 12), round(z.gamma, 12)) for z in self.symmetric_zeros()}
        return all((round(1.0 - beta, 12), gamma) in zeros for beta, gamma in zeros)


class CriterionResult(BaseModel):
    criterion: str
    predicted_rh: bool
    expected_rh: bool
    false_positive: bool
    false_negative: bool


Criterion = Callable[[SyntheticSystem, float], bool]


def _critical_line_window(system: SyntheticSystem, tolerance: float) -> bool:
    return system.max_off_line_deviation() <= tolerance


def _gamma_energy_slope_threshold(system: SyntheticSystem, tolerance: float) -> bool:
    return system.gamma_energy_slope() <= 2.0 * tolerance


CRITERIA: dict[str, Criterion] = {
    "critical-line-window": _critical_line_window,
    "gamma-energy-slope-threshold": _gamma_energy_slope_threshold,
}


def parse_off_line_zeros(items: list[str]) -> list[SyntheticZero]:
    zeros: list[SyntheticZero] = []
    for item in items:
        beta_text, gamma_text, *rest = item.split(":")
        multiplicity = int(rest[0]) if rest else 1
        zeros.append(
            SyntheticZero(beta=float(beta_text), gamma=float(gamma_text), multiplicity=multiplicity)
        )
    return zeros


def _gamma_amplitude(system: SyntheticSystem) -> float:
    total = mp.mpf("0")
    for zero in system.symmetric_zeros():
        rho = zero.beta + 1j * zero.gamma
        total += zero.multiplicity * abs(mp.gamma(1 + rho / system.q))
    return float(total)


def evaluate_criteria(
    system: SyntheticSystem,
    *,
    tolerance: float = 0.0,
    criteria: list[str] | None = None,
) -> list[CriterionResult]:
    selected = criteria or sorted(CRITERIA)
    results: list[CriterionResult] = []
    for name in selected:
        if name not in CRITERIA:
            raise ValueError(f"unknown synthetic criterion: {name}")
        predicted = bool(CRITERIA[name](system, tolerance))
        results.append(
            CriterionResult(
                criterion=name,
                predicted_rh=predicted,
                expected_rh=system.expected_rh,
                false_positive=predicted and not system.expected_rh,
                false_negative=(not predicted) and system.expected_rh,
            )
        )
    return results


def run(
    *,
    critical_gamma: list[float] | None = None,
    off_line: list[SyntheticZero] | None = None,
    q: float = 20.0,
    tolerance: float = 0.0,
    criteria: list[str] | None = None,
) -> ExperimentResult:
    system = SyntheticSystem(
        critical_line_zeros=critical_gamma or [14.134725141734693],
        off_line_zeros=off_line or [],
        q=q,
    )
    results = evaluate_criteria(system, tolerance=tolerance, criteria=criteria)
    false_positives = sum(1 for item in results if item.false_positive)
    false_negatives = sum(1 for item in results if item.false_negative)
    symmetric = system.functional_equation_symmetric()
    max_eta = system.max_off_line_deviation()
    return ExperimentResult(
        name="synthetic-adversarial-counterexamples",
        parameters={
            "critical_gamma": [float(g) for g in system.critical_line_zeros],
            "off_line": [z.model_dump(mode="json") for z in system.off_line_zeros],
            "q": q,
            "tolerance": tolerance,
            "criteria": criteria or sorted(CRITERIA),
        },
        metrics={
            "expected_rh": system.expected_rh,
            "functional_equation_symmetric": symmetric,
            "zero_count_with_symmetry": len(system.symmetric_zeros()),
            "max_off_line_deviation": max_eta,
            "synthetic_gamma_energy_slope": 2.0 * max_eta,
            "gamma_filter_amplitude_proxy": _gamma_amplitude(system),
            "false_positive_count": false_positives,
            "false_negative_count": false_negatives,
            "criteria_count": len(results),
        },
        observations=[
            *(f"{r.criterion}: predicted_rh={r.predicted_rh}, expected_rh={r.expected_rh}" for r in results),
            "Synthetic systems enforce conjugation and functional-equation symmetry by construction.",
            "Synthetic adversarial evidence is non-rigorous and cannot promote mathematical claims.",
        ],
        evidence_class=EvidenceClass.HEURISTIC,
        method_family="python-synthetic-adversary",
        assumptions=[
            "zeta-like zeros are synthetic fixtures, not zeros of the Riemann zeta function",
            "candidate-criterion outcomes are falsification diagnostics only",
        ],
    )


def default_off_line_system(eta: float, gamma: float) -> list[SyntheticZero]:
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if math.isclose(eta, 0.0):
        return []
    return [SyntheticZero(beta=0.5 + eta, gamma=gamma)]
