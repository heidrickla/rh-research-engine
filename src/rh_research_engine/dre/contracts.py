from __future__ import annotations

import json
import math
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ..core.models import (
    NON_DEDUCTIVE_CLASSES,
    WORKER_FORBIDDEN_CLASSES,
    EvidenceClass,
    ExperimentResult,
)

__all__ = [
    "ClaimEffect",
    "DreEvidenceEnvelope",
    "EvidenceClass",
    "WorkerClassError",
]

#: Significant decimal digits retained when a float enters a hash.
#:
#: Raw IEEE doubles are not reproducible across machines: the same reduction
#: differs in the last few ULPs between BLAS builds and vector widths. Rounding
#: before hashing absorbs that, so a hash identifies a *result* rather than a
#: machine.
#:
#: Ten, not twelve. Twelve was chosen against a 3.5e-14 relative drift measured
#: on a metric computed directly, and it does not survive one computed by
#: cancellation: `total_energy` is a sum of thousands of mixed-sign terms whose
#: result is four orders of magnitude smaller than the terms, so one ULP on an
#: operand is a 2.6e-12 relative error -- a 3.8x margin at twelve digits, and CI
#: duly caught ubuntu-latest and windows-latest disagreeing. Ten digits gives
#: 384x, and no research metric here carries a real tenth significant digit.
#:
#: The reductions that feed these metrics also use `math.fsum` now, which is
#: correctly rounded and order-independent, so the drift is removed at source
#: as well as absorbed here.
HASH_SIGNIFICANT_DIGITS = 10


class WorkerClassError(ValueError):
    """Raised when a worker tries to assert an evidence class it cannot earn."""


class ClaimEffect(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


def _canonical_number(value: float | int) -> str | int:
    """Render a number for hashing in a platform-independent way."""
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        return "0.0e+00"
    return f"{value:.{HASH_SIGNIFICANT_DIGITS - 1}e}"


def _canonical_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(mapping):
        value = mapping[key]
        out[key] = _canonical_number(value) if isinstance(value, (int, float)) else value
    return out


class DreEvidenceEnvelope(BaseModel):
    schema_version: str = "0.2.0"
    experiment_name: str
    claim_id: str
    claim_effect: ClaimEffect
    evidence_class: EvidenceClass
    method_family: str
    worker_version: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    primary_metric_name: str | None = None
    primary_metric_value: float | int | None = None
    primary_metric_scale: int = 1_000_000_000
    theta_upper: float | None = None
    bound_exponent: float | None = None
    rh_equivalent: bool = False
    counterexample_found: bool = False
    independently_verified: bool = False
    artifact_ref: str = ""

    @model_validator(mode="after")
    def _enforce_epistemic_boundaries(self):
        if self.evidence_class in WORKER_FORBIDDEN_CLASSES:
            raise WorkerClassError(
                f"a math worker may not assert evidence_class={self.evidence_class.value!r}; "
                "'proved' and 'known' require a formal checker or cited literature outside "
                "this package"
            )
        if self.evidence_class in NON_DEDUCTIVE_CLASSES and self.theta_upper is not None:
            raise ValueError(
                f"evidence_class={self.evidence_class.value!r} carries no deductive force, so it "
                "cannot assert theta_upper; a Theta bound requires a rigorous derivation"
            )
        if self.independently_verified and self.evidence_class in NON_DEDUCTIVE_CLASSES:
            raise ValueError(
                "independently_verified requires a corroborating verifier envelope from a "
                "distinct method family; it cannot be asserted for non-deductive evidence"
            )
        if self.theta_upper is not None and self.theta_upper < 0.5:
            raise ValueError(
                f"theta_upper={self.theta_upper} is below 1/2 and therefore impossible: zeta has "
                "zeros on the critical line, so Theta >= 1/2 unconditionally"
            )
        return self

    @classmethod
    def from_experiment(
        cls,
        result: ExperimentResult,
        *,
        claim_id: str,
        claim_effect: ClaimEffect = ClaimEffect.SUPPORTS,
        primary_metric_name: str | None = None,
        artifact_ref: str = "",
        **overrides: Any,
    ) -> DreEvidenceEnvelope:
        """Build an envelope whose provenance is taken from the experiment record.

        ``evidence_class``, ``method_family``, and ``worker_version`` are read
        from the result rather than accepted from the caller. Relabelling them
        at export time is how one numpy run became three independent witnesses.
        """
        for locked in ("evidence_class", "method_family", "worker_version"):
            if locked in overrides:
                raise WorkerClassError(
                    f"{locked!r} is worker-declared and cannot be overridden at export time"
                )
        primary_value = None
        if primary_metric_name is not None:
            raw = result.metrics.get(primary_metric_name)
            if raw is None:
                raise KeyError(f"metric not found: {primary_metric_name}")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"primary metric must be numeric: {primary_metric_name}")
            primary_value = raw
        return cls(
            experiment_name=result.name,
            claim_id=claim_id,
            claim_effect=claim_effect,
            evidence_class=result.evidence_class,
            method_family=result.method_family,
            worker_version=result.worker_version,
            parameters=result.parameters,
            metrics=result.metrics,
            observations=result.observations,
            assumptions=result.assumptions,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_value,
            artifact_ref=artifact_ref,
            **overrides,
        )

    def canonical_payload(self) -> bytes:
        """Bytes identifying *what was computed*, excluding who computed it.

        Provenance labels are deliberately absent: two envelopes describing the
        same numbers must collide even if one is relabelled, otherwise a
        downstream 'same hash means same evidence' check can be defeated by
        editing a string.
        """
        payload = {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "claim_id": self.claim_id,
            "claim_effect": self.claim_effect.value,
            "parameters": _canonical_mapping(self.parameters),
            "metrics": _canonical_mapping(self.metrics),
            "assumptions": sorted(self.assumptions),
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_scaled": self.scaled_primary_metric(),
            "theta_upper_ppb": self.theta_upper_ppb(),
            "bound_exponent_ppb": self.bound_exponent_ppb(),
            "rh_equivalent": self.rh_equivalent,
            "counterexample_found": self.counterexample_found,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def canonical_provenance(self) -> bytes:
        payload = {
            "evidence_class": self.evidence_class.value,
            "method_family": self.method_family,
            "worker_version": self.worker_version,
            "independently_verified": self.independently_verified,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def payload_hash(self) -> str:
        """Identifies the result. Dedup on this."""
        return sha256(self.canonical_payload()).hexdigest()

    @property
    def provenance_hash(self) -> str:
        """Identifies the producer. Never dedup on this."""
        return sha256(self.canonical_provenance()).hexdigest()

    @property
    def result_hash(self) -> str:
        return self.payload_hash

    @property
    def independence_group(self) -> str:
        # Repeats from one implementation/method are deliberately one voice.
        return f"{self.method_family}:{self.worker_version}"

    @property
    def assumption_count(self) -> int:
        return len(self.assumptions)

    def scaled_primary_metric(self) -> int | None:
        if self.primary_metric_value is None:
            return None
        return int(round(float(self.primary_metric_value) * self.primary_metric_scale))

    def theta_upper_ppb(self) -> int | None:
        if self.theta_upper is None:
            return None
        return int(round(self.theta_upper * 1_000_000_000))

    def bound_exponent_ppb(self) -> int | None:
        if self.bound_exponent is None:
            return None
        return int(round(self.bound_exponent * 1_000_000_000))
