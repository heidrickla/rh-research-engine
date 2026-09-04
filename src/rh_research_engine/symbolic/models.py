from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EquationKind(StrEnum):
    EXPRESSION = "expression"
    EQUATION = "equation"
    INEQUALITY = "inequality"
    UNKNOWN = "unknown"


class ExtractedEquation(BaseModel):
    source: str
    normalized: str
    kind: EquationKind
    lhs: str | None = None
    rhs: str | None = None
    sympy_srepr: str | None = None
    parse_error: str | None = None


class RewriteStep(BaseModel):
    rule_id: str
    description: str
    before: str
    after: str
    assumptions: list[str] = Field(default_factory=list)


class SimplificationResult(BaseModel):
    original: str
    simplified: str
    steps: list[RewriteStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class EquivalenceResult(BaseModel):
    equivalent: bool | None
    method: str
    left_canonical: str
    right_canonical: str
    assumptions: list[str] = Field(default_factory=list)
    detail: str | None = None


class Assumption(BaseModel):
    expression: str
    condition: str
    reason: str


class ProofStep(BaseModel):
    id: str
    statement: str
    status: str
    depends_on: list[str] = Field(default_factory=list)


class ProofGap(BaseModel):
    step_id: str
    statement: str
    reason: str
    blocking_dependencies: list[str] = Field(default_factory=list)


class TransformResult(BaseModel):
    transform: str
    input_expression: str
    output_expression: str
    conditions: list[str] = Field(default_factory=list)
    rule_id: str


class ResidueResult(BaseModel):
    expression: str
    variable: str
    pole: str
    residue: str


class AsymptoticResult(BaseModel):
    expression: str
    variable: str
    point: str
    leading: str | None = None
    limit: str | None = None
    error: str | None = None


class Fingerprint(BaseModel):
    canonical: str
    sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecompositionCandidate(BaseModel):
    kind: str
    original: str
    reconstruction: str
    parts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    verified: bool = False


class CountertermCandidate(BaseModel):
    name: str
    expression: str
    rationale: str
    status: str = "candidate"


class ConjectureMinimization(BaseModel):
    original: str
    goal: str
    weaker_target: str
    changed: bool
    rule_id: str | None = None
    rationale: list[str] = Field(default_factory=list)


class IngestedEquation(BaseModel):
    source: str
    line: int | None = None
    section: str | None = None
    equation: ExtractedEquation
    equation_id: str


class PaperIngestionResult(BaseModel):
    source: str
    equations: list[IngestedEquation] = Field(default_factory=list)
    count: int
