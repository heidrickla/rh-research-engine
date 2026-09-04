from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class ScientificInteger(BaseModel):
    kind: Literal["scientific"] = "scientific"
    mantissa: int
    decimal_exponent: int = 0

    @model_validator(mode="after")
    def normalize(self):
        m = self.mantissa
        e = self.decimal_exponent
        if m == 0:
            self.decimal_exponent = 0
            return self
        while m % 10 == 0:
            m //= 10
            e += 1
        self.mantissa = m
        self.decimal_exponent = e
        return self

    def as_fraction(self) -> Fraction:
        if self.decimal_exponent >= 0:
            return Fraction(self.mantissa * (10 ** self.decimal_exponent), 1)
        return Fraction(self.mantissa, 10 ** (-self.decimal_exponent))


class BigRational(BaseModel):
    kind: Literal["rational"] = "rational"
    numerator: int
    denominator: int = 1

    @model_validator(mode="after")
    def canonicalize(self):
        if self.denominator == 0:
            raise ValueError("denominator must be non-zero")
        n, d = self.numerator, self.denominator
        if d < 0:
            n, d = -n, -d
        g = math.gcd(abs(n), d)
        self.numerator = n // g
        self.denominator = d // g
        return self

    @classmethod
    def from_decimal(cls, text: str) -> BigRational:
        s = text.strip()
        if not s:
            raise ValueError("empty decimal")
        sign = -1 if s.startswith("-") else 1
        if s[:1] in "+-":
            s = s[1:]
        if "e" in s.lower():
            base, exp_text = s.lower().split("e", 1)
            exp = int(exp_text)
        else:
            base, exp = s, 0
        if "." in base:
            whole, frac = base.split(".", 1)
        else:
            whole, frac = base, ""
        whole = whole or "0"
        if not whole.isdigit() or (frac and not frac.isdigit()):
            raise ValueError(f"invalid decimal: {text}")
        digits = int(whole + frac) if (whole + frac) else 0
        scale = len(frac) - exp
        if scale >= 0:
            return cls(numerator=sign * digits, denominator=10 ** scale)
        return cls(numerator=sign * digits * 10 ** (-scale), denominator=1)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def __lt__(self, other: BigRational) -> bool:
        return self.as_fraction() < other.as_fraction()

    def __le__(self, other: BigRational) -> bool:
        return self.as_fraction() <= other.as_fraction()

    def __gt__(self, other: BigRational) -> bool:
        return self.as_fraction() > other.as_fraction()

    def __ge__(self, other: BigRational) -> bool:
        return self.as_fraction() >= other.as_fraction()


class RealInterval(BaseModel):
    kind: Literal["real_interval"] = "real_interval"
    lower: BigRational
    upper: BigRational

    @model_validator(mode="after")
    def ordered(self):
        if self.lower > self.upper:
            raise ValueError("lower endpoint exceeds upper endpoint")
        return self

    @classmethod
    def from_decimals(cls, lower: str, upper: str) -> RealInterval:
        return cls(lower=BigRational.from_decimal(lower), upper=BigRational.from_decimal(upper))


class ComplexInterval(BaseModel):
    kind: Literal["complex_interval"] = "complex_interval"
    real: RealInterval
    imag: RealInterval


class SymbolicExpression(BaseModel):
    kind: Literal["symbolic"] = "symbolic"
    expression: str
    fingerprint: str | None = None


MathValue = Annotated[
    ScientificInteger | BigRational | RealInterval | ComplexInterval | SymbolicExpression,
    Field(discriminator="kind"),
]


class VerifierMetadata(BaseModel):
    method: str
    precision_bits: int | None = None
    worker_version: str | None = None
    worker_hash: str | None = None
    source_hash: str | None = None


class MathCertificate(BaseModel):
    schema_version: Literal["1"] = "1"
    expression: str
    value: MathValue
    verifier: VerifierMetadata
    assumptions: list[str] = Field(default_factory=list)
    expression_hash: str | None = None

    @model_validator(mode="after")
    def ensure_expression_hash(self):
        if self.expression_hash is None:
            self.expression_hash = hashlib.sha256(self.expression.encode("utf-8")).hexdigest()
        return self

    def canonical_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)

    def certificate_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
