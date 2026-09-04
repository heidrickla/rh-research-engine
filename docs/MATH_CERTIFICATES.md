# Math Certificates

Version 0.8 introduces an exact numerical boundary between high-precision mathematics and DRE.

## Why

DRE intentionally bans floating point and its core `Fixed` type carries six decimal places. That is appropriate for deterministic operational reasoning, but RH computations routinely require values far smaller, larger, and more precise than six decimals can represent.

The solution is not to add `f64` to DRE. High-precision workers produce exact certificates; DRE reasons about certified predicates and hashes.

## Exact value types

- `ScientificInteger` — arbitrary-size integer mantissa plus decimal exponent.
- `BigRational` — arbitrary-size canonical numerator/denominator.
- `RealInterval` — closed interval with exact rational endpoints.
- `ComplexInterval` — independent certified real/imaginary intervals.
- `SymbolicExpression` — symbolic object with optional structural fingerprint.

## Certificate

`MathCertificate` commits:

- mathematical expression;
- exact value or enclosure;
- assumptions;
- verifier method;
- precision bits;
- worker/source hashes;
- canonical SHA-256 certificate hash.

Example:

```yaml
schema_version: "1"
expression: screening_remainder(X=100000,q=4)
value:
  kind: real_interval
  lower: {kind: rational, numerator: 1381729347501, denominator: 1000000000000}
  upper: {kind: rational, numerator: 1381729347519, denominator: 1000000000000}
verifier:
  method: arb
  precision_bits: 512
```

## DRE boundary

`certificate_predicates()` deliberately does not export the high-precision endpoints. It exports facts such as:

- `certificate_hash`
- `expression_hash`
- `verifier_method`
- `precision_bits`
- `assumption_count`, `assumptions_present`, `unconditional`
- `assumption_1 … assumption_n` (verbatim)
- `definitely_positive = true|false|unknown`
- `definitely_negative = true|false|unknown`
- `contains_zero = true|false`

The assumption facts are not optional. Omitting them turned a conditional
enclosure into an unconditional three-valued fact: a certificate annotated
"assumes RH" reached DRE indistinguishable from an unconditional one, and an
RH-assuming certificate then counted as evidence *for* RH. Pack rule RH007
blocks promotion whenever `assumptions_present` is true.

The full certificate remains an immutable evidence artifact. DRE can therefore make deterministic three-valued decisions without truncating the underlying mathematics.

## Exact comparisons

For intervals `A` and `B`:

- `A < B` is TRUE only when `upper(A) < lower(B)`;
- FALSE when `lower(A) >= upper(B)`;
- otherwise UNKNOWN.

No epsilon comparisons or floating-point rounding are used.

## Certified exponent propagation

If a rigorous worker encloses a screening exponent

\[
\theta \in [\theta_-,\theta_+],
\]

then the implied zero-edge interval is propagated exactly as

\[
\Theta \le \frac12 + \frac{\theta}{2}.
\]

The research supervisor should use the upper endpoint for a rigorous bound.

## Verification status

**No Arb, FLINT, MPFI, or interval backend is bundled.** `mathcert` imports only
the standard library and pydantic, so `method: arb` in the example above is a
label describing where a certificate *came from*, not evidence that anything
was checked.

`REGISTERED_ADAPTERS` in `mathcert/verifiers.py` is empty on purpose, and
`validate_external_envelope` refuses `ACCEPTED` for any family without a
registered adapter. The honest envelope for a disconnected backend reports
status `unknown`. An accepted envelope must additionally carry
`verifier.worker_hash` and `verifier.source_hash`, because `certificate_hash()`
recomputes over whatever the metadata currently says and therefore detects no
tampering by itself.

## Future verifier workers

Recommended independent evidence groups:

1. Arb/FLINT interval worker;
2. PARI/GP high-precision worker;
3. symbolic exact worker;
4. eventually Lean-verified finite algebra.

Repeated runs of one implementation remain one DRE independence group.
