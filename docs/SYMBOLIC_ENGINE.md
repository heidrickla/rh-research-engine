# Symbolic Math Engine

Version 0.6 adds a conservative symbolic layer around the DRE-supervised RH research workflow.

## Components

- **Equation extractor / AST** — extracts Markdown/LaTeX math blocks and records normalized SymPy structure when parsing is supported.
- **Controlled simplifier** — applies named exact rewrite rules and emits a provenance trace rather than silently rewriting expressions.
- **Equivalence + fingerprinting** — canonicalizes rational/polynomial expressions, compares symbolic differences, and hashes canonical forms.
- **Assumption extractor** — surfaces denominator, logarithm, and Gamma-domain constraints.
- **Proof-gap extractor** — identifies non-rigorous steps and rigorous-labeled steps that depend on non-rigorous inputs.
- **Transform registry** — exact Mellin/Fourier/Laplace identities used by the RH program.
- **Residue/asymptotic utilities** — mechanical helpers for meromorphic and limiting calculations.
- **Exponent propagation** — deterministic maps from proved asymptotic exponents to implied bounds on `Theta`.

## Epistemic boundary

The symbolic engine is a worker. A SymPy result or parser match is not automatically a theorem. Every result that changes authoritative research state must still enter DRE with provenance and the appropriate evidence class.

The equation parser is deliberately conservative. Unsupported LaTeX remains an unparsed extracted equation rather than being guessed into an AST.
