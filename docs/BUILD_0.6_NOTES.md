# Build 0.6 — Symbolic Math Compiler

## Added

- conservative Markdown/LaTeX equation extractor and AST records
- controlled simplifier with named rewrite provenance
- canonical equation fingerprinting and equivalence checking
- hidden-assumption extraction
- proof-gap extraction
- exact transform registry
- residue and asymptotic helpers
- deterministic exponent propagation
- `rhre symbolic ...` CLI

## Verification

- 26 Python tests pass.
- Unsupported equation syntax is retained as an unparsed record instead of guessed.
- Symbolic results remain worker evidence; DRE is authoritative for research status.
