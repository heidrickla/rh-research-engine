# Working in this repository

## The formulas are the product

- Every other artifact — fingerprints, citations, proof-queue verdicts, Lean
  exports, DRE receipts — describes a formula, and describes nothing if the
  formula was read wrong.
- A broken formula is not a broken feature. It is a research record that says
  something its source did not.
- `python tools/formula-guard.py` enforces what is mechanical. It runs in the
  pre-commit hook, the pre-push hook, CI, and at session end.
- Every rule below was learned by shipping its opposite.

## Never leave a formula broken, and never patch around one

- A refusal is a staging post, not a fix. Find the accurate representation.
- `\prod_p` has no SymPy index. Write it over *position*: the product over `k`
  of a term in the k-th prime. Same statement, index SymPy can hold.
- A stub is a refusal wearing a shape. `Function('sigma')(12)` is nothing;
  `divisor_sigma(12)` is 28.
- Use the real function. SymPy has almost all of them; `symbolic/functions.py`
  defines the rest.
- A formula that cannot be evaluated cannot be caught being wrong. Robin's
  inequality was indexed, cited and hashed for months, untestable.
- Not fixes: narrowing the corpus, deleting the failing case, loosening a
  tolerance, moving a formula to a "known limitations" list.

## A clean parse of the wrong thing is worse than a refusal

Assert structure. Never `parse_error is None` — that test passes on every row:

| written | read as | why it went unnoticed |
|---|---|---|
| `\xi(s)=\xi(1-s)` | `Eq(s*xi, xi*(1-s))` | parsed cleanly; printed identically |
| `M(x) = O(...)` | `Mul(Symbol('M'), Symbol('x'))` | M silently stopped being a function |
| `O(\sqrt x \log x)` | SymPy `Order`, a germ at **0** | it also *absorbed* `li(x)`, deleting it |
| `\pi(x)` | `pi * x` | `\pi` is a constant *and* a function |
| `\log\log n` | `Symbol('loglog')` | one name where the source wrote two |
| `\sigma(n) < B` vs `\le B` | identical bytes | the operator was never stored |

## Do not guess between two readings

- Make it explicit in the corpus instead. Bare `\gamma` is Euler's constant in
  Robin's criterion and a zero's ordinate in `\rho = 1/2 + i\gamma`; nothing
  syntactic separates them. Reading it as the constant turned a zero on the
  critical line into a fixed number. The corpus writes `\EulerGamma`.
- Where a genuine convention exists, follow it rather than refusing. Binders,
  integrals, limits and derivatives bind tighter than addition, so
  `\sum_{n=1}^{5} n + 1` is 16.

## One name-resolution policy

- `prepare_for_parsing` in `symbolic/parser.py` is the only one.
- Never `parse_expr` without `local_dict`. It reads the extractor's own output
  under different rules than produced it: `M(x)` became `M*x` in the index, and
  the Lean exporter called a well-formed formula `unparseable`.
- The guard checks this per call, on the syntax tree.
- A formula read under two policies is two formulas, and nothing downstream can
  tell which it has.

## Refusals must be about provability, never about reading

- Honest: the proof queue refusing Robin's inequality — a claim about what
  `ring` over ℚ can prove.
- False: a reader reporting a well-formed formula as `unparseable` — a claim
  about the formula.
- Name what is missing. `not_an_identity` says `ring` cannot discharge this and
  nothing about whether it is true; that is why it is not
  `not_symbolically_true`.

## Docs must not describe limits that no longer exist

- Stale "cannot parse" notes are worse than none: they tell the next reader not
  to try.
- Delete the claim in the same commit that lifts the limitation.

## Meaning is checkable too

A guard can check meaning, not only consistent reading. Believing otherwise is
how `i` sat in the index as a free symbol past every structural check.

- **An undeclared free symbol is a name that failed to resolve.** A `Symbol` is
  indistinguishable from a variable by shape, not by *name*. `DECLARED_SYMBOLS`
  lists every symbol the corpus may contain and what it names. `i`, `it`, `xy`,
  `zeta_`, `loglog`, `igamma_` were all failed resolutions. Adding an entry is a
  decision that the name really is a variable; the list forces that question.
- **An identity that can be evaluated must be true.** Substitute and compare
  wherever both sides share variables. The functional equation without its
  `sin(pi*s/2)` factor gives +1/12 at s = -1 where `zeta(-1)` is -1/12 — it
  parsed, indexed and fingerprinted for weeks; evaluating settles it in
  microseconds.
- Equations introducing a symbol on one side are definitions, not identities.
  Exclude them: they are true of the value they define, not of every value.
- This is why real functions matter. A stub does not merely lose information; it
  removes the possibility of ever catching the error.

## How the data was presented is not part of the answer

- **Truncate at the `print`, never at the record.** A tight set stored at 24 rows
  because 24 is all anyone reads made the characteriser compare 24 against the
  46 actual primes over `n = 2..199`, report `UNRECOGNISED`, and rank the
  finding *up* for being unexplained — a known identity promoted to open
  structure by a display limit. `WitnessCharacter` had already been fixed for
  reading counts off truncated lists; the same bug was two files away.
- **A result must not depend on the order of its inputs.** Testing only
  `left >= right` made `Omega >= omega, tight exactly on the squarefree numbers`
  reachable one way round and invisible the other. Nothing mathematical
  separated them.
- **Never conclude from your own earlier truncated output. Go back to the
  record.** Evaluating von Mangoldt's explicit formula against `psi(x)` left a
  residual of exactly `-log(2 pi)`; the term was in the record, and the listing
  read from was printed at `expression[:96]`. That nearly put a false correction
  in the corpus.
- **Check the mathematics, never a substring of a printed form.** The shape check
  written to catch that searched for `log(2*pi)` and reported it missing, because
  SymPy canonicalises to `log(pi) + log(2)`. Collect the constant terms and
  assert `simplify(constant + log(2*pi)) == 0`.
- When a result could differ because of how the input was *arranged*, that is the
  bug, not a matter of taste.

## Refuted and not tested must never share a verdict

- `patterns/ledger.py` keeps findings nothing has explained so a wider scan can
  judge them. Scan `n = 2..40`, then `n = 50..90`, and the recorded tight set is
  simply absent from the new run — which reads exactly like a refutation while
  nothing was examined.
- **Absence is the refutation, so every other reason for absence is a bug.**
  Three found so far: the range, renaming, and explaining.
- *Range.* Pass it as an argument to the judgement. It cannot be read back off
  whichever finding survived, precisely in the case that matters.
- *Renaming.* `Mertens` comes off the sieve, `cum_mu` from sympy's `mobius`,
  kept apart so their agreement cross-checks two implementations. Same values on
  every row, so every M(x) relation was found and filed twice under two
  spellings — the ledger's size became a fact about how many implementations the
  sweep carried. Report each relation once; a finding carries the names it
  equally holds under, and the ledger looks entries up by all of them.
- *Explaining.* A finding the noise registry retires is absent from the scan too.
  `RETIRED` carries the rule's reason and is read *before* the range check: a
  reason somebody wrote does not depend on what happened to run.
- `retired`, `range` and `columns` are all arguments to the judgement. Each
  changes the verdict, none is recoverable from the surviving finding, so none
  may have a default.
- **A widening can refute a characterisation and cannot confer one.** `EXPLAINED`
  was written and removed: the predicates are pointwise, so they commute with
  restriction. "We now know what this is" belongs in the noise registry, which
  demands a reason.

## A gate that cannot fail is decoration

- **Break the thing a check checks and watch it fail first.** `scratchpad`
  mutation runs are fine for the first pass; the scenario belongs in the test
  file afterwards, so the next person inherits the proof rather than the claim.
- Two checks in `tests/test_formula_guard.py` originally passed a deliberately
  broken tree: the name-resolution check read a five-line window and caught the
  *neighbouring* call's `local_dict`, and the stub check looked for a SymPy
  attribute spelled like the LaTeX name, which `\sigma` → `divisor_sigma` is not.
- **A gate that reads the environment is a gate about the environment.** Five
  trust tests asserted the absent-backend posture — `registered_adapter_count ==
  0`, `verifier_version == "unavailable"`, ACCEPTED refused for want of an
  adapter — and passed only because nobody had `python-flint` installed.
  Installing it broke all five at once.
- **A hole closed by absence is not closed.** `ACCEPTED` from an unregistered
  family maps to `UNKNOWN` — the whole answer only while no family is ever
  registered. With a backend present the family IS registered, so a hand-built
  envelope earned `rigorous_numerical` with no computation behind it.
- Force the environment the test is about and write both branches.
  `_test_only_registered_adapters` takes the empty set for the narrowing. A
  fail-closed refusal nobody has watched work is a refusal nobody has watched
  work.
- **Reach a guard through a seam in *our* code, never by mutating a third-party
  type.** Substituting a method on `flint.types.arb` works against the abi3 wheel
  and raises `cannot set attribute of immutable type` against cp313: the gate ran
  on 3.12 and was skipped on 3.13, in the commit that named the problem.
- **An enumerating check needs a floor.** Breaking the checked thing finds a gate
  that cannot fail *today*; it does not find one that succeeds at scanning
  nothing tomorrow, after a rename or a moved directory. Three passed on empty:
  `formula-guard`'s name-policy check over zero modules, its index check with the
  index file deleted, and `assumption-guard` with an empty registry. `0 == 0`
  counts as agreement, so an empty corpus and an empty index confirm each other.

## Practical notes

- **Verify CI's real conclusion** before merging: `gh run view <id> --json
  conclusion -q .conclusion`. Watching output scroll past is not checking.
- **Hooks need installing once per clone:** `git config core.hooksPath
  tools/githooks`.
- **Re-ingest after editing the corpus.** Record ids are content hashes, so a
  correction files a *new* record; the ingest prunes what the document no longer
  says, and the guard checks the counts match.
- **The open-findings ledger is not committed.** `rhre patterns sweep` rewrites
  `research_state/pattern_open.json` every run, so it is gitignored and named in
  the baseline's `excluded_as_regenerated`. The noise registry beside it *is*
  tracked: a triage decision carries a reason somebody wrote once, and losing one
  is losing research.
- **Line endings are LF.** Hashes are computed over exact bytes.
