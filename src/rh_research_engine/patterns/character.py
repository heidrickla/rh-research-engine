"""What IS the set a relation is tight on?

A bound tight on a proper subset is only a finding if you can see which subset.
`sigma(n) >= n+1, tight in 12 of 39` is a number; `tight exactly on the primes`
is the characterisation of the primes by their divisor sum. Same data, and only
the second one is a result.

So the exception set gets tested against predicates the engine already knows
how to compute. Two outcomes matter, and the second matters more:

  * IT MATCHES. The relation characterises that predicate on this range. Very
    often this means the finding is a known identity in disguise, and it can
    then be retired with a reason rather than by taste.

  * IT MATCHES NOTHING. A relation holding exactly on a set with no
    recognisable description is the interesting case -- it is structure that
    has not been named, which is the whole point of looking. Reported as
    UNRECOGNISED rather than dropped, and ranked up rather than down.

The predicate list is deliberately short and ordinary. It is here to explain
away the explainable so that what is left is genuinely unexplained; a long list
of exotic predicates would explain away things that deserve a second look.

A NEAR MISS MUST SAY WHICH. The headline reported "closest description: prime
(off by 1)" -- which is a number, and the first paragraph of this file is about
the difference between a number and a description. Both lists were computed and
thrown away at the print. `n - phi(n) >= Omega(n)` is tight on the primes
together with 4; read as "off by 1" a reader cannot tell whether the one is a
stray composite, a missing prime, or a bug, and the finding sat in the open
ledger as unexplained structure when the structure was "and also 4".

Naming them does NOT set `characterised`. A description is not an explanation:
it says what to go and prove. Letting a near miss close an open question would
retire it by rewording its headline, which is the promotion the ledger exists
to refuse -- so the exceptions are printed and the finding stays open until
somebody writes a reason into the noise registry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field

#: Ordinary predicates on the positive integers.
#:
#: Ordered roughly by how commonly a tight set turns out to be one of them.
#: Congruences are generated rather than listed so a residue class is
#: recognised without a table of every modulus.
_BASE_PREDICATES: dict[str, Callable[[int], bool]] = {
    "prime": lambda n: bool(sp.isprime(n)),
    "composite": lambda n: n > 1 and not sp.isprime(n),
    "prime power": lambda n: n > 1 and len(sp.factorint(n)) == 1,
    "squarefree": lambda n: all(e == 1 for e in sp.factorint(n).values()) if n > 1 else True,
    "perfect square": lambda n: sp.integer_nthroot(n, 2)[1],
    "power of two": lambda n: n > 0 and (n & (n - 1)) == 0,
    "semiprime": lambda n: sum(sp.factorint(n).values()) == 2,
    "even": lambda n: n % 2 == 0,
    "odd": lambda n: n % 2 == 1,
}


def _predicates() -> dict[str, Callable[[int], bool]]:
    table = dict(_BASE_PREDICATES)
    for modulus in (3, 4, 5, 6):
        for residue in range(modulus):
            table[f"n = {residue} mod {modulus}"] = (
                lambda n, m=modulus, r=residue: n % m == r
            )
    return table


class WitnessCharacter(BaseModel):
    """How well a named predicate describes the set a relation is tight on."""

    model_config = ConfigDict(extra="forbid")

    predicate: str
    #: True when the tight set is EXACTLY the predicate's set on this range.
    exact: bool
    #: In the predicate's set, not tight. TRUNCATED for display.
    missing: list[str] = Field(default_factory=list)
    #: Tight, not in the predicate's set. TRUNCATED for display.
    extra: list[str] = Field(default_factory=list)
    #: True sizes. Kept separately because the lists above are truncated, and
    #: a count read off a truncated list is wrong exactly when the discrepancy
    #: is large -- which is when the count matters most.
    missing_count: int = 0
    extra_count: int = 0

    @property
    def discrepancy(self) -> int:
        return self.missing_count + self.extra_count

    def difference(self) -> str:
        """WHICH rows the predicate and the tight set disagree on.

        "Off by 1" is a number, which is what the module docstring above says
        is not a finding -- and the headline was printing exactly that while
        `missing` and `extra` sat computed and unused beside it. `n -
        totient(n) >= Omega(n)` is tight on the primes together with 4, and
        reading "closest description: prime (off by 1)" a reader cannot tell
        whether that 1 is a stray composite, a missing prime, or a typo.

        Counted from `missing_count` and `extra_count`, never from the lists:
        those hold at most eight, so `len()` of one is the true size only
        while the discrepancy is small enough not to matter. Naming eight of
        twelve and calling it twelve is the truncation bug this file's
        neighbours keep committing.
        """
        parts = []
        for label, shown, total in (
            ("except", self.missing, self.missing_count),
            ("and also", self.extra, self.extra_count),
        ):
            if not total:
                continue
            if total == len(shown):
                parts.append(f"{label} {_and_list(shown)}")
            else:
                rest = total - len(shown)
                parts.append(
                    f"{label} {', '.join(shown)} and {rest} more ({total} in all)"
                )
        return ", ".join(parts)


def _and_list(items: Sequence[str]) -> str:
    """`a`, `a and b`, `a, b and c` -- so a reader can read it aloud."""
    values = list(items)
    if len(values) <= 1:
        return "".join(values)
    return f"{', '.join(values[:-1])} and {values[-1]}"


class Characterisation(BaseModel):
    """The verdict on a tight set."""

    model_config = ConfigDict(extra="forbid")

    #: Predicates matching exactly, then near-misses, best first.
    matches: list[WitnessCharacter] = Field(default_factory=list)

    @property
    def recognised(self) -> bool:
        return any(match.exact for match in self.matches)

    @property
    def headline(self) -> str:
        for match in self.matches:
            if match.exact:
                return f"tight exactly on: {match.predicate}"
        if self.matches:
            best = self.matches[0]
            return f"closest description: {best.predicate}, {best.difference()}"
        return "UNRECOGNISED -- tight on a set matching no known predicate"


def characterise(
    witnesses: Sequence[str], universe: Sequence[str], *, near_misses: int = 3
) -> Characterisation:
    """Describe the tight set, or report that nothing describes it.

    `universe` is every row that was examined, so "missing" means a case the
    predicate claims and the relation did not deliver -- which is only
    meaningful against the range actually tested.
    """
    try:
        tight = {int(value) for value in witnesses}
        every = {int(value) for value in universe}
    except ValueError:
        # Rows that are not integers cannot be described by these predicates,
        # and guessing at a description would be worse than declining to.
        return Characterisation()

    if not tight or not every:
        return Characterisation()

    scored: list[tuple[WitnessCharacter, int]] = []
    for name, predicate in _predicates().items():
        try:
            expected = {n for n in every if predicate(n)}
        except Exception:
            continue
        if not expected:
            continue
        missing = sorted(expected - tight)
        extra = sorted(tight - expected)
        scored.append(
            (
                WitnessCharacter(
                    predicate=name,
                    exact=not missing and not extra,
                    missing=[str(n) for n in missing[:8]],
                    extra=[str(n) for n in extra[:8]],
                    missing_count=len(missing),
                    extra_count=len(extra),
                ),
                len(expected),
            )
        )

    scored.sort(key=lambda item: (not item[0].exact, item[0].discrepancy))
    exact = [match for match, _ in scored if match.exact]
    if exact:
        return Characterisation(matches=exact)

    # A near-miss has to be small relative to BOTH sets. Measured against the
    # tight set alone, a five-element set "nearly" matched the three-element
    # class n = 4 mod 6 -- true, and vacuous, because the class is a subset.
    # "Almost the primes, off by one" is worth printing; "contains a small
    # congruence class" is not.
    close = [
        match
        for match, expected_size in scored[:near_misses]
        if match.discrepancy <= max(1, min(len(tight), expected_size) // 3)
    ]
    return Characterisation(matches=close)
