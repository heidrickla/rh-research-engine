"""An assumption that does not say where it fails is the dangerous kind.

`ExperimentResult.assumptions` was free text. Nothing read it, nothing checked
it, and nothing could: a string is not something a gate can act on. So
`weil_certified` carried "Re psi(z) <= log|z| for Re z > 0" in the assumptions
list of a `rigorous_numerical` record, and it is false -- the margin behaves as
`(x/2 - 1/12)/y^2` and changes sign at `Re z = 1/6`. The computation was fine,
because it only ever evaluates at `Re z = 1/4`. The RECORD was not.

THE FIELD THAT WOULD HAVE CAUGHT IT IS `fails_outside`. The statement named a
domain, `Re z > 0`, and was checked -- informally, by whoever wrote it -- on the
part of that domain the code actually visits. What nobody wrote down is where it
stops being true, because nobody had looked. An assumption with a domain and no
known boundary is not a conservative statement; it is an unexamined one, and the
gap between "holds where we checked" and "holds on the domain we claimed" is
precisely where this one lived.

So `fails_outside` is REQUIRED and may not be empty. Three answers are allowed:
a boundary, `NEVER_FAILS` for something true on its whole stated domain with a
reason, or `UNEXAMINED` -- which is honest, and which the guard reports rather
than accepts silently. An assumption nobody has probed for a boundary and one
known to have none must not read the same, for the same reason "refuted" and
"not tested" must not share a verdict in `patterns/ledger.py`.

`checked_by` names a test that fails if the assumption is false. Not a citation,
not a comment: a test node id. `nogo.py` already does this for refuted routes;
this does it for the things a result still rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

#: True on the whole stated domain, with a reason -- not merely unrefuted.
NEVER_FAILS = "NEVER_FAILS"
#: Nobody has looked for the boundary. Honest, and reported as a gap.
UNEXAMINED = "UNEXAMINED"


@dataclass(frozen=True)
class Assumption:
    """Something a result rests on, with the two facts that make it actionable."""

    id: str
    statement: str
    #: Where it was actually checked -- never the domain it might hold on.
    holds_on: str
    #: Where it is known to break, or NEVER_FAILS, or UNEXAMINED. Never empty.
    fails_outside: str
    #: A pytest node id that fails if the statement is false.
    checked_by: str

    def __post_init__(self) -> None:
        for field in ("id", "statement", "holds_on", "fails_outside", "checked_by"):
            if not getattr(self, field).strip():
                raise ValueError(f"assumption {self.id!r} has an empty {field}")
        if "::" not in self.checked_by:
            raise ValueError(
                f"assumption {self.id!r} names {self.checked_by!r}, which is not a "
                "pytest node id; an assumption is checked by a test that can fail, "
                "not by a citation or a comment"
            )

    @property
    def examined(self) -> bool:
        return self.fails_outside != UNEXAMINED

    def cite(self) -> str:
        """The line that goes in a record: readable, and resolvable to this entry."""
        if self.fails_outside == NEVER_FAILS:
            boundary = "no boundary: true on the whole stated domain"
        elif self.fails_outside == UNEXAMINED:
            boundary = "BOUNDARY UNEXAMINED -- nobody has looked for where this fails"
        else:
            boundary = f"fails outside: {self.fails_outside}"
        return f"[{self.id}] {self.statement} (checked on {self.holds_on}; {boundary})"


REGISTRY: dict[str, Assumption] = {}


def register(assumption: Assumption) -> Assumption:
    if assumption.id in REGISTRY:
        raise ValueError(f"assumption id {assumption.id!r} is already registered")
    REGISTRY[assumption.id] = assumption
    return assumption


def cite(*ids: str) -> list[str]:
    """The `assumptions` list for a record, from registry ids."""
    return [REGISTRY[i].cite() for i in ids]


def assumption_id(line: str) -> str | None:
    """The id a recorded assumption line resolves to, or None if it names none."""
    if not line.startswith("["):
        return None
    closing = line.find("]")
    return line[1:closing] if closing > 1 else None


DIGAMMA_TAIL = register(
    Assumption(
        id="digamma-tail",
        statement="|Re psi(1/4 + i r/2)| <= log r",
        holds_on="real r >= 3, measured to r = 1e5",
        fails_outside=(
            "the general form Re psi(z) <= log|z| on Re z > 0, which is FALSE for "
            "0 < Re z < 1/6 -- the margin goes as (x/2 - 1/12)/y^2 and changes sign "
            "exactly at Re z = 1/6. This record previously declared that form"
        ),
        checked_by=(
            "tests/test_weil_certified.py::"
            "test_the_general_digamma_bound_is_false_and_the_one_used_is_not"
        ),
    )
)

VON_MANGOLDT_BOUND = register(
    Assumption(
        id="von-mangoldt-bound",
        statement="Lambda(n) <= log n, used to bound the prime tail by its integral",
        holds_on="every integer n >= 2",
        fails_outside=NEVER_FAILS,
        checked_by="tests/test_assumptions.py::test_von_mangoldt_never_exceeds_the_log",
    )
)
