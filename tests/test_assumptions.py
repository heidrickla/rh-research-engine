"""The assumption that sat false in a rigorous record for a day.

`weil_certified` declared "Re psi(z) <= log|z| for Re z > 0" in the
`assumptions` list of a `rigorous_numerical` result. It is false below
Re z = 1/6. It survived because `assumptions` was free text: no gate read the
field, so no gate could.

Each test here breaks one of the three things that now make an assumption
actionable -- it resolves to a registry entry, it names a live test, and it says
where it fails.
"""

from __future__ import annotations

import math

import pytest

from rh_research_engine.core.assumptions import (
    NEVER_FAILS,
    REGISTRY,
    UNEXAMINED,
    Assumption,
    assumption_id,
    cite,
)


def _valid(**overrides) -> dict:
    base = dict(
        id="probe",
        statement="something is true",
        holds_on="the range measured",
        fails_outside="outside that range",
        checked_by="tests/test_assumptions.py::test_a_registered_assumption_is_well_formed",
    )
    base.update(overrides)
    return base


def test_a_registered_assumption_is_well_formed():
    entry = Assumption(**_valid())
    assert entry.examined
    assert entry.id in entry.cite()


@pytest.mark.parametrize(
    "field", ["id", "statement", "holds_on", "fails_outside", "checked_by"]
)
def test_no_field_may_be_empty(field):
    """`fails_outside` especially. An assumption with a domain and no known
    boundary is not conservative, it is unexamined -- and that is where the
    false digamma bound lived."""
    with pytest.raises(ValueError, match=f"empty {field}"):
        Assumption(**_valid(**{field: "   "}))


def test_checked_by_must_be_a_test_and_not_a_citation():
    """A comment cannot fail. A reference cannot fail. A test can."""
    with pytest.raises(ValueError, match="not a pytest node id"):
        Assumption(**_valid(checked_by="Titchmarsh, Theorem 9.2"))
    with pytest.raises(ValueError, match="not a pytest node id"):
        Assumption(**_valid(checked_by="see the module docstring"))


def test_unexamined_reads_differently_from_no_boundary():
    """The distinction the whole module exists for.

    "Nobody looked for where this fails" and "there is no boundary" must not
    produce the same line, for the same reason refuted and not-tested must not
    share a verdict.
    """
    unexamined = Assumption(**_valid(fails_outside=UNEXAMINED))
    unbounded = Assumption(**_valid(fails_outside=NEVER_FAILS))
    assert not unexamined.examined
    assert unbounded.examined
    assert "UNEXAMINED" in unexamined.cite()
    assert "no boundary" in unbounded.cite()
    assert unexamined.cite() != unbounded.cite()


def test_a_citation_resolves_back_to_its_entry():
    """A record's line has to be readable by a person AND by the guard."""
    for identifier in REGISTRY:
        assert assumption_id(REGISTRY[identifier].cite()) == identifier
    assert assumption_id("Re psi(z) <= log|z| for Re z > 0") is None, (
        "free text names no entry, which is exactly how the false one survived"
    )
    assert assumption_id("[unregistered] something") == "unregistered"


def test_cite_produces_the_recorded_lines():
    lines = cite("digamma-tail", "von-mangoldt-bound")
    assert len(lines) == 2
    assert all(assumption_id(line) in REGISTRY for line in lines)


def test_the_registry_records_where_the_digamma_bound_fails():
    """The boundary that was missing, named in the entry rather than in prose."""
    entry = REGISTRY["digamma-tail"]
    assert entry.examined
    assert "1/6" in entry.fails_outside
    assert "FALSE" in entry.fails_outside
    assert "r >= 3" in entry.holds_on


def test_von_mangoldt_never_exceeds_the_log():
    """Named by `von-mangoldt-bound` as the test that would fail if it were false.

    `Lambda(n)` is `log p` for `n = p^k` and zero otherwise, so it is at most
    `log n` with equality exactly at the primes. The prime tail bound replaces
    the sum by its integral using this, so it is load-bearing rather than
    decorative.
    """
    from rh_research_engine.experiments.weil_positivity import von_mangoldt

    n, lam = von_mangoldt(20_000)
    assert len(n) > 2000
    for value, weight in zip(n.tolist(), lam.tolist(), strict=True):
        assert weight <= math.log(value) + 1e-12, (value, weight)
    # Equality at the primes is what makes it tight rather than merely true.
    assert abs(lam[0] - math.log(2)) < 1e-12
    assert abs(lam[list(n).index(4.0)] - math.log(2)) < 1e-12


def test_the_guard_refuses_to_pass_on_an_empty_enumeration(tmp_path):
    """A check that enumerates can succeed at scanning nothing.

    The first version of `assumption-guard.py` reported PASS with an empty
    registry and no recorded assumptions -- "every recorded assumption resolves"
    is vacuously true of nothing. Breaking the checked thing does not find that:
    an injected violation still fails. The hole is not that the gate cannot fail
    today, it is that it can succeed at seeing nothing tomorrow, after a rename
    or a wrong working directory.

    So both enumerations carry a floor, and this asserts the floors exist and
    are positive. Verified by raising each one absurdly high and watching the
    guard exit 1, and by deleting the records file, which now fails where it
    used to pass.
    """
    import importlib.util
    from pathlib import Path as _Path

    tool = _Path(__file__).resolve().parent.parent / "tools" / "assumption-guard.py"
    spec = importlib.util.spec_from_file_location("assumption_guard", tool)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    assert guard.MINIMUM_REGISTERED >= 1
    assert guard.MINIMUM_RECORDED >= 1
    # The floors must be satisfiable by the real repository, or they are a
    # different kind of decoration.
    assert len(REGISTRY) >= guard.MINIMUM_REGISTERED
    rows, _ = guard._recorded_assumption_lines()
    assert len(rows) >= guard.MINIMUM_RECORDED, (
        "records carrying assumptions exist here; finding none means the guard "
        "read the wrong file"
    )

