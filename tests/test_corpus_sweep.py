"""Pointing the discovery loop at the corpus's own quantities.

The columns are fast because every summatory one is accumulated instead of
re-summed, so what is under test is that they are still RIGHT -- and that the
scan over them still finds the things it is supposed to find, which is how a
column silently going wrong would show up.
"""

from __future__ import annotations

import mpmath
import pytest
import sympy as sp

from rh_research_engine.patterns import (
    RegularityKind,
    escalate,
    scan_for_regularities,
)
from rh_research_engine.patterns.corpus_sweep import (
    CorpusColumns,
    build_observations,
    exact_text,
)


@pytest.fixture(scope="module")
def columns() -> CorpusColumns:
    built = CorpusColumns(300)
    built.load_zeros()
    return built


def test_every_accumulated_column_agrees_with_the_slow_route(columns):
    """A column that is merely fast is a column that has not been checked."""
    assert columns.verify_shortcuts()


def test_the_two_summatory_mobius_columns_are_computed_differently(columns):
    """`Mertens = cum_mu` is only a rediscovery if they are two computations.

    One is accumulated from the sieve, the other from sympy's `mobius`. They
    read the same array at first, which made the finding a restatement of an
    assignment.
    """
    assert columns.mertens is not columns.independent_mertens
    assert columns.mertens[:302] == columns.independent_mertens[:302]


def test_von_mangoldt_is_taken_from_the_factorisation(columns):
    """Not as `psi(n) - psi(n-1)`.

    The accumulated sums carry rounding, so their difference is `log p` plus an
    error in the last bits -- enough that `Lambda(n) = log n exactly at the
    primes` stopped surviving a doubling of the precision, losing the von
    Mangoldt characterisation of the primes to an arithmetic convenience.
    """
    for n in (2, 4, 6, 8, 9, 12, 49, 128):
        expected = (
            mpmath.log(min(sp.factorint(n)))
            if len(sp.factorint(n)) == 1
            else mpmath.mpf(0)
        )
        assert columns.von_mangoldt(n) == expected, n

    difference = columns.psi[128] - columns.psi[127]
    assert difference != columns.von_mangoldt(128), (
        "if these agreed exactly, the reason for computing Lambda separately "
        "would be gone"
    )


def test_an_exact_value_stays_exact():
    """Exactness is the property the scan measures."""
    assert exact_text(sp.Rational(137, 60), 30) == "137/60"
    assert exact_text(sp.Integer(28), 30) == "28"
    assert "." in exact_text(mpmath.log(2), 30)


def test_the_scan_rediscovers_what_the_corpus_already_states(columns):
    """The validation: a scan that cannot find what it knows will not find more.

    `Lambda(n) <= log n` with equality exactly at the primes is von Mangoldt's
    definition read back as a characterisation, and it comes from two columns
    that both appear in the indexed corpus.
    """
    findings = scan_for_regularities(build_observations(columns, 30))
    statements = {f.statement.split(" throughout")[0] for f in findings}
    assert "log >= Lambda" in statements
    # The cross-check between two implementations of M(x), which is the whole
    # reason `cum_mu` is accumulated from sympy's mobius instead of read off
    # the sieve. Reported ONCE, naming both -- and the relations that mention
    # M(x) are then reported once too, instead of once per name.
    agreement, = [
        f
        for f in findings
        if f.kind is RegularityKind.EXACT_EQUALITY and "Mertens" in f.columns
    ]
    assert agreement.columns == ["Mertens", "cum_mu"]
    assert "examined rows" in agreement.statement
    assert not [
        f for f in findings if f.kind is not RegularityKind.EXACT_EQUALITY
        and "cum_mu" in f.columns
    ], "a second name for M(x) is not a second finding"

    von_mangoldt = next(
        f
        for f in findings
        if f.kind is RegularityKind.SATURATED_BOUND
        and f.columns == ["log", "Lambda"]
    )
    assert von_mangoldt.character == "tight exactly on: prime"


def test_the_scan_characterises_the_primes_and_the_squarefree_numbers(columns):
    """Two named sets, from columns nobody asked to have compared."""
    ranked = escalate(scan_for_regularities(build_observations(columns, 30)))
    characters = {f.character for f in ranked}
    assert "tight exactly on: prime" in characters
    assert "tight exactly on: squarefree" in characters


def test_the_zero_columns_produce_no_exact_relation(columns):
    """A real limit of the instrument, recorded so it is not read as a result.

    The detector looks for EXACT relations. Zero ordinates are irrational and
    satisfy none with an arithmetic function, so no widening will produce one --
    the structure that is there is distributional, and a detector built to
    refuse statistical trends cannot see it by construction.
    """
    assert columns.ordinates is not None
    zero_columns = {
        "zero_ordinate",
        "zero_gap",
        "normalised_gap",
        "S_at_zero",
    }
    findings = scan_for_regularities(build_observations(columns, 30))
    involved = [f for f in findings if zero_columns.intersection(f.columns)]
    assert all(f.kind is not RegularityKind.EXACT_EQUALITY for f in involved), [
        f.statement for f in involved
    ]


def test_a_sweep_needs_more_than_a_couple_of_rows():
    with pytest.raises(ValueError, match="at least a few rows"):
        CorpusColumns(3)
