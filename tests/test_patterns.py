"""The pattern-detection research function.

The behaviour under test is the sequence that produced Anthropic's zeta result:
audit the premise, measure what was not asked for, escalate an exact
regularity -- and, above all, refuse to let a regularity become a result.
"""

from __future__ import annotations

import pytest
import sympy as sp
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import NON_DEDUCTIVE, RIGOROUS, Confidence
from rh_research_engine.patterns import (
    MINIMUM_SAMPLE,
    NoiseGround,
    NoiseRegistry,
    NoiseRule,
    Observation,
    OpenLedger,
    PatternFinding,
    PremiseVerdict,
    RegularityKind,
    RevisitVerdict,
    audit_premise,
    characterise,
    escalate,
    scan_columns,
    scan_for_regularities,
    scan_universe,
)

# --- the guard, which is the whole safety property ------------------------


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_a_finding_cannot_claim_a_rigorous_confidence(confidence):
    """Thirteen of thirteen is a reason to look for a proof, not a substitute.

    Refused at construction rather than checked at export: a record that can be
    built wrong will eventually be built wrong, and this is the exact step --
    unanimous agreement reading as certainty -- that the repository exists to
    prevent.
    """
    with pytest.raises(ValidationError) as caught:
        PatternFinding(
            kind=RegularityKind.EXACT_EQUALITY,
            statement="a = b",
            columns=["a", "b"],
            support=13,
            sampled=13,
            exact=True,
            confidence=confidence,
        )
    assert "conjecture" in str(caught.value)


@pytest.mark.parametrize("confidence", sorted(NON_DEDUCTIVE, key=str))
def test_a_finding_may_claim_any_non_deductive_confidence(confidence):
    finding = PatternFinding(
        kind=RegularityKind.EXACT_EQUALITY,
        statement="a = b",
        columns=["a", "b"],
        support=3,
        sampled=3,
        exact=True,
        confidence=confidence,
    )
    assert finding.confidence is confidence


def test_the_default_confidence_is_conjectural():
    finding = PatternFinding(
        kind=RegularityKind.INTEGRALITY,
        statement="n is an integer",
        columns=["n"],
        support=5,
        sampled=5,
        exact=True,
    )
    assert finding.confidence is Confidence.CONJECTURAL
    assert finding.confidence not in RIGOROUS


# --- auditing the premise -------------------------------------------------


def test_a_quantity_that_is_identically_zero_has_nothing_to_fit():
    """The move that closed the assigned route.

    Fifty minutes in and before any computation, the sub-agent argued the
    brief's own quantity "is IDENTICALLY ZERO in every computable range ...
    There is nothing to fit."
    """
    result = audit_premise("kappa", ["0", "0", "0", "0", "0"])
    assert result.verdict is PremiseVerdict.EMPTY
    assert "nothing to fit" in result.evidence
    assert result.confidence not in RIGOROUS


def test_a_constant_non_zero_quantity_is_also_empty():
    result = audit_premise("c", ["7", "7", "7", "7"])
    assert result.verdict is PremiseVerdict.EMPTY
    assert "identically 7" in result.evidence


def test_a_quantity_pinned_to_the_expected_value_is_degenerate():
    """Fitting it would recover the assumption rather than test it.

    Reported as DEGENERATE rather than EMPTY because that is the more useful
    of the two true statements: such a fit looks like a confirmation.
    """
    result = audit_premise("theta", ["1/2", "1/2", "1/2", "1/2"], target="1/2")
    assert result.verdict is PremiseVerdict.DEGENERATE
    assert "recovers the assumption" in result.evidence


def test_a_varying_quantity_is_live():
    result = audit_premise("n", ["3", "5", "8", "11"])
    assert result.verdict is PremiseVerdict.LIVE


def test_too_few_points_says_so_rather_than_guessing():
    """Two points make anything constant. Defaulting to LIVE would be a guess."""
    result = audit_premise("n", ["1", "1"])
    assert result.verdict is PremiseVerdict.INCONCLUSIVE
    assert str(MINIMUM_SAMPLE) in result.evidence


# --- scanning for structure -----------------------------------------------


def _columns():
    """A bound that is saturated in every case, in an unrequested column."""
    return [
        Observation(name="frobenius_sq", values=["9", "14", "21", "30", "41", "54"]),
        Observation(
            name="n_on_plus_4n_p",
            values=["9", "14", "21", "30", "41", "54"],
            requested=False,
            source="measured while verifying facts the brief only asserted",
        ),
        Observation(name="trace", values=["6", "9", "12", "15", "18", "21"]),
    ]


def test_the_scan_finds_the_relation_between_columns():
    findings = scan_for_regularities(_columns())
    equalities = [f for f in findings if f.kind is RegularityKind.EXACT_EQUALITY]
    assert len(equalities) == 1
    assert equalities[0].unanimous
    assert equalities[0].support == 6
    assert equalities[0].from_unrequested


def test_escalation_puts_the_unrequested_relation_first():
    """The column nobody asked for is the one that carried the theorem.

    A scan that only examines the named quantity cannot find it, however
    carefully it examines it -- so the ordering surfaces the unrequested
    relation ahead of properties of a single asked-for column.
    """
    ranked = escalate(scan_for_regularities(_columns()))
    assert ranked[0].kind is RegularityKind.EXACT_EQUALITY
    assert ranked[0].from_unrequested


def test_escalation_drops_properties_of_a_single_column():
    """Integrality of a column of counts is a fact about the generator.

    The first blind run reported twenty-four of those and buried the two
    findings that mattered, so single-column characterisations score zero and
    fall below the floor.
    """
    findings = scan_for_regularities(_columns())
    characterisations = [f for f in findings if len(f.columns) == 1]
    assert characterisations, "the scan should still FIND them"
    assert all(f.surprise == 0 for f in characterisations)
    assert not [f for f in escalate(findings) if len(f.columns) == 1]


def test_a_partially_tight_bound_is_escalated():
    """Unanimity is the wrong filter for a characterisation.

    `sigma(n) >= n+1` is tight exactly on the primes, and a filter keeping only
    relations true in every case throws it away.
    """
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["5", "9", "13", "20", "25"]),
            Observation(name="b", values=["5", "9", "12", "20", "24"]),
        ]
    )
    saturated = [f for f in findings if f.kind is RegularityKind.SATURATED_BOUND]
    assert saturated and not saturated[0].unanimous
    assert saturated[0] in escalate(findings)


def test_a_bound_saturated_in_every_case_is_reported_as_such():
    """`a >= b` that is never slack is an identity nobody has written down."""
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["5", "9", "13", "20"]),
            Observation(name="b", values=["5", "9", "12", "20"]),
        ]
    )
    saturated = [f for f in findings if f.kind is RegularityKind.SATURATED_BOUND]
    assert len(saturated) == 1
    assert saturated[0].support == 3
    assert saturated[0].sampled == 4
    assert not saturated[0].unanimous
    # Escalated even though it is not unanimous: a bound tight on a subset is
    # a characterisation of that subset, which is the whole point.
    assert saturated[0] in escalate(findings)


def test_exactness_is_preserved_rather_than_flattened_through_a_float():
    """"Equality in 13 of 13" is a claim about exactness.

    A tenth is not representable as a double, so a column of tenths compared
    against itself must still come out equal -- and a column that differs in
    the last place must not.
    """
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["1/10", "2/10", "3/10"]),
            Observation(name="b", values=["0.1", "0.2", "0.3"]),
        ]
    )
    assert any(f.kind is RegularityKind.EXACT_EQUALITY for f in findings)

    findings = scan_for_regularities(
        [
            Observation(name="a", values=["1/3", "2/3", "1"]),
            Observation(name="b", values=["0.3333333333", "2/3", "1"]),
        ]
    )
    assert not any(f.kind is RegularityKind.EXACT_EQUALITY for f in findings)


def test_a_constant_ratio_is_found():
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["2", "4", "6", "8"]),
            Observation(name="b", values=["1", "2", "3", "4"]),
        ]
    )
    ratios = [f for f in findings if f.kind is RegularityKind.CONSTANT_RATIO]
    assert len(ratios) == 1
    assert ratios[0].evidence["ratio"] == "2"


def test_a_relation_holding_in_most_cases_is_not_escalated():
    """A trend is not a finding. Unanimity is the filter."""
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["1", "2", "3", "9"]),
            Observation(name="b", values=["1", "2", "3", "4"]),
        ]
    )
    assert not any(
        f.kind is RegularityKind.EXACT_EQUALITY for f in escalate(findings)
    )


def test_too_short_a_column_is_not_scanned():
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["1", "1"]),
            Observation(name="b", values=["1", "1"]),
        ]
    )
    assert findings == []


# --- describing the set a relation is tight on ----------------------------


def test_a_tight_set_that_is_the_primes_is_named_as_such():
    """`sigma(n) = n+1` exactly at the primes, recognised without being told.

    This is the difference between a finding and a number: "tight in 12 of 39"
    is data, "tight exactly on the primes" is the characterisation of the
    primes by their divisor sum.
    """
    import sympy as sp

    rows = list(range(2, 41))
    findings = scan_for_regularities(
        [
            Observation(
                name="sigma",
                values=[str(sp.divisor_sigma(n)) for n in rows],
                labels=[str(n) for n in rows],
            ),
            Observation(
                name="n_plus_1",
                values=[str(n + 1) for n in rows],
                labels=[str(n) for n in rows],
            ),
        ]
    )
    saturated = next(
        f for f in findings if f.kind is RegularityKind.SATURATED_BOUND
    )
    assert saturated.character == "tight exactly on: prime"
    assert saturated.witnesses[:5] == ["2", "3", "5", "7", "11"]


# --- a near miss must say WHICH ------------------------------------------


def test_a_near_miss_names_the_exceptions_rather_than_counting_them():
    """"Off by 1" is a number, and this file is about the difference.

    `n - totient(n) >= Omega(n)` is tight exactly on the primes together with
    4. The characteriser had every ingredient -- it computes `extra` and
    `missing` -- and then threw them away at the headline, reporting `closest
    description: prime (off by 1)`. A reader cannot act on that: it says
    something is almost the primes without saying what the difference is, so
    the finding sat in the open ledger as unexplained structure when the
    structure was "and also 4".

    Same failure as `sigma(n) >= n+1, tight in 12 of 39`, one level down.
    """
    universe = [str(n) for n in range(2, 200)]
    tight = [str(n) for n in range(2, 200) if sp.isprime(n)] + ["4"]

    character = characterise(sorted(tight, key=int), universe)
    assert not character.recognised
    assert character.headline == "closest description: prime, and also 4"


def test_a_near_miss_that_is_missing_members_says_which_are_missing():
    """The other direction, and both at once."""
    universe = [str(n) for n in range(2, 200)]
    primes = [n for n in range(2, 200) if sp.isprime(n)]

    short = [str(n) for n in primes if n not in (11, 13)]
    assert characterise(short, universe).headline == (
        "closest description: prime, except 11 and 13"
    )

    both = [str(n) for n in primes if n != 11] + ["9"]
    assert characterise(sorted(both, key=int), universe).headline == (
        "closest description: prime, except 11, and also 9"
    )


def test_a_long_exception_list_is_counted_as_well_as_shown():
    """Truncated for reading, and never counted off the truncation.

    `WitnessCharacter` stores at most eight of each and keeps the true sizes
    beside them, precisely because a count read off a truncated list is wrong
    when the discrepancy is large -- which is when it matters. The headline
    has to respect that: name what it has, and say how many there really are.
    """
    universe = [str(n) for n in range(2, 400)]
    primes = [n for n in range(2, 400) if sp.isprime(n)]
    dropped = primes[:12]
    short = [str(n) for n in primes if n not in dropped]

    headline = characterise(short, universe).headline
    assert "12 in all" in headline, headline
    assert headline.count(",") <= 10, "the list itself must stay short"


def test_naming_the_exceptions_does_not_make_the_finding_explained():
    """A description is not an explanation, and must not close the ledger.

    Saying "prime, and also 4" tells a reader what to go and prove. It is not
    a predicate that named the set, and letting it set `characterised` would
    retire an open question by rewording its headline -- exactly the promotion
    the open ledger exists to refuse.
    """
    universe = [str(n) for n in range(2, 200)]
    tight = sorted(
        [str(n) for n in range(2, 200) if sp.isprime(n)] + ["4"], key=int
    )
    character = characterise(tight, universe)
    assert not character.recognised

    finding = PatternFinding(
        kind=RegularityKind.SATURATED_BOUND,
        statement="n_minus_totient >= Omega",
        columns=["n_minus_totient", "Omega"],
        support=len(tight),
        sampled=len(universe),
        exact=True,
        witnesses=tight,
        character=character.headline,
        characterised="",
    )
    assert finding.unexplained


def test_an_undescribable_tight_set_is_ranked_up_not_dropped():
    """Structure nobody has named is the reason to look at all."""
    # Chosen to dodge the near-miss reporting too: {1, 5, 8} is within two of
    # `n = 1 mod 4`, and the check correctly said so rather than claiming the
    # set was undescribable.
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    findings = scan_for_regularities(
        [
            Observation(
                name="a",
                values=[("7" if int(r) in tight_on else "9") for r in rows],
                labels=rows,
            ),
            Observation(
                name="b",
                values=[("7" if int(r) in tight_on else "2") for r in rows],
                labels=rows,
            ),
        ]
    )
    saturated = next(
        f for f in findings if f.kind is RegularityKind.SATURATED_BOUND
    )
    assert "UNRECOGNISED" in saturated.character
    assert saturated.surprise >= 2


def test_a_tight_set_too_small_to_describe_is_not_boosted():
    """Every two-element set is unnamed. Boosting that promotes coincidence."""
    findings = scan_for_regularities(
        [
            Observation(name="a", values=["3", "9", "9", "9"], labels=["1", "2", "3", "4"]),
            Observation(name="b", values=["3", "2", "2", "2"], labels=["1", "2", "3", "4"]),
        ]
    )
    saturated = next(
        f for f in findings if f.kind is RegularityKind.SATURATED_BOUND
    )
    assert saturated.character == ""


def test_characterise_declines_on_non_integer_rows():
    """Guessing a description for rows it cannot read would be worse."""
    from rh_research_engine.patterns import characterise

    result = characterise(["a", "b"], ["a", "b", "c"])
    assert result.matches == []
    assert "UNRECOGNISED" in result.headline


# --- retiring verified noise ----------------------------------------------


def test_a_rule_retires_only_its_own_columns():
    """Suppressing a KIND would eat a result; suppressing a pair does not."""
    from rh_research_engine.patterns import NoiseGround, NoiseRegistry, NoiseRule

    findings = scan_for_regularities(
        [
            Observation(name="n", values=["1", "2", "3", "4"]),
            Observation(name="n_plus_1", values=["2", "3", "4", "5"]),
            Observation(name="other", values=["5", "6", "7", "8"]),
        ]
    )
    registry = NoiseRegistry()
    assert registry.add(
        NoiseRule(
            columns=["n", "n_plus_1"],
            reason="n_plus_1 is defined as n + 1",
            ground=NoiseGround.CONSTRUCTION,
        )
    )
    suppression = registry.apply(findings)
    assert suppression.removed_total >= 1
    assert all(
        set(f.columns) != {"n", "n_plus_1"} for f in suppression.kept
    )
    # The identically-shaped relation on OTHER columns survives.
    assert any(set(f.columns) == {"n_plus_1", "other"} for f in suppression.kept)


def test_suppression_reports_what_it_removed():
    """A filter that cannot say how much it hid will eventually hide the find."""
    from rh_research_engine.patterns import NoiseGround, NoiseRegistry, NoiseRule

    findings = scan_for_regularities(
        [
            Observation(name="a", values=["1", "2", "3"]),
            Observation(name="b", values=["2", "3", "4"]),
        ]
    )
    registry = NoiseRegistry()
    registry.add(
        NoiseRule(columns=["a", "b"], reason="constructed", ground=NoiseGround.CONSTRUCTION)
    )
    suppression = registry.apply(findings)
    assert suppression.removed == {"constructed": suppression.removed_total}
    assert suppression.removed_total > 0


def test_a_duplicate_rule_is_not_added_twice():
    from rh_research_engine.patterns import NoiseGround, NoiseRegistry, NoiseRule

    registry = NoiseRegistry()
    rule = NoiseRule(columns=["a", "b"], reason="x", ground=NoiseGround.TRIAGED)
    assert registry.add(rule)
    assert not registry.add(rule)
    assert len(registry.rules) == 1


def test_the_registry_round_trips(tmp_path):
    from rh_research_engine.patterns import NoiseGround, NoiseRegistry, NoiseRule

    registry = NoiseRegistry()
    registry.add(
        NoiseRule(
            columns=["psi", "cum_Lambda"],
            reason="psi(x) = sum Lambda(n) is the definition",
            ground=NoiseGround.KNOWN_IDENTITY,
        )
    )
    path = tmp_path / "noise.json"
    registry.save(path)
    again = NoiseRegistry.load(path)
    assert again.rules == registry.rules
    assert again.rules[0].ground is NoiseGround.KNOWN_IDENTITY


def test_a_rule_must_carry_a_reason():
    """"Noise" without one is indistinguishable from an inconvenient result."""
    from pydantic import ValidationError

    from rh_research_engine.patterns import NoiseGround, NoiseRule

    with pytest.raises(ValidationError):
        NoiseRule(columns=["a"], ground=NoiseGround.TRIAGED)


# --- the tight set is the finding, so it is stored whole ------------------


def _sigma_scan(top: int):
    """`sigma` and `n + 1` over n = 2..top-1, and the rows examined."""
    import sympy as sp

    rows = list(range(2, top))
    labels = [str(n) for n in rows]
    columns = [
        Observation(
            name="sigma",
            values=[str(sp.divisor_sigma(n)) for n in rows],
            labels=labels,
        ),
        Observation(
            name="n_plus_1", values=[str(n + 1) for n in rows], labels=labels
        ),
    ]
    return scan_for_regularities(columns), scan_universe(columns)


def test_a_tight_set_larger_than_a_display_is_still_characterised():
    """The primes stay the primes when there are more than a screenful.

    Stored truncated at 24 rows, `sigma(n) >= n+1` over n = 2..199 compared a
    kept 24 against the 46 actual primes, reported UNRECOGNISED -- and was then
    ranked UP for being unexplained. A known identity promoted to open
    structure, by a display limit.
    """
    findings, _ = _sigma_scan(200)
    saturated = next(f for f in findings if f.kind is RegularityKind.SATURATED_BOUND)
    assert len(saturated.witnesses) == 46
    assert saturated.character == "tight exactly on: prime"
    assert saturated.characterised == "prime"
    assert not saturated.unexplained


def test_a_saturated_bound_is_found_whichever_way_the_columns_are_listed():
    """`Omega >= omega`, tight exactly on the squarefree numbers -- both ways.

    Only `left >= right` was tested, so which of the two statements the scan
    could reach was decided by the order the caller happened to pass the
    columns in. Listed as (omega, Omega) it found nothing at all.
    """
    import sympy as sp

    rows = list(range(2, 40))
    labels = [str(n) for n in rows]
    little = Observation(
        name="omega",
        values=[str(len(sp.factorint(n))) for n in rows],
        labels=labels,
    )
    big = Observation(
        name="Omega",
        values=[str(sum(sp.factorint(n).values())) for n in rows],
        labels=labels,
    )

    for columns in ([little, big], [big, little]):
        saturated = [
            f
            for f in scan_for_regularities(columns)
            if f.kind is RegularityKind.SATURATED_BOUND
        ]
        assert len(saturated) == 1, [c.name for c in columns]
        assert saturated[0].statement.startswith("Omega >= omega")
        assert saturated[0].character == "tight exactly on: squarefree"


def test_only_the_true_direction_of_a_bound_is_reported():
    """Testing both directions must not invent the false one."""
    findings = scan_for_regularities(
        [
            Observation(name="small", values=["1", "2", "3", "4"], labels=list("abcd")),
            Observation(name="big", values=["1", "9", "9", "9"], labels=list("abcd")),
        ]
    )
    saturated = [f for f in findings if f.kind is RegularityKind.SATURATED_BOUND]
    assert [f.statement.split(" throughout")[0] for f in saturated] == [
        "big >= small"
    ]


# --- the open ledger ------------------------------------------------------


def _unexplained_ledger():
    """A ledger holding one relation tight on {3, 4, 10} over rows 1..20."""
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    columns = [
        Observation(
            name="a",
            values=[("7" if int(r) in tight_on else "9") for r in rows],
            labels=rows,
        ),
        Observation(
            name="b",
            values=[("7" if int(r) in tight_on else "2") for r in rows],
            labels=rows,
        ),
    ]
    findings = scan_for_regularities(columns)
    ledger = OpenLedger()
    ledger.record(findings, scan_universe(columns))
    return ledger, rows, tight_on


def _relation(rows, tight_on, *, holds=True):
    """Scan the same relation over `rows`, optionally broken at one row.

    Returns what a revisit needs: the findings, the rows examined, and the
    columns examined.
    """
    values_a, values_b = [], []
    for index, row in enumerate(rows):
        tight = int(row) in tight_on
        values_a.append("7" if tight else "9")
        # One row where a < b is enough to destroy `a >= b throughout`.
        values_b.append("7" if tight else ("2" if holds or index else "99"))
    columns = [
        Observation(name="a", values=values_a, labels=list(rows)),
        Observation(name="b", values=values_b, labels=list(rows)),
    ]
    return (
        scan_for_regularities(columns),
        scan_universe(columns),
        scan_columns(columns),
        [],
    )


def test_only_unexplained_findings_are_filed():
    """A named tight set is an answer, not an open question."""
    findings, universe = _sigma_scan(60)
    ledger = OpenLedger()
    assert ledger.record(findings, universe) == []
    assert ledger.entries == []


def test_a_finding_that_holds_on_a_wider_range_is_extended():
    ledger, rows, tight_on = _unexplained_ledger()
    wider = [str(n) for n in range(1, 41)]

    (entry, outcome), = ledger.revisit(*_relation(wider, tight_on))
    assert outcome.verdict is RevisitVerdict.EXTENDED
    assert outcome.universe == 40
    assert entry.open, "surviving a widening is evidence, not an answer"
    assert entry.widest == 40


def test_a_finding_that_stops_holding_is_broken_and_kept():
    """A coincidence of the narrow range -- and the record of it survives."""
    ledger, rows, tight_on = _unexplained_ledger()
    wider = [str(n) for n in range(1, 41)]

    (entry, outcome), = ledger.revisit(*_relation(wider, tight_on, holds=False))
    assert outcome.verdict is RevisitVerdict.BROKEN
    assert not entry.open
    assert entry in ledger.entries, "nothing is deleted"
    assert entry.history[-1].note


def test_a_disjoint_range_refutes_nothing():
    """The whole point of the file.

    Scan rows 1..20, then rows 50..90, and the recorded tight set is absent
    from the new scan -- which reads exactly like a refutation while in fact
    nothing was examined. A comparison that cannot tell "refuted" from "not
    tested" is the failure this repository is built against.
    """
    ledger, _, tight_on = _unexplained_ledger()
    elsewhere = [str(n) for n in range(50, 91)]

    (entry, outcome), = ledger.revisit(*_relation(elsewhere, tight_on))
    assert outcome.verdict is RevisitVerdict.INCOMPARABLE
    assert entry.open, "an untested entry stays open"
    assert "nothing about this finding was tested" in outcome.note


def test_rerunning_the_same_range_is_not_evidence():
    """Reproduction is not a widening, and must not be reported as one."""
    ledger, rows, tight_on = _unexplained_ledger()
    (_, outcome), = ledger.revisit(*_relation(rows, tight_on))
    assert outcome.verdict is RevisitVerdict.INCOMPARABLE


def test_a_tight_set_that_changes_on_the_old_rows_is_contradicted():
    """Recurrence is not extension.

    The relation is still there and the range genuinely widened, but on the
    rows the record covers the tight set is a different set. Reporting that as
    a survival would let the ledger accumulate confirmations of a claim no
    later run ever reproduced.
    """
    ledger, _, tight_on = _unexplained_ledger()
    wider = [str(n) for n in range(1, 41)]

    (entry, outcome), = ledger.revisit(*_relation(wider, tight_on | {7}))
    assert outcome.verdict is RevisitVerdict.CONTRADICTED
    assert "7" in outcome.note
    assert not entry.open


def test_the_predicate_table_is_stable_under_restriction():
    """Why there is no verdict for "solved", stated as a property.

    Every predicate is pointwise, so it commutes with restriction: naming a
    tight set on a wider range would have named it on the narrower one. That is
    what makes a widening able to refute a characterisation and never able to
    confer one -- and it is why an entry closes only by refutation here, with
    "we know what this is" living in the noise registry where it needs a reason.

    A predicate that consulted the RANGE rather than the point -- "unusually
    large", "a record value", "a local maximum" -- would break it, and is
    exactly the kind somebody will reasonably want to add.
    """
    from rh_research_engine.patterns.character import _predicates

    narrow = list(range(2, 20))
    wide = list(range(2, 120))
    for name, predicate in _predicates().items():
        on_narrow = {n for n in narrow if predicate(n)}
        on_wide = {n for n in wide if predicate(n)}
        assert on_wide & set(narrow) == on_narrow, name


def test_a_widening_that_names_the_tight_set_says_so_rather_than_hiding_it():
    """The impossible case is reported, not swallowed.

    It cannot arise while the predicates stay pointwise. If one stops being
    pointwise it will, and a note that reads like an ordinary survival would
    bury the only evidence that the table had changed character.
    """
    from rh_research_engine.patterns.ledger import OpenFinding, judge

    entry = OpenFinding(
        kind=RegularityKind.SATURATED_BOUND,
        columns=["a", "b"],
        statement="a >= b",
        universe=["2", "3", "4"],
        witnesses=["2", "3"],
        character="UNRECOGNISED -- tight on a set matching no known predicate",
    )
    named = PatternFinding(
        kind=RegularityKind.SATURATED_BOUND,
        statement="a >= b",
        columns=["a", "b"],
        support=2,
        sampled=4,
        exact=True,
        witnesses=["2", "3"],
        character="tight exactly on: prime",
        characterised="prime",
    )
    outcome = judge(entry, [named], ["2", "3", "4", "5"], ["a", "b"], [])
    assert outcome.verdict is RevisitVerdict.EXTENDED
    assert "cannot do without having named it here" in outcome.note


def test_a_closed_entry_is_not_reopened_by_a_narrower_run():
    """A narrow run must not overturn a wide one."""
    ledger, rows, tight_on = _unexplained_ledger()
    wider = [str(n) for n in range(1, 41)]
    ledger.revisit(*_relation(wider, tight_on, holds=False))
    assert ledger.entries[0].verdict is RevisitVerdict.BROKEN

    assert ledger.revisit(*_relation(wider, tight_on)) == []
    assert ledger.entries[0].verdict is RevisitVerdict.BROKEN


def test_the_first_observation_is_never_overwritten():
    """It is the only thing a revisit has to compare against."""
    ledger, rows, tight_on = _unexplained_ledger()
    before = list(ledger.entries[0].witnesses)

    wider = [str(n) for n in range(1, 41)]
    findings, universe, _, _ = _relation(wider, tight_on | {33})
    assert ledger.record(findings, universe) == []
    assert ledger.entries[0].witnesses == before
    assert len(ledger.entries) == 1


def test_a_ledger_survives_a_round_trip(tmp_path):
    ledger, rows, tight_on = _unexplained_ledger()
    wider = [str(n) for n in range(1, 41)]
    ledger.revisit(*_relation(wider, tight_on))

    path = tmp_path / "open.json"
    ledger.save(path)
    assert OpenLedger.load(path) == ledger
    assert OpenLedger.load(tmp_path / "absent.json").entries == []


def test_a_scan_that_never_measured_the_columns_refutes_nothing():
    """The range check, in the second dimension.

    Pointed at the corpus over a wider range but a shorter column list, the
    ledger reported `omega >= Mertens` BROKEN having evaluated neither column.
    A relation absent from a scan that never measured it is not refuted; it was
    not looked at, and the two must not produce the same verdict.
    """
    ledger, _, tight_on = _unexplained_ledger()
    wider = [str(n) for n in range(1, 41)]
    findings, universe, _, _ = _relation(wider, tight_on)

    (entry, outcome), = ledger.revisit(findings, universe, ["something", "else"], [])
    assert outcome.verdict is RevisitVerdict.INCOMPARABLE
    assert "did not measure" in outcome.note
    assert entry.open


def test_a_column_the_scan_dropped_counts_as_unmeasured():
    """What was EXAMINED, not what was handed over.

    A column whose values will not parse exactly is dropped before anything is
    compared. Counting it as measured would let an unparseable column refute
    the findings that mention it.
    """
    rows = [str(n) for n in range(1, 21)]
    columns = [
        Observation(name="fine", values=[str(n) for n in range(1, 21)], labels=rows),
        Observation(name="broken", values=["not a number"] * 20, labels=rows),
    ]
    assert scan_columns(columns) == ["fine"]


# --- one quantity, one finding --------------------------------------------


def _two_names_for_one_quantity(extra_names: list[str]) -> list[Observation]:
    """`mertens` under however many names, plus a column to relate it to.

    The corpus really does carry two: `Mertens` from the sieve and `cum_mu`
    accumulated from sympy's mobius, kept apart on purpose so that
    `Mertens = cum_mu` is a cross-check between two implementations rather
    than one array compared against itself.
    """
    rows = [str(n) for n in range(1, 21)]
    mertens = ["1", "0", "-1", "-1", "-2", "-1", "-2", "-2", "-2", "-1",
               "-2", "-2", "-3", "-2", "-1", "-1", "-2", "-2", "-3", "-3"]
    absolute = [value.lstrip("-") for value in mertens]
    columns = [Observation(name="abs_mertens", values=absolute, labels=rows)]
    columns += [
        Observation(name=name, values=mertens, labels=rows, requested=name == "Mertens")
        for name in extra_names
    ]
    return columns


def test_one_quantity_under_two_names_is_one_finding():
    """Two names for one column is not two pieces of evidence.

    `abs_Mertens >= Mertens` and `abs_Mertens >= cum_mu` reached the open
    ledger as separate unexplained findings, with byte-identical witnesses,
    universe, surprise and character -- the same relation, over the same rows,
    counted twice because the quantity has two names. Every relation involving
    `Mertens` was reported twice, so the size of the ledger of things nothing
    explains was set by how many implementations the sweep happened to carry.
    """
    findings = scan_for_regularities(_two_names_for_one_quantity(["Mertens", "cum_mu"]))
    bounds = [f for f in findings if f.kind is RegularityKind.SATURATED_BOUND]
    assert len(bounds) == 1, [f.statement for f in bounds]
    assert bounds[0].columns == ["abs_mertens", "Mertens"]
    assert bounds[0].aliases == {"Mertens": ["cum_mu"]}


def test_the_agreement_between_two_implementations_is_still_reported():
    """Deduplicating must not eat the cross-check that motivated it.

    `cum_mu` exists precisely so that agreeing with `Mertens` means something.
    Collapsing the two and saying nothing would hide the one finding that
    validates the sweep, and "merged silently" and "never compared" would again
    share a verdict.
    """
    findings = scan_for_regularities(_two_names_for_one_quantity(["Mertens", "cum_mu"]))
    agreements = [f for f in findings if f.kind is RegularityKind.EXACT_EQUALITY]
    assert len(agreements) == 1
    assert set(agreements[0].columns) == {"Mertens", "cum_mu"}
    assert "20 examined rows" in agreements[0].statement


def test_a_third_name_for_the_same_quantity_changes_nothing():
    """The sharp form: the count must not move when a name is added.

    A third implementation, added as a further cross-check, would have tripled
    every relation involving it. Whether a quantity is carried once or three
    times is a fact about the sweep's plumbing, and it must not reach the
    findings.
    """
    two = scan_for_regularities(_two_names_for_one_quantity(["Mertens", "cum_mu"]))
    three = scan_for_regularities(
        _two_names_for_one_quantity(["Mertens", "cum_mu", "mertens_again"])
    )
    assert [f.statement for f in two if f.kind is not RegularityKind.EXACT_EQUALITY] == [
        f.statement for f in three if f.kind is not RegularityKind.EXACT_EQUALITY
    ]
    agreement, = [f for f in three if f.kind is RegularityKind.EXACT_EQUALITY]
    assert agreement.columns == ["Mertens", "cum_mu", "mertens_again"]


def test_the_surviving_name_does_not_depend_on_the_order_of_the_list():
    """The `omega, Omega` bug, in the second dimension.

    A representative picked as "whichever was listed first" would put a
    different name into the statement, the ledger key and the noise rule
    depending on how the columns were assembled -- and a noise rule naming
    `cum_mu` would stop matching the day the list was reordered.
    """
    forwards = scan_for_regularities(_two_names_for_one_quantity(["Mertens", "cum_mu"]))
    backwards = scan_for_regularities(_two_names_for_one_quantity(["cum_mu", "Mertens"]))
    assert [f.statement for f in forwards] == [f.statement for f in backwards]


def _relation_under_two_names(rows, tight_on):
    """The same relation, with the lower column carried under two names."""
    values_a, values_b = [], []
    for row in rows:
        tight = int(row) in tight_on
        values_a.append("7" if tight else "9")
        values_b.append("7" if tight else "2")
    columns = [
        Observation(name="a", values=values_a, labels=list(rows)),
        Observation(name="b", values=values_b, labels=list(rows), requested=True),
        Observation(name="b_again", values=list(values_b), labels=list(rows)),
    ]
    return (
        scan_for_regularities(columns),
        scan_universe(columns),
        scan_columns(columns),
    )


def _ledger_recorded_under(name, rows, tight_on):
    """A ledger holding `a >= <name>`, tight on `tight_on` over `rows`."""
    columns = [
        Observation(
            name="a",
            values=[("7" if int(r) in tight_on else "9") for r in rows],
            labels=list(rows),
        ),
        Observation(
            name=name,
            values=[("7" if int(r) in tight_on else "2") for r in rows],
            labels=list(rows),
        ),
    ]
    findings = scan_for_regularities(columns)
    ledger = OpenLedger()
    assert ledger.record(findings, scan_universe(columns)), "fixture recorded nothing"
    return ledger


def test_a_relation_reported_under_one_of_its_names_refutes_nothing():
    """Deduplicating names must not read as refuting the duplicate.

    The ledger holds `abs_Mertens >= cum_mu`. Reporting that relation once,
    under `Mertens`, makes it absent from the scan by column name -- and
    absence is the BROKEN verdict. Nothing about the mathematics changed: the
    relation is exactly as true, over exactly the same rows. A record turned
    REFUTED by a change in how findings are NAMED is the failure this ledger
    exists to prevent, wearing a different hat.
    """
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    ledger = _ledger_recorded_under("b_again", rows, tight_on)
    assert ledger.entries[0].columns == ["a", "b_again"]

    wider = [str(n) for n in range(1, 41)]
    findings, universe, columns = _relation_under_two_names(wider, tight_on)
    assert any(f.aliases for f in findings), "the fixture must exercise an alias"
    assert not any(
        set(f.columns) == {"a", "b_again"} for f in findings
    ), "the relation must no longer be reported under the recorded name"

    (entry, outcome), = ledger.revisit(findings, universe, columns, [])
    assert outcome.verdict is RevisitVerdict.EXTENDED, outcome.note
    assert "b" in outcome.note and "b_again" in outcome.note, (
        "a verdict reached under a different name has to say so: "
        f"{outcome.note!r}"
    )
    assert entry.open


def test_a_name_that_was_never_measured_is_still_incomparable():
    """And the alias must not turn INCOMPARABLE into a survival.

    Matching modulo aliases widens what counts as the recorded relation. If it
    widened far enough to cover a column the scan never looked at, it would
    manufacture evidence out of a name -- the entry below is judged by a scan
    that never carried `b_again` at all.
    """
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    ledger = _ledger_recorded_under("b_again", rows, tight_on)

    wider = [str(n) for n in range(1, 41)]
    (entry, outcome), = ledger.revisit(*_relation(wider, tight_on))
    assert outcome.verdict is RevisitVerdict.INCOMPARABLE
    assert "did not measure" in outcome.note
    assert entry.open


def test_retiring_a_finding_as_noise_does_not_refute_it():
    """Explaining a relation must not read as refuting it.

    The third case the ledger was missing. It separates REFUTED from NOT
    TESTED with great care, and then a finding retired by a noise rule is
    simply absent from the scan it is judged against -- so writing down WHY a
    relation is an artifact recorded "was a regularity of the narrower range"
    about a relation that is untouched and perfectly true.

    It would have fired on the two rules retiring `abs_mu >= mu` and
    `abs_Mertens >= Mertens`: both are `|x| >= x`, both hold at every range
    there is, and both had an entry in the open ledger waiting to be told so.
    """
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    ledger = _ledger_recorded_under("b", rows, tight_on)

    registry = NoiseRegistry(
        rules=[
            NoiseRule(
                columns=["a", "b"],
                reason="a was defined as b rounded up",
                ground=NoiseGround.CONSTRUCTION,
            )
        ]
    )
    wider = [str(n) for n in range(1, 41)]
    findings, universe, columns, _ = _relation(wider, tight_on)
    suppression = registry.apply(findings)
    assert not [
        f for f in suppression.kept if set(f.columns) == {"a", "b"}
    ], "the fixture must actually retire the relation"

    (entry, outcome), = ledger.revisit(
        suppression.kept, universe, columns, suppression.retired
    )
    assert outcome.verdict is RevisitVerdict.RETIRED
    assert "rounded up" in outcome.note, "the rule's reason is the whole point"
    assert not entry.open


def test_a_retirement_is_read_before_the_range_is():
    """A written reason does not depend on the range that happened to run.

    INCOMPARABLE says nothing was tested. A retirement is not a test and has
    nothing to do with what this scan covered, so a narrower run must still
    report the entry as explained rather than as unexamined.
    """
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    ledger = _ledger_recorded_under("b", rows, tight_on)

    registry = NoiseRegistry(
        rules=[
            NoiseRule(
                columns=["a", "b"],
                reason="a was defined as b rounded up",
                ground=NoiseGround.CONSTRUCTION,
            )
        ]
    )
    narrower = [str(n) for n in range(1, 11)]
    findings, universe, columns, _ = _relation(narrower, tight_on)
    suppression = registry.apply(findings)

    (_, outcome), = ledger.revisit(
        suppression.kept, universe, columns, suppression.retired
    )
    assert outcome.verdict is RevisitVerdict.RETIRED


def test_a_scan_with_no_registry_behind_it_still_judges():
    """Nothing retired is stated, never defaulted.

    `retired` is required for the same reason `columns` is: both change the
    verdict, and neither can be recovered from the surviving finding afterwards
    -- so a caller that leaves one out gets a wrong answer with no way to tell.
    An empty list is a caller saying nothing was retired.
    """
    rows = [str(n) for n in range(1, 21)]
    tight_on = {3, 4, 10, 15, 16}
    ledger = _ledger_recorded_under("b", rows, tight_on)
    wider = [str(n) for n in range(1, 41)]

    (_, outcome), = ledger.revisit(*_relation(wider, tight_on))
    assert outcome.verdict is RevisitVerdict.EXTENDED


def test_no_noise_ground_is_spelled_like_a_confidence():
    """A triage ground must never be readable as an epistemic status.

    The grounds say how somebody decided a finding was noise. `Confidence`
    says what the engine holds about the mathematics. They travel as bare
    strings in JSON, so a ground spelled `proved` would be indistinguishable
    from the confidence of that name by anything reading the record -- and
    `ELEMENTARY` was very nearly called exactly that, for a rule whose reason
    is a three-line argument.

    Retiring a finding is a filing decision. It has never established
    anything, and no value in this vocabulary may suggest it has.
    """
    from rh_research_engine.contracts.epistemic import Confidence
    from rh_research_engine.patterns import NoiseGround

    collisions = {g.value for g in NoiseGround} & {c.value for c in Confidence}
    assert not collisions, f"a noise ground reads as a confidence: {collisions}"


def test_retiring_a_finding_does_not_touch_its_confidence():
    """The other half: suppression files, it does not promote or demote.

    A finding that survives the filter and one that is retired by it are the
    same record afterwards. If retirement could move a confidence, the
    registry would be an epistemic mechanism wearing the shape of a filter.
    """
    from rh_research_engine.patterns import NoiseGround, NoiseRegistry, NoiseRule

    finding = PatternFinding(
        kind=RegularityKind.SATURATED_BOUND,
        statement="a >= b",
        columns=["a", "b"],
        support=3,
        sampled=9,
        exact=True,
    )
    before = finding.confidence
    registry = NoiseRegistry(
        rules=[
            NoiseRule(
                columns=["a", "b"],
                reason="settled by a short argument, written out here",
                ground=NoiseGround.ELEMENTARY,
            )
        ]
    )
    suppression = registry.apply([finding])
    retired, = suppression.retired
    assert retired.ground is NoiseGround.ELEMENTARY
    assert retired.finding.confidence is before
    assert before not in RIGOROUS
