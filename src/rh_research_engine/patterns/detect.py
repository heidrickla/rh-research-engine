"""Audit a premise, and scan measurements for exact structure.

The two halves of the move described in `models`: refuse to fit a quantity that
has nothing in it, and look at everything that was measured rather than the one
thing the task named.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from .character import characterise
from .models import (
    Observation,
    PatternFinding,
    PremiseAudit,
    PremiseVerdict,
    RegularityKind,
)

#: Below this many points, nothing is concluded.
#:
#: Two points make any pair of columns "equal in 2 of 2" and any value
#: "constant". A regularity worth reporting has to survive more cases than it
#: takes to state, and three is the smallest number where that is true.
MINIMUM_SAMPLE = 3


def _exact(value: str) -> Fraction | None:
    """Parse a measurement exactly, or report that it is not exact.

    Exactness is the property being measured. A saturated bound that holds to
    1e-12 and one that holds identically are different findings, and going
    through a float erases the difference before anything can look at it.
    """
    text = value.strip()
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        # Decimal strings are exact; `Fraction` accepts them directly, so
        # reaching here means it is a float literal in some other spelling.
        return Fraction(float(text))
    except (ValueError, OverflowError):
        return None


def _column(observation: Observation) -> list[Fraction] | None:
    parsed = [_exact(value) for value in observation.values]
    return None if any(item is None for item in parsed) else parsed  # type: ignore[return-value]


def audit_premise(
    quantity: str,
    values: Sequence[str],
    *,
    target: str | None = None,
) -> PremiseAudit:
    """Is there anything in this quantity to fit?

    A quantity that never moves over the whole computable range has nothing to
    fit, and saying so is a finding rather than a failure to produce one. This
    is the check the Anthropic sub-agent ran before any computation, and it is
    what closed its assigned route: the brief's quantity was identically zero
    everywhere it could be evaluated.

    `target` is the value the task expected. When the quantity equals it
    everywhere, fitting recovers the assumption instead of testing it, which is
    DEGENERATE rather than LIVE -- a distinction worth keeping, because such a
    fit looks like a confirmation.
    """
    parsed = [_exact(value) for value in values]
    usable = [item for item in parsed if item is not None]
    if len(usable) < MINIMUM_SAMPLE:
        return PremiseAudit(
            quantity=quantity,
            verdict=PremiseVerdict.INCONCLUSIVE,
            sampled=len(usable),
            evidence=(
                f"{len(usable)} usable value(s); {MINIMUM_SAMPLE} are needed "
                "before constancy means anything"
            ),
        )

    # The target check comes FIRST. A quantity pinned to the value the task
    # expected is both constant and degenerate, and DEGENERATE is the more
    # useful of the two verdicts -- it says the fit would recover the
    # assumption, where EMPTY only says the fit has nothing to work with.
    # Testing constancy first made this branch unreachable.
    if target is not None:
        expected = _exact(target)
        if expected is not None and all(item == expected for item in usable):
            return PremiseAudit(
                quantity=quantity,
                verdict=PremiseVerdict.DEGENERATE,
                sampled=len(usable),
                evidence=(
                    f"equals the expected {expected} at every one of "
                    f"{len(usable)} sampled points, so a fit recovers the "
                    "assumption rather than testing it"
                ),
            )

    distinct = set(usable)
    if len(distinct) == 1:
        only = next(iter(distinct))
        zero = only == 0
        return PremiseAudit(
            quantity=quantity,
            verdict=PremiseVerdict.EMPTY,
            sampled=len(usable),
            evidence=(
                f"identically {'zero' if zero else only} across all "
                f"{len(usable)} sampled points -- there is nothing to fit"
            ),
        )

    return PremiseAudit(
        quantity=quantity,
        verdict=PremiseVerdict.LIVE,
        sampled=len(usable),
        evidence=(
            f"takes {len(distinct)} distinct values across {len(usable)} points"
        ),
    )


#: Kinds that relate two columns. Everything else describes one.
RELATION_KINDS = frozenset(
    {
        RegularityKind.EXACT_EQUALITY,
        RegularityKind.SATURATED_BOUND,
        RegularityKind.CONSTANT_RATIO,
        RegularityKind.CONSTANT_DIFFERENCE,
    }
)


def _labels(observation: Observation, count: int) -> list[str]:
    if observation.labels and len(observation.labels) == count:
        return list(observation.labels)
    return [str(index) for index in range(count)]


def _usable(
    observations: Sequence[Observation],
) -> list[tuple[Observation, list[Fraction]]]:
    columns: list[tuple[Observation, list[Fraction]]] = []
    for observation in observations:
        values = _column(observation)
        if values is not None and len(values) >= MINIMUM_SAMPLE:
            columns.append((observation, values))
    return columns


def scan_columns(observations: Sequence[Observation]) -> list[str]:
    """The columns a scan of these observations actually examines.

    Not the names that were passed in: a column whose values will not parse
    exactly, or which is shorter than `MINIMUM_SAMPLE`, is dropped before
    anything is compared. A relation over a dropped column was not tested, and
    the open ledger has to be able to tell that from a relation that failed.
    """
    return [observation.name for observation, _ in _usable(observations)]


def scan_universe(observations: Sequence[Observation]) -> list[str]:
    """The rows a scan of these observations examines.

    Public, and the ONLY definition of it, because the open ledger tests
    whether one scan's range contains another's -- and a caller that worked out
    the range for itself would be comparing its own arithmetic against the
    scan's. That is the same mistake as reading a formula under a second
    name-resolution policy: the two agree until the day they do not, and
    nothing downstream can tell.
    """
    columns = _usable(observations)
    return _labels(columns[0][0], len(columns[0][1])) if columns else []


def _clone_groups(
    columns: Sequence[tuple[Observation, list[Fraction]]],
) -> list[list[Observation]]:
    """Columns carrying the same values on every examined row, grouped.

    Ordered so the representative comes first, by a rule that does not depend
    on how the columns were assembled: a requested column before a derived
    one, then alphabetically. "Whichever was listed first" would put a
    different name into the statement, the ledger key and any noise rule
    written against it, and reordering the sweep's own table would silently
    stop those rules matching.
    """
    grouped: dict[tuple[Fraction, ...], list[Observation]] = {}
    for observation, values in columns:
        grouped.setdefault(tuple(values), []).append(observation)
    return [
        sorted(names, key=lambda o: (not o.requested, o.name))
        for names in grouped.values()
        if len(names) > 1
    ]


def _agreement_finding(group: list[Observation], sampled: int) -> PatternFinding:
    """The coincidence itself, reported once, naming every name.

    Deduplicating must not eat the cross-check that motivates it. `cum_mu` is
    accumulated from sympy's mobius rather than read off the sieve precisely so
    that agreeing with `Mertens` means two implementations agree; collapsing
    the two and saying nothing would hide the finding that validates the sweep.

    Stated as agreement ON THE EXAMINED ROWS, never as equality. Two
    implementations that agree this far may part company higher up, and at that
    point they are two quantities again -- which the wider scan will show, and
    can only show if this record did not overstate what it saw.
    """
    names = [observation.name for observation in group]
    return PatternFinding(
        kind=RegularityKind.EXACT_EQUALITY,
        statement=f"{', '.join(names)} agree at all {sampled} examined rows",
        columns=names,
        support=sampled,
        sampled=sampled,
        exact=True,
        from_unrequested=not all(observation.requested for observation in group),
        evidence={"names": len(names)},
    )


def _single_column_findings(
    observation: Observation, values: list[Fraction]
) -> list[PatternFinding]:
    findings: list[PatternFinding] = []
    sampled = len(values)

    integral = sum(1 for value in values if value.denominator == 1)
    if integral == sampled and any(value != 0 for value in values):
        findings.append(
            PatternFinding(
                kind=RegularityKind.INTEGRALITY,
                statement=f"{observation.name} is an integer in every case",
                columns=[observation.name],
                support=integral,
                sampled=sampled,
                exact=True,
                from_unrequested=not observation.requested,
                evidence={"distinct": len({str(v) for v in values})},
            )
        )

    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    if sampled >= MINIMUM_SAMPLE and (positive == sampled or negative == sampled):
        sign = "positive" if positive == sampled else "negative"
        findings.append(
            PatternFinding(
                kind=RegularityKind.SIGN_CONSTANCY,
                statement=f"{observation.name} is {sign} in every case",
                columns=[observation.name],
                support=sampled,
                sampled=sampled,
                exact=True,
                from_unrequested=not observation.requested,
                evidence={"sign": sign},
            )
        )
    return findings


def _pair_findings(
    left: Observation,
    left_values: list[Fraction],
    right: Observation,
    right_values: list[Fraction],
) -> list[PatternFinding]:
    findings: list[PatternFinding] = []
    sampled = len(left_values)
    unrequested = not (left.requested and right.requested)

    tight = [
        index
        for index, (a, b) in enumerate(zip(left_values, right_values, strict=True))
        if a == b
    ]
    equal = len(tight)
    rows = _labels(left, sampled)
    if equal == sampled:
        findings.append(
            PatternFinding(
                kind=RegularityKind.EXACT_EQUALITY,
                statement=f"{left.name} = {right.name} in every case",
                columns=[left.name, right.name],
                support=equal,
                sampled=sampled,
                exact=True,
                from_unrequested=unrequested,
            )
        )
    elif equal:
        # BOTH DIRECTIONS. `a >= b` and `b >= a` are different statements and
        # only one of them can hold once the columns differ somewhere, so
        # testing one is testing whichever the caller happened to list first.
        # Listed as `omega, Omega` the scan found nothing; listed the other way
        # round it found `Omega >= omega, tight exactly on the squarefree
        # numbers`. Same columns, same range, same mathematics -- and a
        # characterisation that appeared or vanished on the order of a list.
        for over, under in ((left, right), (right, left)):
            above = left_values if over is left else right_values
            below = right_values if over is left else left_values
            if not all(a >= b for a, b in zip(above, below, strict=True)):
                continue
            # A bound that is sometimes tight is ordinary. One that is ALWAYS
            # tight is an identity nobody has written down, and that is the
            # finding worth escalating -- the shape of "equality in 13 of 13".
            findings.append(
                PatternFinding(
                    kind=RegularityKind.SATURATED_BOUND,
                    statement=(
                        f"{over.name} >= {under.name} throughout, with "
                        f"equality in {equal} of {sampled}"
                    ),
                    columns=[over.name, under.name],
                    support=equal,
                    sampled=sampled,
                    exact=True,
                    from_unrequested=unrequested,
                    # COMPLETE, never truncated. Stored at 24 rows, the primes
                    # up to 200 characterised as UNRECOGNISED -- 46 tight cases
                    # compared against a stored 24 -- and were then ranked UP
                    # for being unexplained. Truncation belongs at the display,
                    # and the display already does it.
                    witnesses=[rows[index] for index in tight],
                    # Peaks when the tight set is a proper, SUBSTANTIAL subset.
                    # Tight everywhere is an identity and is reported as one;
                    # tight once is a coincidence; tight on a third of the
                    # cases is a characterisation of that third -- which is
                    # what `sigma(n) = n+1 exactly at the primes` looks like
                    # from the outside.
                    surprise=min(equal, sampled - equal),
                    evidence={"slack_cases": sampled - equal},
                )
            )
            # At most one direction survives: both would make the columns equal
            # everywhere, which is the branch above.
            break

    differences = {a - b for a, b in zip(left_values, right_values, strict=True)}
    if len(differences) == 1 and equal != sampled:
        (only,) = differences
        findings.append(
            PatternFinding(
                kind=RegularityKind.CONSTANT_DIFFERENCE,
                statement=f"{left.name} - {right.name} = {only} in every case",
                columns=[left.name, right.name],
                support=sampled,
                sampled=sampled,
                exact=True,
                from_unrequested=unrequested,
                evidence={"difference": str(only)},
            )
        )

    if all(value != 0 for value in right_values):
        ratios = {a / b for a, b in zip(left_values, right_values, strict=True)}
        if len(ratios) == 1:
            (only,) = ratios
            if only != 1:
                findings.append(
                    PatternFinding(
                        kind=RegularityKind.CONSTANT_RATIO,
                        statement=f"{left.name} / {right.name} = {only} in every case",
                        columns=[left.name, right.name],
                        support=sampled,
                        sampled=sampled,
                        exact=True,
                        from_unrequested=unrequested,
                        evidence={"ratio": str(only)},
                    )
                )
    return findings


def scan_for_regularities(observations: Sequence[Observation]) -> list[PatternFinding]:
    """Look at every column and every pair, not the one that was asked about.

    This is the mechanical form of measuring what the task did not request.
    The Anthropic sub-agent's script verified the facts its brief asserted and
    also measured the sign of cross terms, which the brief had stated but never
    asked to have checked; that unrequested column carried the theorem. A scan
    that only examines the named quantity cannot find that, however carefully
    it examines it.

    Findings are returned unranked and unfiltered. Deciding which is worth
    chasing is a judgement, and a scan that quietly drops the odd-looking ones
    is a scan that cannot surprise anybody.

    ONE QUANTITY IS ONE FINDING, however many names carry it. This is not the
    filter the paragraph above refuses: nothing is judged boring and nothing is
    dropped. Two columns holding the same values on every examined row are one
    column, checkably, and reporting their relations twice makes the count of
    open questions a fact about the sweep's plumbing rather than about the
    mathematics. `Mertens` and `cum_mu` differ only in which implementation
    produced them, and every relation involving the quantity reached the ledger
    twice, with byte-identical witnesses, universe, surprise and character.

    Nothing is lost by it. The agreement itself is reported -- it is the
    cross-check between the two implementations, and the reason `cum_mu` is
    computed a second way at all -- and every other finding carries the names
    it equally holds under, so a record made under one of them can still be
    recognised. `ledger._keys` is the other half of that: without it, reporting
    a relation once would REFUTE the copy of it somebody had already filed.
    """
    examined = _usable(observations)

    # ONE QUANTITY, ONE FINDING. `Mertens` from the sieve and `cum_mu`
    # accumulated from sympy's mobius hold the same values on every row, so
    # every relation involving the quantity was found twice -- and reached the
    # open ledger twice, with byte-identical witnesses, universe, surprise and
    # character. Two names is not two pieces of evidence, and the size of the
    # ledger of things nothing explains must not be set by how many
    # implementations the sweep happens to carry.
    groups = _clone_groups(examined)
    aliases = {group[0].name: [o.name for o in group[1:]] for group in groups}
    dropped = {o.name for group in groups for o in group[1:]}
    columns = [item for item in examined if item[0].name not in dropped]

    measured = {observation.name: values for observation, values in examined}
    findings: list[PatternFinding] = []
    for observation, values in columns:
        findings.extend(_single_column_findings(observation, values))

    for index, (left, left_values) in enumerate(columns):
        for right, right_values in columns[index + 1 :]:
            if len(left_values) != len(right_values):
                continue
            findings.extend(_pair_findings(left, left_values, right, right_values))

    # Every relation carries the names it equally holds under. The agreement
    # findings are added AFTERWARDS and never get this treatment: they already
    # name the whole group, and an alias on top of that would say a coincidence
    # between two names also holds between one of them and itself.
    for finding in findings:
        held = {name: aliases[name] for name in finding.columns if name in aliases}
        if held:
            finding.aliases = held
    findings = [
        _agreement_finding(group, len(measured[group[0].name])) for group in groups
    ] + findings

    # A relation over many DISTINCT values constrains more than one over a
    # handful: `a = b` across thirty-nine different values is a claim, across
    # three it is close to an accident. Saturated bounds are already scored on
    # the shape of their tight set and keep that score.
    # Over EVERY examined column, not the collapsed set: an agreement finding
    # names the aliases too, and a name missing from this table scores zero.
    spread = {
        observation.name: len(set(values)) for observation, values in examined
    }
    universe = scan_universe(observations)
    for finding in findings:
        if finding.kind in RELATION_KINDS and finding.surprise == 0:
            finding.surprise = min(spread.get(name, 0) for name in finding.columns)

        # Describe the set the relation is tight on, when that is a proper
        # subset. A tight set nothing describes is the interesting case, so it
        # is ranked UP rather than dropped -- unexplained structure is what
        # looking for surprises means.
        # Only when the tight set is big enough to HAVE a description.
        # "Unrecognised" over one or two rows says nothing -- every
        # two-element set is unnamed -- and boosting it promotes coincidence.
        if (
            finding.witnesses
            and not finding.unanimous
            and len(finding.witnesses) >= MINIMUM_SAMPLE
        ):
            character = characterise(finding.witnesses, universe)
            finding.character = character.headline
            if character.recognised:
                finding.characterised = character.matches[0].predicate
            else:
                finding.surprise += 2
    return findings


def escalate(
    findings: Sequence[PatternFinding], *, minimum_surprise: int = 2
) -> list[PatternFinding]:
    """The findings worth a human's attention, most constraining first.

    Unanimity is the wrong filter, and the first blind run over the corpus's
    own functions showed why: twenty-four of twenty-eight unanimous findings
    were integrality and sign-constancy of columns that are integer and
    positive by construction, while the two that mattered were excluded for
    not being unanimous. `sigma(n) >= n+1, tight in 12 of 39` IS the
    characterisation of the primes; a filter that drops it in favour of
    "sigma is positive in every case" is not a discovery tool.

    So: rank by how much a relation constrains, drop properties of a single
    column by default, and put the unrequested first among equals -- the
    column nobody asked for is the one worth looking at.
    """
    return sorted(
        (finding for finding in findings if finding.surprise >= minimum_surprise),
        key=lambda finding: (
            not finding.from_unrequested,
            -finding.surprise,
            -finding.sampled,
        ),
    )
