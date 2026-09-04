"""The findings nothing has explained yet, kept so a wider range can judge them.

THE COUNTERPART TO THE NOISE REGISTRY. `noise.py` retires a finding that has
been explained. This keeps the ones that have NOT been -- a relation tight on a
set no predicate names -- because that is the case the scan exists to produce
and the case a terminal session throws away.

WHY A RECORD RATHER THAN A LONGER RUN. An exact regularity over 38 rows is weak
evidence and the same regularity over 198 is much better evidence, but they are
not the same observation and the difference between them is the whole content.
Re-running at a wider range and printing the result again answers nothing; what
answers something is asking whether the SECOND run agrees with what the first
one recorded. That question cannot be asked unless the first run wrote down the
range it saw and the set it saw it on.

WHAT A REVISIT CAN CONCLUDE, and why each of these is a result:

  * EXTENDED -- held on a strictly wider range and agrees with the record
    everywhere the record spoke. The structure survived a widening.
  * CONTRADICTED -- still present, but restricted to the old range the new
    tight set is NOT the old one. The record describes something the wider
    range does not confirm.
  * BROKEN -- the relation is gone. A coincidence of the narrow range, and
    finding that out is worth more than never having asked.
  * INCOMPARABLE -- the new range does not strictly contain the old one, so
    none of the above was tested.
  * RETIRED -- a noise rule explained it. The relation holds; it is not news.
    This one does not come from the range at all, and is here because without
    it the act of writing down WHY a relation is an artifact made the relation
    disappear from the scan, which is the BROKEN verdict. Explaining and
    refuting are opposite outcomes and they shared a verdict.

THERE IS NO "EXPLAINED". It was written, and it could not fire. The predicates
in `character.py` are pointwise, so they commute with restriction: for any
predicate P and nested ranges U1 < U2, `(P inter U2) inter U1 = P inter U1`. An
entry is only filed here because nothing named its tight set on U1; if a
widening leaves the old rows alone -- which is exactly what EXTENDED means --
then no predicate can name the tight set on U2 either, because naming it there
would have named it here. A widening can REFUTE a characterisation and can
never confer one.

So a widening never answers the question, and the place a finding gets
explained is the noise registry, which demands a reason for it. Two mechanisms
for "we know what this is" would have meant one of them recording an
explanation nobody had to justify. `test_the_predicate_table_is_stable_under_
restriction` guards the property this rests on: a predicate that consulted the
range rather than the point -- "unusually large", "a record value", "a local
maximum" -- would break it, and is the kind of predicate somebody will
reasonably want to add.

THE LAST ONE IS THE POINT OF THE FILE. Scan n = 2..40, then scan n = 50..90,
and the recorded tight set is absent from the new one -- which reads exactly
like a refutation while in fact nothing whatsoever was examined. A comparison
that cannot tell "refuted" from "not tested" is the failure this repository is
built against.

So BOTH the range and the columns are ARGUMENTS to the judgement rather than
things read back off whichever finding survived. A relation that disappears
tells you nothing until you know the new scan covered the old rows AND
measured the old columns, and asking the vanished finding what it was measured
over is not an option precisely in the case that matters. The columns were the
second half of this, and were missing from the first version: pointed at the
corpus over a wider range but a shorter column list, the ledger reported
`omega >= Mertens` as BROKEN, having never evaluated either column. Note that
what counts is the columns the scan EXAMINED, not the ones it was handed --
`scan_columns` drops what will not parse, and a relation over a dropped column
was not tested either.

NOTHING IS DELETED. A broken entry keeps its history and the range it died on,
for the same reason a noise rule carries a reason: a ledger that quietly drops
what did not survive will file the same coincidence again next month, and
"held on 38 rows, gone by 198" is calibration for every other 38-row finding.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import PatternFinding, RegularityKind
from .noise import Retirement


class RevisitVerdict(StrEnum):
    """What a wider range said about an open finding."""

    #: Held on a strictly wider range, agreeing with the record on the old one,
    #: and still nothing names the tight set. Open, and better evidenced.
    EXTENDED = "extended"
    #: Still present, but the new tight set restricted to the old range is not
    #: the old tight set. The record describes something unconfirmed.
    CONTRADICTED = "contradicted"
    #: The relation does not hold on the wider range at all.
    BROKEN = "broken"
    #: The new range does not strictly contain the old one. Nothing was tested,
    #: and saying so is not the same as saying the finding failed.
    INCOMPARABLE = "incomparable"
    #: A noise rule retired the relation. It still holds; somebody wrote down
    #: what it is. Distinct from BROKEN, which says the data stopped agreeing,
    #: and the distinction is the same one this file is about: explaining a
    #: relation and refuting it are opposite outcomes and had one verdict.
    RETIRED = "retired"


#: Verdicts that settle an entry.
#:
#: `EXTENDED` is absent deliberately. Surviving a widening is evidence, and
#: evidence is not an answer -- an entry that closed on EXTENDED would stop
#: being asked the question exactly when the answer started looking
#: interesting. Two of the three are refutations, which is all a wider range
#: can deliver; see the module docstring on why there is nothing here for
#: "solved". RETIRED is the exception and does not come from a range at all --
#: it is the noise registry's decision, carrying the reason somebody had to
#: write, which is exactly the mechanism the docstring names.
CLOSING = frozenset(
    {RevisitVerdict.CONTRADICTED, RevisitVerdict.BROKEN, RevisitVerdict.RETIRED}
)


class Revisit(BaseModel):
    """One examination of an open finding at a new range."""

    model_config = ConfigDict(extra="forbid")

    verdict: RevisitVerdict
    #: Rows the new scan examined. The size of the evidence, and the only thing
    #: that makes one revisit stronger than another.
    universe: int
    #: Rows where it was tight. Against the record's count, this is where a
    #: relation that kept holding but stopped growing shows up.
    tight: int
    #: Plain sentence saying what was compared and what came of it.
    note: str


class OpenFinding(BaseModel):
    """A relation tight on a set nothing named, and what later ranges said.

    Identity is the RELATION -- its columns, and its kind -- and never the
    tight set or the range. Keying on the tight set would make every widening a
    new entry, and whether last time's entry survived this time is the one
    question the ledger exists to answer.
    """

    model_config = ConfigDict(extra="forbid")

    kind: RegularityKind
    columns: list[str]
    statement: str
    #: Every row the scan examined, complete. Truncating this breaks the
    #: containment test silently: a short universe looks like a narrow range,
    #: so a real widening is judged INCOMPARABLE and a genuinely disjoint range
    #: is compared as though it were nested.
    universe: list[str] = Field(default_factory=list)
    #: The rows where it was tight, complete. Same reason -- the extension test
    #: compares this set exactly, and a truncated one fails it always.
    witnesses: list[str] = Field(default_factory=list)
    surprise: int = 0
    from_unrequested: bool = False
    #: The headline as first recorded, for a reader scanning the ledger.
    character: str = ""
    history: list[Revisit] = Field(default_factory=list)

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        """What makes this the same open question as a later scan's finding."""
        return (self.kind.value, _columns_key(self.columns))

    @property
    def verdict(self) -> RevisitVerdict | None:
        """The latest verdict, or None if never revisited."""
        return self.history[-1].verdict if self.history else None

    @property
    def open(self) -> bool:
        """Still an unanswered question."""
        return self.verdict not in CLOSING

    @property
    def widest(self) -> int:
        """The largest range this has been confirmed on.

        Taken from the history rather than the record, because the record's own
        universe is the range it was FIRST seen at and stays that way.
        """
        confirmed = [
            item.universe
            for item in self.history
            if item.verdict is RevisitVerdict.EXTENDED
        ]
        return max([len(self.universe), *confirmed])


def _columns_key(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(columns))


def _key(finding: PatternFinding) -> tuple[str, tuple[str, ...]]:
    """A finding's identity, in the same shape as `OpenFinding.key`.

    One definition, used by every lookup. Two spellings of "the same relation"
    would agree until the day they did not, and the ledger's whole content is
    whether this scan's finding is the one the last scan recorded.
    """
    return (finding.kind.value, _columns_key(finding.columns))


def _keys(finding: PatternFinding) -> set[tuple[str, tuple[str, ...]]]:
    """Every identity this finding answers to, including its aliases.

    A relation carried under two names is reported once, under one of them.
    The record made before that was true names the other -- and looking it up
    by name alone finds nothing, which is the BROKEN verdict. `abs_Mertens >=
    cum_mu` would have been REFUTED by a scan in which it holds exactly as
    well as it ever did, because a different name reached the report.

    So a finding answers to every name that carried the same values. This
    widens what counts as the recorded relation and cannot widen what counts
    as having been measured: `judge` settles that first, against the columns
    the scan examined.
    """
    spellings = [
        [name, *finding.aliases.get(name, [])] for name in finding.columns
    ]
    return {
        (finding.kind.value, _columns_key(combination))
        for combination in product(*spellings)
    }


def judge(
    entry: OpenFinding,
    findings: Sequence[PatternFinding],
    universe: Sequence[str],
    columns: Sequence[str],
    retired: Sequence[Retirement],
) -> Revisit:
    """What does this scan, over this range and these columns, say about it?

    `findings` is a WHOLE fresh scan rather than a hand-picked finding: the
    entry is looked up in it, and its absence is the BROKEN verdict. Passing in
    the single matching finding would make "it is gone" unrepresentable, and
    that is the verdict most worth being able to reach.

    `retired` is what the noise registry took OUT of that scan, and it is
    required for exactly the reason `columns` is. Absence is the BROKEN
    verdict, and a retired finding is absent -- so a caller that leaves this
    out has the ledger record "was a regularity of the narrower range" about a
    relation somebody just explained. Neither argument can be recovered from
    the surviving finding afterwards, which is why neither is a default.
    """
    was_universe = set(entry.universe)
    now_universe = {str(row) for row in universe}

    # RETIRED comes before everything, INCLUDING the range check. A reason
    # somebody wrote is not a measurement and does not depend on what this
    # scan happened to cover; reporting an explained relation as "nothing was
    # tested" would be as wrong as reporting it as refuted. It can only fire
    # when a matching finding was actually found and then retired, so nothing
    # unmeasured reaches it.
    explained = next(
        (item for item in retired if entry.key in _keys(item.finding)), None
    )
    if explained is not None:
        return Revisit(
            verdict=RevisitVerdict.RETIRED,
            universe=len(now_universe),
            tight=len(explained.finding.witnesses),
            note=(
                f"retired as {explained.ground.value}, not refuted -- the "
                f"relation holds and is explained: {explained.reason}"
            ),
        )

    # BOTH checks come BEFORE anything is concluded about the finding. A
    # relation missing from a scan that never covered the old range, or never
    # measured the old columns, has not been refuted; it has not been looked
    # at, and those two must not produce the same verdict.
    unmeasured = [name for name in entry.columns if name not in set(columns)]
    if unmeasured:
        return Revisit(
            verdict=RevisitVerdict.INCOMPARABLE,
            universe=len(now_universe),
            tight=0,
            note=(
                f"the new scan did not measure {', '.join(unmeasured)} -- "
                "nothing about this finding was tested"
            ),
        )

    if not (was_universe < now_universe):
        shared = len(was_universe & now_universe)
        return Revisit(
            verdict=RevisitVerdict.INCOMPARABLE,
            universe=len(now_universe),
            tight=0,
            note=(
                f"the new range ({len(now_universe)} rows) does not strictly "
                f"contain the recorded one ({len(was_universe)} rows, {shared} "
                "shared) -- nothing about this finding was tested"
            ),
        )

    here = [
        f
        for f in findings
        if any(key[1] == entry.key[1] for key in _keys(f))
    ]
    if not here:
        return Revisit(
            verdict=RevisitVerdict.BROKEN,
            universe=len(now_universe),
            tight=0,
            note=(
                f"over {len(now_universe)} rows nothing relates "
                f"{' and '.join(entry.columns)} any more: {entry.statement!r} "
                "was a regularity of the narrower range"
            ),
        )

    match = next((f for f in here if entry.key in _keys(f)), None)
    # Said out loud, because the reader is comparing a recorded statement
    # against a current one and the names in them differ. A verdict reached
    # under a name the record does not use, without saying so, is the same
    # trap as a merge that reports nothing.
    renamed = (
        ""
        if match is None or _key(match) == entry.key
        else (
            f" (reported as {' and '.join(match.columns)}, which carried the "
            f"same values as {' and '.join(entry.columns)} on every row)"
        )
    )
    if match is None:
        # The columns still relate, in a different shape. Not BROKEN -- the
        # relation is there -- and not a survival either, because a record that
        # says "tight on these rows, slack on those" is contradicted by any
        # reading of the same columns that does not reproduce that split.
        shapes = ", ".join(sorted({f.kind.value for f in here}))
        return Revisit(
            verdict=RevisitVerdict.CONTRADICTED,
            universe=len(now_universe),
            tight=0,
            note=(
                f"over {len(now_universe)} rows these columns relate as "
                f"{shapes}, not {entry.kind.value}; the recorded tight/slack "
                "split is not reproduced"
            ),
        )

    was_tight = set(entry.witnesses)
    now_tight = set(match.witnesses)
    restricted = now_tight & was_universe
    if restricted != was_tight:
        gained = sorted(restricted - was_tight)[:6]
        lost = sorted(was_tight - restricted)[:6]
        return Revisit(
            verdict=RevisitVerdict.CONTRADICTED,
            universe=len(now_universe),
            tight=len(now_tight),
            note=(
                "on the rows the record covers the new tight set differs: "
                f"{len(restricted - was_tight)} added {gained}, "
                f"{len(was_tight - restricted)} dropped {lost}{renamed}"
            ),
        )

    named = (
        f"; the wider scan names that set {match.characterised!r}, which a "
        "restriction-stable predicate cannot do without having named it here"
        if match.characterised
        else "; and still nothing names that set"
    )
    return Revisit(
        verdict=RevisitVerdict.EXTENDED,
        universe=len(now_universe),
        tight=len(now_tight),
        note=(
            f"held over {len(now_universe)} rows, agreeing on all "
            f"{len(was_universe)} the record covers; tight on "
            f"{len(now_tight)}{named}{renamed}"
        ),
    )


class OpenLedger(BaseModel):
    """The accumulated open questions."""

    model_config = ConfigDict(extra="forbid")

    entries: list[OpenFinding] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> OpenLedger:
        if not path.exists():
            return cls()
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="",
        )

    def find(self, finding: PatternFinding) -> OpenFinding | None:
        wanted = _keys(finding)
        return next((e for e in self.entries if e.key in wanted), None)

    def record(
        self, findings: Sequence[PatternFinding], universe: Sequence[str]
    ) -> list[OpenFinding]:
        """File the unexplained findings of a scan. Returns the NEW ones.

        Only `unexplained` findings are filed: a tight set was found and no
        predicate named it. A finding already in the ledger is left ALONE
        rather than refreshed -- the record is the first observation, and
        overwriting it with a later one destroys the only thing a revisit has
        to compare against.
        """
        added: list[OpenFinding] = []
        rows = [str(row) for row in universe]
        for finding in findings:
            if not finding.unexplained or self.find(finding) is not None:
                continue
            entry = OpenFinding(
                kind=finding.kind,
                columns=list(finding.columns),
                statement=finding.statement,
                universe=rows,
                witnesses=list(finding.witnesses),
                surprise=finding.surprise,
                from_unrequested=finding.from_unrequested,
                character=finding.character,
            )
            self.entries.append(entry)
            added.append(entry)
        return added

    def revisit(
        self,
        findings: Sequence[PatternFinding],
        universe: Sequence[str],
        columns: Sequence[str],
        retired: Sequence[Retirement],
    ) -> list[tuple[OpenFinding, Revisit]]:
        """Judge every still-open entry against a fresh scan.

        Closed entries are skipped rather than re-judged. An entry BROKEN over
        198 rows is not un-broken by a later scan that happens not to reach
        that far, and re-opening it on one would let a narrow run overturn a
        wide one.
        """
        judged: list[tuple[OpenFinding, Revisit]] = []
        for entry in self.entries:
            if not entry.open:
                continue
            outcome = judge(entry, findings, universe, columns, retired)
            entry.history.append(outcome)
            judged.append((entry, outcome))
        return judged
